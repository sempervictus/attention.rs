//! Grammar-constrained speculative-decode sampling kernels.
//!
//! This module provides the fused rejection sampler (the Leviathan lossless
//! verify): a single kernel launch performs the min(1, p/q) accept test, the
//! Gumbel-max residual draw, and the all-accepted bonus draw, with an optional
//! per-position VOB mask constraining the draws to the grammar-legal set. The
//! sampler is grammar-agnostic: it consumes a VOB mask plus the target and draft
//! probability tensors, and tracks no grammar state of its own, so it is
//! independent of whatever produced the mask.

use candle_core::{DType, Result, Storage, Tensor};

use crate::kernels::ffi;


/// The fused speculative-decode rejection sampler (the Leviathan lossless
/// verify, the GPU path). A single kernel launch performs the min(1, p/q)
/// accept test, the Gumbel-max residual draw, and the all-accepted bonus draw,
/// with an optional per-position VOB mask that constrains the residual and
/// bonus draws to the grammar-legal set.
///
/// Layouts (single sequence; the MTP/DFlash verify uses batch size 1):
/// - `target_probs`: [k_max+1, vocab] F32. The target's post-sampling
///   distribution at each verify position: positions 0..k_max are the draft
///   positions, and position k_max is the all-accepted bonus draw. These are
///   already-normalized probabilities.
/// - `draft_probs`: [k_max+1, vocab] F32. The drafter's per-position softmax.
///   The bonus row is unused by the residual draw but must be allocated.
/// - `draft_tokens`: [k_max] U32. The proposed draft token at each position.
/// - `vob`: [k_max, words_per_vob] U32. The optional per-position grammar mask,
///   where a set bit means the token is allowed, over the k_max draft
///   positions. Pass `None` for the plain no-grammar path.
///
/// Returns `(num_accepted, continuation_token)`: the count of leading drafts
/// accepted, plus the token to commit at the first rejection (the residual
/// draw) or at the end (the bonus draw when all drafts are accepted). The
/// `seed` keys the per-row Philox RNG, making the rejection draws reproducible
/// and batch-independent.
///
/// Fail-closed contract: if the all-accepted bonus draw finds no legal token
/// (an empty allow-set at position k_max, a degenerate mask), the continuation
/// is u32::MAX and this function returns an error rather than silently
/// injecting token 0. The caller (the xinfer verify step) must treat that error
/// as a mask-correctness failure and abort.
#[cfg(feature = "cuda")]
pub fn fused_reject(
    target_probs: &Tensor,
    draft_probs: &Tensor,
    draft_tokens: &Tensor,
    vob: Option<&Tensor>,
    k_max: usize,
    vocab: usize,
    seed: u64,
) -> Result<(u32, u32)> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let dev = target_probs.device();
    let stream = *dev.as_cuda_device()?.cu_stream() as i64;
    let words_per_vob = (vocab + 31) / 32;

    // The output scalars (the num_accepted + the continuation), read back after
    // the launch.
    let out = Tensor::zeros((2,), DType::U32, &dev)?;

    fn ptr_f32(t: &Tensor) -> Result<*const f32> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<f32>()?.device_ptr() as *const f32),
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }
    fn ptr_u32(t: &Tensor) -> Result<*const u32> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<u32>()?.device_ptr() as *const u32),
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }
    fn ptr_u32_mut(t: &Tensor) -> Result<*mut u32> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<u32>()?.device_ptr() as *mut u32),
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    let p_target = ptr_f32(target_probs)?;
    let p_draft = ptr_f32(draft_probs)?;
    let p_drafts = ptr_u32(draft_tokens)?;
    let p_out = ptr_u32_mut(&out)?;
    let (p_vob, wpv) = match vob {
        Some(v) => (ptr_u32(v)?, words_per_vob as u32),
        None => (std::ptr::null(), 0u32),
    };

    unsafe {
        ffi::fused_reject(
            p_target,
            p_draft,
            p_drafts,
            p_out,
            p_out.add(1),
            p_vob,
            wpv,
            k_max as i32,
            vocab as i32,
            seed,
            1, // the single-seq verify (the batch=1)
            stream,
        );
    }
    dev.synchronize()?;
    let vals: Vec<u32> = out.flatten_all()?.to_vec1()?;
    let num_accepted = vals[0];
    let continuation = vals[1];
    // Fail-closed sentinel: the kernel writes UINT32_MAX when the all-accepted
    // bonus draw found no legal token (a degenerate empty allow-set at position
    // k_max). That means the grammar mask was not exact (it allowed nothing where
    // it should allow the continuation), so emitting a token here would be an
    // injection. Refuse instead.
    if continuation == u32::MAX {
        candle_core::bail!(
            "fused_reject: empty allow-set at the bonus position (num_accepted={num_accepted}); the grammar mask was not exact. Refusing to emit a token (R5 fail-closed)."
        );
    }
    Ok((num_accepted, continuation))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The fused rejection sampler (the Leviathan lossless verify, the GPU path):
    /// a fully accepted draft returns num_accepted == k_max; a rejected draft
    /// returns the count plus the residual-draw continuation. The VOB mask
    /// constrains the residual and bonus draws to the allowed set. These cases
    /// are deterministic (the p_target/p_draft ratios are 0 or 1, and the
    /// Gumbel-max over a single allowed token is that token), so the GPU result
    /// is exact.
    #[test]
    fn fused_reject_matches_cpu_semantics() {
        let dev = Device::new_cuda(0).unwrap();
        let vocab = 4;
        let k_max = 2;
        // The kernel reads k_max+1 rows: positions 0..k_max (the drafts) plus
        // position k_max (the all-accepted bonus draw). Allocate that many rows.
        let rows = k_max + 1;

        // Case 1: the draft is fully accepted (p_target >= p_draft at every draft
        // position). All rows are all-1.0, so the accept probability is 1.0 at both
        // positions; num_accepted == k_max, and the bonus draw picks from the all-1.0
        // row (any token, deterministic under the fixed seed).
        let target = Tensor::from_vec(vec![1.0f32; rows * vocab], (rows, vocab), &dev).unwrap();
        let draft = Tensor::from_vec(vec![1.0f32; rows * vocab], (rows, vocab), &dev).unwrap();
        let drafts = Tensor::from_vec(vec![0u32, 1], (k_max,), &dev).unwrap();
        let (num_accepted, _cont) =
            fused_reject(&target, &draft, &drafts, None, k_max, vocab, 1234).unwrap();
        assert_eq!(num_accepted, k_max as u32, "p_target == p_draft == 1.0 gives accept probability 1.0, so all drafts are accepted");

        // Case 2: the draft is rejected at position 0 (p_target(0)=0.0 <
        // p_draft(0)=1.0), but a different token has target mass. At position 0 the
        // target has mass on token 2 and the draft has mass on token 0, so the draft
        // token 0 is rejected and the residual is max(0, target - draft) = [0,0,1,0],
        // which picks token 2. The bonus row (index k_max) is all zero (not reached,
        // because the reject happens at position 0).
        let target2 = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 1.0, 0.0, // position 0: token 2 has mass 1.0
                1.0, 0.0, 0.0, 0.0, // position 1: token 0 has mass 1.0
                0.0, 0.0, 0.0, 0.0, // position k_max (bonus): all zero (not reached)
            ],
            (rows, vocab),
            &dev,
        )
        .unwrap();
        let draft2 = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, // position 0: token 0 has mass 1.0
                0.0, 0.0, 0.0, 1.0, // position 1: token 3 has mass 1.0
                0.0, 0.0, 0.0, 0.0, // position k_max (bonus)
            ],
            (rows, vocab),
            &dev,
        )
        .unwrap();
        let drafts2 = Tensor::from_vec(vec![0u32, 3], (k_max,), &dev).unwrap();
        let (num_acc2, cont2) =
            fused_reject(&target2, &draft2, &drafts2, None, k_max, vocab, 999).unwrap();
        assert_eq!(num_acc2, 0, "position 0 is rejected (p_target(0)=0 < p_draft(0)=1)");
        assert_eq!(cont2, 2, "the residual draw picks token 2 (the only target mass at position 0)");

        // Case 3: the VOB mask constrains the residual. Same target and draft as
        // case 2, but the VOB allows only token 3 at position 0 (bit 3 set). The
        // residual (token 2) is masked out, so the continuation falls back to the
        // draft token (0).
        let words_per_vob = (vocab + 31) / 32; // 1 word
        let vob = Tensor::from_vec(vec![1u32 << 3, u32::MAX], (k_max, words_per_vob), &dev).unwrap();
        let (num_acc3, cont3) =
            fused_reject(&target2, &draft2, &drafts2, Some(&vob), k_max, vocab, 999).unwrap();
        assert_eq!(num_acc3, 0, "position 0 is still rejected");
        assert_eq!(cont3, 0, "the residual token 2 is masked out by the VOB, so the continuation falls back to the draft token 0");
    }
}
