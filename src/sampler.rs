use candle_core::{Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal;

pub struct Sampler {}

impl Sampler {
    pub fn new() -> Self {
        Self {}
    }

    #[cfg(feature = "cuda")]
    fn sample_cuda(
        &self,
        logits: &Tensor,
        k: usize,
        p: f32,
        temperature: f32,
        seed: u64,
        token_pos: u64,
    ) -> Result<Vec<u32>> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;

        let (b, v) = logits.dims2()?;
        let dev = logits.device();

        // 1. Ensure logits are contiguous and on GPU
        let logits = if !logits.is_contiguous() {
            logits.contiguous()?
        } else {
            logits.clone()
        };

        let storage = logits.storage_and_layout().0;
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Sampler expects CUDA tensor"),
        };

        // 2. Alloc output buffer
        let out_tokens = unsafe { dev.alloc::<i32>(b) }.w()?;

        // 3. Get pointers
        let logits_ptr = match &cuda_storage.slice {
            CudaStorageSlice::F32(inp) => inp.device_ptr(),
            _ => candle_core::bail!("Sampler expects F32 logits"),
        };
        let out_ptr = out_tokens.device_ptr();
        let stream = *dev.cu_stream() as i64;

        let logits_ptr = *logits_ptr as *const core::ffi::c_void;
        let out_ptr = *out_ptr as *mut core::ffi::c_void;

        // 4. Call FFI
        unsafe {
            ffi::sampling_f32(
                logits_ptr as *const f32,
                out_ptr as *mut i32,
                b as i32,
                v as i32,
                k as i32,
                p,
                temperature as f64,
                seed,
                token_pos,
                stream,
            );
        }

        // 5. Copy back to host
        let mut host_out = vec![0i32; b];
        dev.dtoh_sync_copy_into(&out_tokens, &mut host_out).w()?;

        Ok(host_out.into_iter().map(|x| x as u32).collect())
    }

    #[cfg(feature = "metal")]
    pub fn sample(&self, _: &Tensor, _: usize, _: f32, _: f32, _: u64, _: u64) -> Result<Vec<u32>> {
        candle_core::bail!("Sampler requires CUDA or Metal device")
    }
}
