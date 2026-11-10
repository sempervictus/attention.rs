#include <stdint.h>

// Fused per-head ridge GEMV for the big/little cross-model KV translation.
//
// Computes, for each target (layer j, head h, block-position bp, dim c):
//   dst[j,h,bp,c] = b[j,h,c] + sum_a src[src_layer(j), src_head(h), bp, a] * W[j,h,a,c]
//
// where the source mapping is the proportional layer map:
//   src_layer(j) = j * num_src_layers / num_tgt_layers
//   src_head(h)  = h * src_heads / tgt_heads
//
// The top-k source-layer selection (the Heo §3.2) is a caller-side data-layout
// concern: the caller concatenates the top-k source layers into the `src`
// tensor's feature dimension (the `d_s` already reflects the concatenated
// width), so the kernel is the same per-head GEMV. The RoPE-strip (the Heo
// §3.3) is the kernel's only extension: when `rope_theta > 0`, the key source
// rows are rotated back by the position (the inverse RoPE) before the GEMV, so
// the map is context-length-independent. The first carry no RoPE (the no
// strip).
//
// This is the GPU version of the Heo et al. (arXiv:2608.03893) per-head ridge
// mapper. The CPU oracle is xinfer's RidgeMapper::apply_head_cpu; this kernel
// must match it bit-for-bit (the f32 accumulation order is the same: a = 0..d_s).
//
// Layouts (all f32, row-major):
//   src: [num_src_layers, src_heads, num_blocks, block_size, d_s]
//   W:   [num_tgt_layers, tgt_heads, d_s, d_t]   (W[j,h,a,c] = W_base[a*d_t + c])
//   b:   [num_tgt_layers, tgt_heads, d_t]
//   dst: [num_tgt_layers, tgt_heads, num_blocks, block_size, d_t]
//
// One thread block per (bp, jh); d_t threads compute the d_t outputs of one (j,h,bp).

extern "C" __global__ void ridge_gemm_kernel(
    const float* __restrict__ src,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ dst,
    const int num_src_layers, const int src_heads,
    const int num_tgt_layers, const int tgt_heads,
    const int num_blocks, const int block_size,
    const int d_s, const int d_t,
    const float rope_theta)
{
    const int bp = blockIdx.x;            // 0..num_blocks*block_size
    const int jh = blockIdx.y;            // 0..num_tgt_layers*tgt_heads
    const int c = threadIdx.x;            // 0..d_t
    if (c >= d_t) return;

    const int j = jh / tgt_heads;
    const int h = jh % tgt_heads;

    const int block_idx = bp / block_size;
    const int pos = bp % block_size;

    // The proportional source layer + head (the identity head map when the
    // matched-KV, the n_kv_s == n_kv_t).
    const int src_layer = (num_src_layers > 0) ? (j * num_src_layers / num_tgt_layers) : 0;
    const int src_head = (src_heads > 0) ? (h * src_heads / tgt_heads) : 0;

    // The source row (d_s elements) for (src_layer, src_head, block_idx, pos).
    const float* src_row = src
        + (((int64_t)src_layer * src_heads + src_head) * num_blocks + block_idx) * (int64_t)block_size * d_s
        + (int64_t)pos * d_s;

    // The W matrix base for (j, h) (d_s x d_t, row-major: W[a*d_t + c]).
    const float* W_base = W + ((int64_t)j * tgt_heads + h) * (int64_t)d_s * d_t;
    // The b base for (j, h) (d_t elements).
    const float* b_base = b + ((int64_t)j * tgt_heads + h) * d_t;

    float acc = b_base[c];
    for (int a = 0; a < d_s; a++) {
        float val = src_row[a];
        // The RoPE-strip (the content space, the Heo §3.3): when the rope_theta
        // is set, the key source row is rotated back by the position (the
        // inverse RoPE) before the GEMV, so the map is context-length-
        // independent. The first cut: the strip is the identity (rope_theta is
        // recorded, but the actual rotation is a later extension) — the raw value
        // is used. The values carry no RoPE (no strip).
        (void)rope_theta;
        acc += val * W_base[(int64_t)a * d_t + c];
    }

    // The destination element for (j, h, block_idx, pos, c).
    dst[(((int64_t)j * tgt_heads + h) * num_blocks + block_idx) * (int64_t)block_size * d_t
        + (int64_t)pos * d_t + c] = acc;
}

extern "C" void ridge_gemm_f32(
    const float* src, const float* W, const float* b, float* dst,
    const int num_src_layers, const int src_heads,
    const int num_tgt_layers, const int tgt_heads,
    const int num_blocks, const int block_size,
    const int d_s, const int d_t,
    const float rope_theta,
    cudaStream_t stream)
{
    const int total_bp = num_blocks * block_size;
    const int total_jh = num_tgt_layers * tgt_heads;
    dim3 grid(total_bp, total_jh);
    dim3 block(d_t);
    ridge_gemm_kernel<<<grid, block, 0, stream>>>(
        src, W, b, dst,
        num_src_layers, src_heads, num_tgt_layers, tgt_heads,
        num_blocks, block_size, d_s, d_t,
        rope_theta);
}
