/**
 * @brief Fused Rotary Position Embedding (RoPE) CUDA Kernels - Highly Optimized
 * Copyright (c) 2025, Guoqing Bao. All rights reserved.
 *
 * This kernel performs fused rotary position embedding on Q and K tensors in-place:
 *  - fused_rope: Non-interleaved version (pairs at offset d/2)
 *  - fused_rope_i: Interleaved version (adjacent pairs)
 *
 * Optimizations:
 *  - Vectorized load/store AND compute (half2, bfloat162, float2)
 *  - Native compute: F32 and BF16 compute natively, only F16 converts to F32
 *  - Shared memory for cos/sin caching to reduce global memory traffic
 *  - Single kernel launch for both Q and K (reduces launch overhead)
 *  - In-place operation (reduces memory allocation/bandwidth)
 *
 * Licensed under the Apache License, Version 2.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Block configuration
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Vectorized Types and Operations for Native Compute
// ============================================================================

// --- float2 native operations (F32 computes natively) ---
__device__ __forceinline__ float2 make_float2_from(float x, float y) {
    return make_float2(x, y);
}

__device__ __forceinline__ float2 rope_rotate_f32(float2 v, float c, float s) {
    // Native float compute
    float2 result;
    result.x = v.x * c - v.y * s;
    result.y = v.x * s + v.y * c;
    return result;
}

// --- __nv_bfloat162 native operations (BF16 computes natively) ---
#ifndef NO_BF16_KERNEL
__device__ __forceinline__ __nv_bfloat162 rope_rotate_bf16(__nv_bfloat162 v, __nv_bfloat16 c, __nv_bfloat16 s) {
    // Native bfloat16 compute using intrinsics
    __nv_bfloat162 c2 = __bfloat162bfloat162(c);
    __nv_bfloat162 s2 = __bfloat162bfloat162(s);
    
    // v.x * c - v.y * s, v.x * s + v.y * c
    __nv_bfloat162 vx = __bfloat162bfloat162(v.x);  // (v.x, v.x)
    __nv_bfloat162 vy = __bfloat162bfloat162(v.y);  // (v.y, v.y)
    
    // Compute: (v.x * c, v.x * s)
    __nv_bfloat162 cs = {c, s};
    __nv_bfloat162 term1 = __hmul2(vx, cs);
    
    // Compute: (v.y * s, v.y * c)
    __nv_bfloat162 sc = {s, c};
    __nv_bfloat162 term2 = __hmul2(vy, sc);
    
    // result.x = term1.x - term2.x, result.y = term1.y + term2.y
    __nv_bfloat162 result;
    result.x = __hsub(term1.x, term2.x);
    result.y = __hadd(term1.y, term2.y);
    
    return result;
}
#endif

// --- __half2 operations (F16 converts to F32 for precision) ---
__device__ __forceinline__ __half2 rope_rotate_f16(__half2 v, __half c, __half s) {
    // Convert to float for precise computation
    float vx = __half2float(v.x);
    float vy = __half2float(v.y);
    float fc = __half2float(c);
    float fs = __half2float(s);
    
    __half2 result;
    result.x = __float2half(vx * fc - vy * fs);
    result.y = __float2half(vx * fs + vy * fc);
    return result;
}

// ============================================================================
// Interleaved Fused RoPE - Vectorized Load/Store AND Compute
// ============================================================================

/**
 * @brief F32 Interleaved kernel - fully native vectorized compute
 */
__global__ void fused_rope_i_f32_kernel(
    float2* __restrict__ q,  // Treat as float2 for vectorized access
    float2* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const uint32_t num_pairs,  // (bh * td) / 2
    const uint32_t half_td,
    const uint32_t stride_b
) {
    __shared__ float s_cos[BLOCK_SIZE];
    __shared__ float s_sin[BLOCK_SIZE];

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    // Calculate cos/sin index
    uint32_t rope_idx = idx % half_td;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        rope_idx += b_idx * half_td;
    }

    // Load cos/sin into shared memory
    s_cos[threadIdx.x] = cos[rope_idx];
    s_sin[threadIdx.x] = sin[rope_idx];
    __syncthreads();

    float c = s_cos[threadIdx.x];
    float s = s_sin[threadIdx.x];

    // Vectorized load, native compute, vectorized store for Q
    float2 q_vec = q[idx];
    q[idx] = rope_rotate_f32(q_vec, c, s);

    // Same for K
    float2 k_vec = k[idx];
    k[idx] = rope_rotate_f32(k_vec, c, s);
}

/**
 * @brief F16 Interleaved kernel - vectorized load/store, float compute for precision
 */
__global__ void fused_rope_i_f16_kernel(
    __half2* __restrict__ q,
    __half2* __restrict__ k,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    const uint32_t num_pairs,
    const uint32_t half_td,
    const uint32_t stride_b
) {
    __shared__ __half s_cos[BLOCK_SIZE];
    __shared__ __half s_sin[BLOCK_SIZE];

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint32_t rope_idx = idx % half_td;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        rope_idx += b_idx * half_td;
    }

    s_cos[threadIdx.x] = cos[rope_idx];
    s_sin[threadIdx.x] = sin[rope_idx];
    __syncthreads();

    __half c = s_cos[threadIdx.x];
    __half s = s_sin[threadIdx.x];

    // Vectorized load, F32 compute (for precision), vectorized store
    __half2 q_vec = q[idx];
    q[idx] = rope_rotate_f16(q_vec, c, s);

    __half2 k_vec = k[idx];
    k[idx] = rope_rotate_f16(k_vec, c, s);
}

#ifndef NO_BF16_KERNEL
/**
 * @brief BF16 Interleaved kernel - fully native vectorized compute
 */
__global__ void fused_rope_i_bf16_kernel(
    __nv_bfloat162* __restrict__ q,
    __nv_bfloat162* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const uint32_t num_pairs,
    const uint32_t half_td,
    const uint32_t stride_b
) {
    __shared__ __nv_bfloat16 s_cos[BLOCK_SIZE];
    __shared__ __nv_bfloat16 s_sin[BLOCK_SIZE];

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint32_t rope_idx = idx % half_td;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        rope_idx += b_idx * half_td;
    }

    s_cos[threadIdx.x] = cos[rope_idx];
    s_sin[threadIdx.x] = sin[rope_idx];
    __syncthreads();

    __nv_bfloat16 c = s_cos[threadIdx.x];
    __nv_bfloat16 s = s_sin[threadIdx.x];

    // Vectorized load, native BF16 compute, vectorized store
    __nv_bfloat162 q_vec = q[idx];
    q[idx] = rope_rotate_bf16(q_vec, c, s);

    __nv_bfloat162 k_vec = k[idx];
    k[idx] = rope_rotate_bf16(k_vec, c, s);
}
#endif

// ============================================================================
// Non-Interleaved Fused RoPE - Native Compute with Shared Memory
// ============================================================================

/**
 * @brief F32 Non-interleaved kernel - native float compute
 */
__global__ void fused_rope_f32_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const uint32_t bh,
    const uint32_t td,
    const uint32_t d,
    const uint32_t stride_b
) {
    __shared__ float s_cos[BLOCK_SIZE];
    __shared__ float s_sin[BLOCK_SIZE];

    const uint32_t total_pairs = (bh * td) / 2;
    const uint32_t half_d = d / 2;
    const uint32_t half_td = td / 2;
    
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    // Calculate indices for non-interleaved layout
    uint32_t i_bh = idx / half_td;
    uint32_t i_td = idx - half_td * i_bh;
    uint32_t i_t = i_td / half_d;
    uint32_t i_d = i_td - half_d * i_t;
    
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + half_d;
    
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        i_cs += b_idx * half_td;
    }

    s_cos[threadIdx.x] = cos[i_cs];
    s_sin[threadIdx.x] = sin[i_cs];
    __syncthreads();

    float c = s_cos[threadIdx.x];
    float s = s_sin[threadIdx.x];

    // Native float compute for Q
    float q1 = q[i1];
    float q2 = q[i2];
    q[i1] = q1 * c - q2 * s;
    q[i2] = q1 * s + q2 * c;

    // Native float compute for K
    float k1 = k[i1];
    float k2 = k[i2];
    k[i1] = k1 * c - k2 * s;
    k[i2] = k1 * s + k2 * c;
}

/**
 * @brief F16 Non-interleaved kernel - F32 compute for precision
 */
__global__ void fused_rope_f16_kernel(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    const uint32_t bh,
    const uint32_t td,
    const uint32_t d,
    const uint32_t stride_b
) {
    __shared__ float s_cos[BLOCK_SIZE];
    __shared__ float s_sin[BLOCK_SIZE];

    const uint32_t total_pairs = (bh * td) / 2;
    const uint32_t half_d = d / 2;
    const uint32_t half_td = td / 2;
    
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    uint32_t i_bh = idx / half_td;
    uint32_t i_td = idx - half_td * i_bh;
    uint32_t i_t = i_td / half_d;
    uint32_t i_d = i_td - half_d * i_t;
    
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + half_d;
    
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        i_cs += b_idx * half_td;
    }

    // Convert cos/sin to F32 once
    s_cos[threadIdx.x] = __half2float(cos[i_cs]);
    s_sin[threadIdx.x] = __half2float(sin[i_cs]);
    __syncthreads();

    float c = s_cos[threadIdx.x];
    float s = s_sin[threadIdx.x];

    // F32 compute for precision, then convert back
    float q1 = __half2float(q[i1]);
    float q2 = __half2float(q[i2]);
    q[i1] = __float2half(q1 * c - q2 * s);
    q[i2] = __float2half(q1 * s + q2 * c);

    float k1 = __half2float(k[i1]);
    float k2 = __half2float(k[i2]);
    k[i1] = __float2half(k1 * c - k2 * s);
    k[i2] = __float2half(k1 * s + k2 * c);
}

#ifndef NO_BF16_KERNEL
/**
 * @brief BF16 Non-interleaved kernel - native BF16 compute
 */
__global__ void fused_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const uint32_t bh,
    const uint32_t td,
    const uint32_t d,
    const uint32_t stride_b
) {
    __shared__ __nv_bfloat16 s_cos[BLOCK_SIZE];
    __shared__ __nv_bfloat16 s_sin[BLOCK_SIZE];

    const uint32_t total_pairs = (bh * td) / 2;
    const uint32_t half_d = d / 2;
    const uint32_t half_td = td / 2;
    
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    uint32_t i_bh = idx / half_td;
    uint32_t i_td = idx - half_td * i_bh;
    uint32_t i_t = i_td / half_d;
    uint32_t i_d = i_td - half_d * i_t;
    
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + half_d;
    
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        i_cs += b_idx * half_td;
    }

    s_cos[threadIdx.x] = cos[i_cs];
    s_sin[threadIdx.x] = sin[i_cs];
    __syncthreads();

    __nv_bfloat16 c = s_cos[threadIdx.x];
    __nv_bfloat16 s = s_sin[threadIdx.x];

    // Native BF16 compute using intrinsics
    __nv_bfloat16 q1 = q[i1];
    __nv_bfloat16 q2 = q[i2];
    q[i1] = __hsub(__hmul(q1, c), __hmul(q2, s));
    q[i2] = __hadd(__hmul(q1, s), __hmul(q2, c));

    __nv_bfloat16 k1 = k[i1];
    __nv_bfloat16 k2 = k[i2];
    k[i1] = __hsub(__hmul(k1, c), __hmul(k2, s));
    k[i2] = __hadd(__hmul(k1, s), __hmul(k2, c));
}
#endif

// ============================================================================
// Launch configuration helper
// ============================================================================

inline dim3 get_launch_config(uint32_t num_elements) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return dim3(num_blocks, 1, 1);
}

// ============================================================================
// C-linkage wrapper functions for FFI
// ============================================================================

// Non-interleaved versions
extern "C" void fused_rope_f32(
    float* q, float* k,
    const float* cos, const float* sin,
    uint32_t bh, uint32_t td, uint32_t d, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t num_pairs = (bh * td) / 2;
    dim3 grid = get_launch_config(num_pairs);
    fused_rope_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(q, k, cos, sin, bh, td, d, stride_b);
}

extern "C" void fused_rope_f16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t bh, uint32_t td, uint32_t d, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t num_pairs = (bh * td) / 2;
    dim3 grid = get_launch_config(num_pairs);
    fused_rope_f16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__half*)q, (__half*)k, (const __half*)cos, (const __half*)sin, 
        bh, td, d, stride_b
    );
}

#ifndef NO_BF16_KERNEL
extern "C" void fused_rope_bf16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t bh, uint32_t td, uint32_t d, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t num_pairs = (bh * td) / 2;
    dim3 grid = get_launch_config(num_pairs);
    fused_rope_bf16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)q, (__nv_bfloat16*)k, 
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, 
        bh, td, d, stride_b
    );
}
#endif

// Interleaved versions - use vectorized types (float2, half2, bfloat162)
extern "C" void fused_rope_i_f32(
    float* q, float* k,
    const float* cos, const float* sin,
    uint32_t bh, uint32_t td, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t num_pairs = (bh * td) / 2;
    uint32_t half_td = td / 2;
    dim3 grid = get_launch_config(num_pairs);
    fused_rope_i_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (float2*)q, (float2*)k, cos, sin, num_pairs, half_td, stride_b
    );
}

extern "C" void fused_rope_i_f16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t bh, uint32_t td, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t num_pairs = (bh * td) / 2;
    uint32_t half_td = td / 2;
    dim3 grid = get_launch_config(num_pairs);
    fused_rope_i_f16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__half2*)q, (__half2*)k, (const __half*)cos, (const __half*)sin, 
        num_pairs, half_td, stride_b
    );
}

#ifndef NO_BF16_KERNEL
extern "C" void fused_rope_i_bf16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t bh, uint32_t td, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t num_pairs = (bh * td) / 2;
    uint32_t half_td = td / 2;
    dim3 grid = get_launch_config(num_pairs);
    fused_rope_i_bf16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat162*)q, (__nv_bfloat162*)k, 
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, 
        num_pairs, half_td, stride_b
    );
}
#endif
