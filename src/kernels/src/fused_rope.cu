/**
 * @brief Fused Rotary Position Embedding (RoPE) CUDA Kernels - GQA Support
 * Copyright (c) 2025, Guoqing Bao. All rights reserved.
 *
 * This kernel performs fused rotary position embedding on Q and K tensors in-place:
 *  - fused_rope: Non-interleaved version (pairs at offset d/2)
 *  - fused_rope_i: Interleaved version (adjacent pairs)
 *
 * Supports Grouped Query Attention (GQA) where Q and K have different head counts.
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
__device__ __forceinline__ float2 rope_rotate_f32(float2 v, float c, float s) {
    float2 result;
    result.x = v.x * c - v.y * s;
    result.y = v.x * s + v.y * c;
    return result;
}

// --- __nv_bfloat162 native operations (BF16 computes natively) ---
#ifndef NO_BF16_KERNEL
__device__ __forceinline__ __nv_bfloat162 rope_rotate_bf16(__nv_bfloat162 v, __nv_bfloat16 c, __nv_bfloat16 s) {
    __nv_bfloat162 result;
    result.x = __hsub(__hmul(v.x, c), __hmul(v.y, s));
    result.y = __hadd(__hmul(v.x, s), __hmul(v.y, c));
    return result;
}
#endif

// --- __half2 operations (F16 converts to F32 for precision) ---
__device__ __forceinline__ __half2 rope_rotate_f16(__half2 v, __half c, __half s) {
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
// Single-tensor RoPE kernels (process Q or K independently)
// ============================================================================

/**
 * @brief F32 Interleaved kernel for single tensor
 */
__global__ void rope_i_f32_kernel(
    float2* __restrict__ x,  // Treat as float2 for vectorized access
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

    uint32_t rope_idx = idx % half_td;
    if (stride_b > 0) {
        uint32_t b_idx = (2 * idx) / stride_b;
        rope_idx += b_idx * half_td;
    }

    s_cos[threadIdx.x] = cos[rope_idx];
    s_sin[threadIdx.x] = sin[rope_idx];
    __syncthreads();

    float c = s_cos[threadIdx.x];
    float s = s_sin[threadIdx.x];

    float2 vec = x[idx];
    x[idx] = rope_rotate_f32(vec, c, s);
}

/**
 * @brief F16 Interleaved kernel for single tensor
 */
__global__ void rope_i_f16_kernel(
    __half2* __restrict__ x,
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

    __half2 vec = x[idx];
    x[idx] = rope_rotate_f16(vec, c, s);
}

#ifndef NO_BF16_KERNEL
/**
 * @brief BF16 Interleaved kernel for single tensor
 */
__global__ void rope_i_bf16_kernel(
    __nv_bfloat162* __restrict__ x,
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

    __nv_bfloat162 vec = x[idx];
    x[idx] = rope_rotate_bf16(vec, c, s);
}
#endif

// ============================================================================
// Non-Interleaved Single-tensor Kernels
// ============================================================================

/**
 * @brief F32 Non-interleaved kernel for single tensor
 */
__global__ void rope_f32_kernel(
    float* __restrict__ x,
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

    float x1 = x[i1];
    float x2 = x[i2];
    x[i1] = x1 * c - x2 * s;
    x[i2] = x1 * s + x2 * c;
}

/**
 * @brief F16 Non-interleaved kernel for single tensor
 */
__global__ void rope_f16_kernel(
    __half* __restrict__ x,
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

    s_cos[threadIdx.x] = __half2float(cos[i_cs]);
    s_sin[threadIdx.x] = __half2float(sin[i_cs]);
    __syncthreads();

    float c = s_cos[threadIdx.x];
    float s = s_sin[threadIdx.x];

    float x1 = __half2float(x[i1]);
    float x2 = __half2float(x[i2]);
    x[i1] = __float2half(x1 * c - x2 * s);
    x[i2] = __float2half(x1 * s + x2 * c);
}

#ifndef NO_BF16_KERNEL
/**
 * @brief BF16 Non-interleaved kernel for single tensor
 */
__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
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

    __nv_bfloat16 x1 = x[i1];
    __nv_bfloat16 x2 = x[i2];
    x[i1] = __hsub(__hmul(x1, c), __hmul(x2, s));
    x[i2] = __hadd(__hmul(x1, s), __hmul(x2, c));
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
// C-linkage wrapper functions for FFI (now process Q and K separately)
// ============================================================================

// Non-interleaved versions - process Q and K with separate bh values
extern "C" void fused_rope_f32(
    float* q, float* k,
    const float* cos, const float* sin,
    uint32_t q_bh, uint32_t k_bh,  // Separate batch*heads for Q and K
    uint32_t td, uint32_t d, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    
    // Process Q
    uint32_t q_num_pairs = (q_bh * td) / 2;
    dim3 q_grid = get_launch_config(q_num_pairs);
    rope_f32_kernel<<<q_grid, BLOCK_SIZE, 0, stream>>>(q, cos, sin, q_bh, td, d, stride_b);
    
    // Process K
    uint32_t k_num_pairs = (k_bh * td) / 2;
    dim3 k_grid = get_launch_config(k_num_pairs);
    rope_f32_kernel<<<k_grid, BLOCK_SIZE, 0, stream>>>(k, cos, sin, k_bh, td, d, stride_b);
}

extern "C" void fused_rope_f16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t td, uint32_t d, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    
    uint32_t q_num_pairs = (q_bh * td) / 2;
    dim3 q_grid = get_launch_config(q_num_pairs);
    rope_f16_kernel<<<q_grid, BLOCK_SIZE, 0, stream>>>(
        (__half*)q, (const __half*)cos, (const __half*)sin, q_bh, td, d, stride_b
    );
    
    uint32_t k_num_pairs = (k_bh * td) / 2;
    dim3 k_grid = get_launch_config(k_num_pairs);
    rope_f16_kernel<<<k_grid, BLOCK_SIZE, 0, stream>>>(
        (__half*)k, (const __half*)cos, (const __half*)sin, k_bh, td, d, stride_b
    );
}

#ifndef NO_BF16_KERNEL
extern "C" void fused_rope_bf16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t td, uint32_t d, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    
    uint32_t q_num_pairs = (q_bh * td) / 2;
    dim3 q_grid = get_launch_config(q_num_pairs);
    rope_bf16_kernel<<<q_grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)q, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, 
        q_bh, td, d, stride_b
    );
    
    uint32_t k_num_pairs = (k_bh * td) / 2;
    dim3 k_grid = get_launch_config(k_num_pairs);
    rope_bf16_kernel<<<k_grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)k, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, 
        k_bh, td, d, stride_b
    );
}
#endif

// Interleaved versions - process Q and K with separate bh values
extern "C" void fused_rope_i_f32(
    float* q, float* k,
    const float* cos, const float* sin,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t td, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_td = td / 2;
    
    // Process Q
    uint32_t q_num_pairs = (q_bh * td) / 2;
    dim3 q_grid = get_launch_config(q_num_pairs);
    rope_i_f32_kernel<<<q_grid, BLOCK_SIZE, 0, stream>>>(
        (float2*)q, cos, sin, q_num_pairs, half_td, stride_b
    );
    
    // Process K
    uint32_t k_num_pairs = (k_bh * td) / 2;
    dim3 k_grid = get_launch_config(k_num_pairs);
    rope_i_f32_kernel<<<k_grid, BLOCK_SIZE, 0, stream>>>(
        (float2*)k, cos, sin, k_num_pairs, half_td, stride_b
    );
}

extern "C" void fused_rope_i_f16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t td, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_td = td / 2;
    
    uint32_t q_num_pairs = (q_bh * td) / 2;
    dim3 q_grid = get_launch_config(q_num_pairs);
    rope_i_f16_kernel<<<q_grid, BLOCK_SIZE, 0, stream>>>(
        (__half2*)q, (const __half*)cos, (const __half*)sin, 
        q_num_pairs, half_td, stride_b
    );
    
    uint32_t k_num_pairs = (k_bh * td) / 2;
    dim3 k_grid = get_launch_config(k_num_pairs);
    rope_i_f16_kernel<<<k_grid, BLOCK_SIZE, 0, stream>>>(
        (__half2*)k, (const __half*)cos, (const __half*)sin, 
        k_num_pairs, half_td, stride_b
    );
}

#ifndef NO_BF16_KERNEL
extern "C" void fused_rope_i_bf16(
    void* q, void* k,
    const void* cos, const void* sin,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t td, uint32_t stride_b,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_td = td / 2;
    
    uint32_t q_num_pairs = (q_bh * td) / 2;
    dim3 q_grid = get_launch_config(q_num_pairs);
    rope_i_bf16_kernel<<<q_grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat162*)q, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, 
        q_num_pairs, half_td, stride_b
    );
    
    uint32_t k_num_pairs = (k_bh * td) / 2;
    dim3 k_grid = get_launch_config(k_num_pairs);
    rope_i_bf16_kernel<<<k_grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat162*)k, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, 
        k_num_pairs, half_td, stride_b
    );
}
#endif
