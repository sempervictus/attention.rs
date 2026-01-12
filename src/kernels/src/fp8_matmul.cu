#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "attention/dtype_fp8.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

__device__ __forceinline__ float get_scale(const float *__restrict__ scale,
                                int n, int k, int scale_stride,
                                int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;

  // Validate bounds before load
  if (sr >= scale_stride || sc >= (block_size_x + BLOCK_K - 1) / BLOCK_K) {
      return 0.0f; // or abort with debug print
  }

  return __ldg(&scale[sr * scale_stride + sc]);
}

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_kernel(const T *__restrict__ input,
                                 const uint8_t *__restrict__ weight,
                                 const float *__restrict__ weight_scale,
                                 T *__restrict__ output,
                                 int M,
                                 int N,
                                 int K,
                                 int scale_row_stride,
                                 int block_size_y,
                                 int block_size_x) {

    __shared__ float s_input[BLOCK_M][BLOCK_K + 4];
    __shared__ float s_weight[BLOCK_N][BLOCK_K + 4];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * BLOCK_M + ty;
    const int col = bx * BLOCK_N + tx;

    float acc = 0.0f;

    for (int k_tile=0; k_tile < K; k_tile += BLOCK_K) {
        // Determine if we can use uniform tile-scale path
        bool tile_scale_uniform =
            block_size_x >= BLOCK_K &&
            ((k_tile % block_size_x) + BLOCK_K <= block_size_x);

        int scale_k_idx_tile = tile_scale_uniform ? (k_tile / block_size_x) : 0;

        // Load input into shared memory
        for (int i=tx+ty*BLOCK_M; i<BLOCK_M*BLOCK_K; i+=BLOCK_N*BLOCK_M) {
            int lm = i / BLOCK_K;
            int lk = i % BLOCK_K;
            int gm = by * BLOCK_M + lm;
                int gk = k_tile + lk;

                if (gm < M && gk < K) {
                if constexpr(std::is_same_v<T, half>) {
                    s_input[lm][lk] = __half2float(__ldg(&input[gm * K + gk]));
            } else {
#ifndef NO_BF16_KERNEL
                    s_input[lm][lk] = __bfloat162float(__ldg(&input[gm * K + gk]));
#endif
                    }
            } else {
                s_input[lm][lk] = 0.0f;
            }
        }

    // Load weights into shared memory with scaling
        for (int i=tx+ty*BLOCK_N; i<BLOCK_N*BLOCK_K; i+=BLOCK_N*BLOCK_M) {
            int ln = i / BLOCK_K;
            int lk = i % BLOCK_K;
            int gn = bx * BLOCK_N + ln;
            int gk = k_tile + lk;

            if (gn >= N || gk >= K) {
                    s_weight[ln][lk] = 0.0f;
                    continue;
            }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

            if (tile_scale_uniform) {

                // Load 4 FP8 weights at once using uint32_t
                const uint32_t* w4_ptr = reinterpret_cast<const uint32_t*>(&weight[gn * K + lk]);

                // Get scale for this row/column group
                int scale_row = gn / block_size_y;
                float s = __ldg(&weight_scale[scale_row * scale_row_stride + scale_k_idx_tile]);

                if (lk+3 < K) {
                    // Vectorized conversion: fp8x4 -> Float4 with scaling
                    float4 w_vals = vllm::fp8::scaled_vec_conversion<Float4, uint32_t>(*w4_ptr, s);

                    // Store into shared memory (if within bounds)
                    if (lk+0 < BLOCK_K) s_weight[ln][lk+0] = w_vals.x;
                    if (lk+1 < BLOCK_K) s_weight[ln][lk+1] = w_vals.y;
                    if (lk+2 < BLOCK_K) s_weight[ln][lk+2] = w_vals.z;
                    if (lk+3 < BLOCK_K) s_weight[ln][lk+3] = w_vals.w;
                }
            } else {

#endif

fallback_scalar_weight_load:

                uint8_t w_raw = __ldg(&weight[gn * K + gk]);
                float s = 0.0f;

                if (tile_scale_uniform) {
                    int scale_row = gn / block_size_y;
                    s = __ldg(&weight_scale[scale_row * scale_row_stride + scale_k_idx_tile]);
                } else {
                    s = get_scale(weight_scale, gn, gk, scale_row_stride, block_size_y,
                                                block_size_x);
                }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Use native FP8 intrinsic for single element
                float f_val = __half2float(__nv_cvt_fp8_to_halfraw(w_raw, __NV_E4M3));
#else
    // Software fallback using dispatch_fp8_to_float
                float f_val = vllm::fp8::dispatch_fp8_to_float(w_raw);
#endif

                s_weight[ln][lk] = f_val * s;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
            }
#endif

        } // end weight load loop

        __syncthreads();

        if (row < M && col < N) {
#pragma unroll
            for (int k=0; k<BLOCK_K; k+=2) {
                float2 in2 = *reinterpret_cast<float2*>(&s_input[ty][k]);
                float2 w2 = *reinterpret_cast<float2*>(&s_weight[tx][k]);
                acc = fmaf(in2.x, w2.x, acc);
                acc = fmaf(in2.y, w2.y, acc);
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if constexpr(std::is_same_v<T, half>) {
            output[row * N + col] = __float2half(acc);
        } else {
#ifndef NO_BF16_KERNEL
            output[row * N + col] = __float2bfloat16(acc);
#endif
        }
    }
}

extern "C" void fp8_matmul_f16(const __half *input,
                                      const uint8_t *weight,
                                      const float *weight_scale,
                                      __half *output,
                                      int M, int N, int K,
                                      int scale_row_stride,
                                      int block_size_y,
                                      int block_size_x,
                                      cudaStream_t stream) {
  constexpr int TILE = 32;
  constexpr int TILE_K = 32;

  dim3 block(TILE, TILE);
  dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

  fp8_matmul_kernel<half, TILE, TILE, TILE_K>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                   scale_row_stride, block_size_y,
                                   block_size_x);
}

extern "C" void fp8_matmul_bf16(const __nv_bfloat16 *input,
                       const uint8_t *weight,
                       const float *weight_scale,
                       __nv_bfloat16 *output,
                       int M,
                       int N,
                       int K,
                       int scale_row_stride,
                       int block_size_y,
                       int block_size_x,
                       cudaStream_t stream) {
  constexpr int TILE = 32;
  constexpr int TILE_K = 32;

  dim3 block(TILE, TILE);
  dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));
#ifndef NO_BF16_KERNEL
  fp8_matmul_kernel<__nv_bfloat16, TILE, TILE, TILE_K>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                   scale_row_stride, block_size_y,
                                   block_size_x);
#endif
}

