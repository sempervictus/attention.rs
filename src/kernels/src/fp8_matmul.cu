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
  return __ldg(&scale[sr * scale_stride + sc]);
}

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_kernel(const T *__restrict__ input,
                                 const uint8_t *__restrict__ weight,
                                 const float *__restrict__ weight_scale,
                                 T *__restrict__ output, int M, int N, int K,
                                 int scale_row_stride, int block_size_y,
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

  const int num_threads = BLOCK_M * BLOCK_N;
  const int tid = ty * BLOCK_N + tx;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    const bool tile_scale_uniform =
        block_size_x >= BLOCK_K &&
        ((k_tile % block_size_x) + BLOCK_K <= block_size_x);
    const int scale_k_idx_tile = tile_scale_uniform ? (k_tile / block_size_x) : 0;

    for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
      int lm = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gm = by * BLOCK_M + lm;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[gm * K + gk]));
        } else {
#ifndef NO_BF16_KERNEL
          val = __bfloat162float(__ldg(&input[gm * K + gk]));
#endif
        }
      }
      s_input[lm][lk] = val;
    }

    for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
      int ln = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gn < N && gk < K) {
        uint8_t w_raw = __ldg(&weight[gn * K + gk]);
        float s = 0.0f;
        if (tile_scale_uniform) {
          int scale_row = gn / block_size_y;
          s = __ldg(&weight_scale[scale_row * scale_row_stride + scale_k_idx_tile]);
        } else {
          s = get_scale(weight_scale, gn, gk, scale_row_stride, block_size_y,
                        block_size_x);
        }
        val = vllm::fp8::dispatch_fp8_to_float(w_raw) * s;
      }
      s_weight[ln][lk] = val;
    }

    __syncthreads();

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_K; k += 2) {
        float2 in2 = *reinterpret_cast<float2*>(&s_input[ty][k]);
        float2 w2 = *reinterpret_cast<float2*>(&s_weight[tx][k]);
        acc = fmaf(in2.x, w2.x, acc);
        acc = fmaf(in2.y, w2.y, acc);
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    if constexpr (std::is_same_v<T, half>) {
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
                                      const float *weight_scale, __half *output,
                                      int M, int N, int K, int scale_row_stride,
                                      int block_size_y, int block_size_x,
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

extern "C" void fp8_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                       const float *weight_scale, __nv_bfloat16 *output, int M,
                       int N, int K, int scale_row_stride, int block_size_y,
                       int block_size_x, cudaStream_t stream) {
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
