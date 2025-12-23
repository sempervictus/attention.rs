#include "gpu_sampling.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>

static __device__ __forceinline__ uint4 philox4x32_10(uint2 key, uint4 ctr) {
  const uint32_t M0 = 0xD2511F53u;
  const uint32_t M1 = 0xCD9E8D57u;
  const uint32_t W0 = 0x9E3779B9u;
  const uint32_t W1 = 0xBB67AE85u;

  uint32_t k0 = key.x, k1 = key.y;
  #pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint64_t p0 = (uint64_t)M0 * (uint64_t)ctr.x;
    uint64_t p1 = (uint64_t)M1 * (uint64_t)ctr.z;

    uint32_t hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo0 = (uint32_t)(p0 & 0xffffffffu);
    uint32_t hi1 = (uint32_t)(p1 >> 32);
    uint32_t lo1 = (uint32_t)(p1 & 0xffffffffu);

    uint32_t nx = hi1 ^ ctr.y ^ k0;
    uint32_t ny = lo1;
    uint32_t nz = hi0 ^ ctr.w ^ k1;
    uint32_t nw = lo0;

    ctr = make_uint4(nx, ny, nz, nw);
    k0 += W0; k1 += W1;
  }
  return ctr;
}

static __device__ __forceinline__ float u01_from_u32(uint32_t x) {
  return ((float)x + 0.5f) * (1.0f / 4294967296.0f);
}

template<int K>
static __device__ __forceinline__ void merge_topk_desc_safe(
    const float* aV, const int* aI, // sorted desc, len K
    const float* bV, const int* bI, // sorted desc, len K
    float* outV, int* outI          // sorted desc, len K
) {
  int ia = 0, ib = 0;
  #pragma unroll
  for (int t = 0; t < K; ++t) {
    float va = (ia < K) ? aV[ia] : -CUDART_INF_F;
    float vb = (ib < K) ? bV[ib] : -CUDART_INF_F;
    bool take_b = (vb > va);
    outV[t] = take_b ? vb : va;
    outI[t] = take_b ? bI[ib] : aI[ia];
    if (take_b) ib++; else ia++;
  }
}

// ---------------- Stage A: tiled local topK per (batch, tile) ----------------
//
// One block handles one tile of the vocab for one batch element.
// It sorts a CHUNK of logits within the block (descending) and writes top-K.
template<int K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void stageA_local_topk(
    const float* __restrict__ logits, // [B,V]
    int B, int V,
    float temperature,
    int tiles_per_row,                // number of tiles along vocab
    float* __restrict__ out_vals,      // [B, tiles, K]
    int* __restrict__ out_idx          // [B, tiles, K]
) {
  int b = blockIdx.x;
  int tile = blockIdx.y;
  if (b >= B) return;

  constexpr int CHUNK = BLOCK_THREADS * ITEMS_PER_THREAD;
  int base = tile * CHUNK;

  const float* row = logits + (size_t)b * (size_t)V;

  float invT = (temperature > 1e-6f) ? (1.0f / temperature) : 1e6f;

  float v[ITEMS_PER_THREAD];
  int   i[ITEMS_PER_THREAD];

  #pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    int idx = base + threadIdx.x + it * BLOCK_THREADS;
    if (idx < V) {
      v[it] = row[idx] * invT;
      i[it] = idx;
    } else {
      v[it] = -CUDART_INF_F;
      i[it] = -1;
    }
  }

  using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockSort::TempStorage temp;
  BlockSort(temp).SortDescending(v, i);
  __syncthreads();

  // Write out the first K ranks from the sorted block.
  // Rank = threadIdx.x + it*BLOCK_THREADS
  int out_base = (b * tiles_per_row + tile) * K;

  #pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    int rank = threadIdx.x + it * BLOCK_THREADS;
    if (rank < K) {
      out_vals[out_base + rank] = v[it];
      out_idx[out_base + rank]  = i[it];
    }
  }
}

// ---------------- Stage B: reduce tiles -> global topK -> softmax -> top-p -> sample ----------------
//
// One block per batch element.
// Merges (tiles_per_row) lists of K candidates into one global K.
// Then samples.
template<int K, int BLOCK_THREADS>
__global__ void stageB_reduce_and_sample(
    int B,
    int tiles_per_row,
    const float* __restrict__ tile_vals, // [B, tiles, K] sorted desc per tile
    const int* __restrict__ tile_idx,    // [B, tiles, K]
    float top_p,
    uint64_t seed,
    uint64_t token_pos,
    int* __restrict__ out_tokens         // [B]
) {
  int b = blockIdx.x;
  if (b >= B) return;
  int tid = threadIdx.x;

  __shared__ float topv[K];
  __shared__ int   topi[K];
  __shared__ float tmpv[K];
  __shared__ int   tmpi[K];

  // init with -inf
  for (int t = tid; t < K; t += BLOCK_THREADS) {
    topv[t] = -CUDART_INF_F;
    topi[t] = -1;
  }
  __syncthreads();

  // Sequentially merge each tile's topK into running topK.
  // K is small (32/64/128), tiles_per_row is modest; this is fast in practice.
  for (int tile = 0; tile < tiles_per_row; ++tile) {
    const int base = (b * tiles_per_row + tile) * K;

    // Load tile candidates into tmp (shared)
    for (int t = tid; t < K; t += BLOCK_THREADS) {
      tmpv[t] = tile_vals[base + t];
      tmpi[t] = tile_idx [base + t];
    }
    __syncthreads();

    if (tid == 0) {
      float outV[K];
      int   outI[K];
      merge_topk_desc_safe<K>(topv, topi, tmpv, tmpi, outV, outI);
      #pragma unroll
      for (int t = 0; t < K; ++t) { topv[t] = outV[t]; topi[t] = outI[t]; }
    }
    __syncthreads();
  }

  // Softmax over topK (stable). Use one thread (K is small) for robustness.
  __shared__ float probs[K];
  if (tid == 0) {
    float mx = topv[0];
    #pragma unroll
    for (int t = 1; t < K; ++t) mx = fmaxf(mx, topv[t]);

    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
      float e = __expf(topv[t] - mx);
      probs[t] = e;
      sum += e;
    }
    sum = fmaxf(sum, 1e-20f);
    #pragma unroll
    for (int t = 0; t < K; ++t) probs[t] /= sum;

    // top-p within top-k
    int cutoff = K;
    if (top_p > 0.0f && top_p < 1.0f) {
      float cum = 0.0f;
      for (int t = 0; t < K; ++t) {
        cum += probs[t];
        if (cum >= top_p) { cutoff = t + 1; break; }
      }
    }

    float psum = 0.0f;
    for (int t = 0; t < cutoff; ++t) psum += probs[t];
    psum = fmaxf(psum, 1e-20f);

    // Deterministic RNG: per (b, token_pos)
    uint2 key = make_uint2((uint32_t)(seed ^ (0x9E3779B97f4A7C15ULL + (uint64_t)b)),
                           (uint32_t)((seed >> 32) + 0xD1B54A32D192ED03ULL));
    uint4 ctr = make_uint4((uint32_t)token_pos, (uint32_t)(token_pos >> 32),
                           (uint32_t)b, 0x12345678u);
    uint4 r = philox4x32_10(key, ctr);
    float u = u01_from_u32(r.x);

    float acc = 0.0f;
    int picked = topi[0]; // fallback
    for (int t = 0; t < cutoff; ++t) {
      acc += probs[t] / psum;
      if (u <= acc) { picked = topi[t]; break; }
    }
    out_tokens[b] = picked;
  }
}

template<int K>
void gpu_topk_topp_sample(
    const float* logits_d,
    int* out_tokens_d,
    const SamplerParams& p,
    cudaStream_t stream
) {
  // Best-practice defaults:
  // - 256 threads is a good balance for scanning vocab.
  // - ITEMS_PER_THREAD controls chunk size. 4 is safe; 8 may be faster on some GPUs.
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int CHUNK = BLOCK_THREADS * ITEMS_PER_THREAD;

  // tiles along vocab; ensure at least 1
  int tiles = (p.V + CHUNK - 1) / CHUNK;

  // Allocate intermediate buffers: [B, tiles, K]
  // In production, allocate once and reuse.
  // For now, consistent with user request, we allocate here.
  float* tile_vals_d = nullptr;
  int*   tile_idx_d  = nullptr;

  size_t vals_bytes = (size_t)p.B * (size_t)tiles * (size_t)K * sizeof(float);
  size_t idx_bytes  = (size_t)p.B * (size_t)tiles * (size_t)K * sizeof(int);

  CUDA_CHECK(cudaMallocAsync(&tile_vals_d, vals_bytes, stream));
  CUDA_CHECK(cudaMallocAsync(&tile_idx_d,  idx_bytes,  stream));

  // Stage A: grid = (B, tiles)
  dim3 gridA(p.B, tiles, 1);
  dim3 blockA(BLOCK_THREADS, 1, 1);
  stageA_local_topk<K, BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<gridA, blockA, 0, stream>>>(
          logits_d, p.B, p.V, p.temperature, tiles, tile_vals_d, tile_idx_d);

  // Stage B: one block per batch element
  dim3 gridB(p.B, 1, 1);
  dim3 blockB(256, 1, 1);
  stageB_reduce_and_sample<K, 256>
      <<<gridB, blockB, 0, stream>>>(
          p.B, tiles, tile_vals_d, tile_idx_d, p.top_p, p.seed, p.token_pos, out_tokens_d);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(tile_vals_d, stream));
  CUDA_CHECK(cudaFreeAsync(tile_idx_d,  stream));
}

// Explicit instantiations
template void gpu_topk_topp_sample<32>(const float*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<64>(const float*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<128>(const float*, int*, const SamplerParams&, cudaStream_t);


extern "C" void sampling_f32(
    const float* logits_d,
    int* out_tokens_d,
    int B,
    int V,
    int K,
    float temperature,
    float top_p,
    uint64_t seed,
    uint64_t token_pos,
    int64_t stream_ptr) 
{
    SamplerParams p;
    p.B = B;
    p.V = V;
    p.temperature = temperature;
    p.top_p = top_p;
    p.seed = seed;
    p.token_pos = token_pos;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    if (K <= 32) {
        gpu_topk_topp_sample<32>(logits_d, out_tokens_d, p, stream);
    } else if (K <= 64) {
        gpu_topk_topp_sample<64>(logits_d, out_tokens_d, p, stream);
    } else {
        // Fallback or max supported
        gpu_topk_topp_sample<128>(logits_d, out_tokens_d, p, stream);
    }
}
