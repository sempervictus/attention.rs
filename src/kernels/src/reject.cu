// Grammar-constrained speculative-decode sampling kernel.
//
// The fused rejection sampler (the Leviathan lossless verify): one launch does
// the min(1, p/q) accept test, the Gumbel-max residual draw, and the all-accepted
// bonus draw, with an optional per-position VOB mask constraining the draws to
// the grammar-legal set. This is the only kernel in this file.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>

__global__ void fused_reject_kernel(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    const uint32_t* __restrict__ draft_tokens,
    uint32_t* __restrict__ out_num_accepted,
    uint32_t* __restrict__ out_continuation,
    const uint32_t* __restrict__ vob,
    uint32_t words_per_vob,
    int k_max,
    int vocab,
    uint64_t seed,
    int batch)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed ^ 0x9E3779B97F4A7C15ull, (unsigned)seq, 0, &rng);

    const uint32_t* drafts = draft_tokens + (size_t)seq * k_max;
    uint32_t num_accepted = 0;
    uint32_t continuation = 0u;

    for (int i = 0; i < k_max; i++) {
        int draft_tok = (int)drafts[i];
        float p_target = target_probs[(size_t)i * vocab + draft_tok];
        float p_draft  = draft_probs[(size_t)i * vocab + draft_tok];
        bool legal = true;
        if (vob != nullptr) {
            unsigned w = (unsigned)(draft_tok / 32);
            if (w < words_per_vob &&
                ((vob[(size_t)i * words_per_vob + w] >> (draft_tok & 31)) & 1u) == 0u)
                legal = false;
        }
        float accept_prob;
        if (!legal) accept_prob=0.0f;
        else if (p_draft > 0.0f) accept_prob=fminf(1.0f, p_target / p_draft);
        else accept_prob=(p_target > 0.0f)? 1.0f : 0.0f;

        if (curand_uniform(&rng)<accept_prob){ num_accepted++; continue;}

        // Reject at position i: Gumbel-max over the VOB-masked residual.
        curandStatePhilox4_32_10_t grng;
        curand_init(seed ^ 0x9E3779B97F4A7C15ull, (unsigned)(seq*31u + (unsigned)i), 0, &grng);
        float best_g = -FLT_MAX; int best_t = draft_tok; bool found=false;
        for (int t=0; t<vocab; t++) {
            float residual=fmaxf(0.0f, target_probs[(size_t)i*vocab+t]-draft_probs[(size_t)i*vocab+t]);
            if (residual<=0.0f) continue;
            if (vob!=nullptr) {
                unsigned w=(unsigned)(t/32);
                if (w>=words_per_vob || ((vob[(size_t)i*words_per_vob+w]>>(t&31))&1u)==0u) continue;
            }
            float u=curand_uniform(&grng);
            float g=logf(fmaxf(residual,1e-30f))-logf(-logf(fmaxf(u,1e-30f)));
            if (g>best_g){ best_g=g; best_t=t; found=true; }
        }
        continuation = found ? (uint32_t)best_t : (uint32_t)draft_tok;
        break;
    }

    // All K accepted: bonus draw from the target at position k_max.
    if (num_accepted == (uint32_t)k_max) {
        curandStatePhilox4_32_10_t brng;
        curand_init(seed ^ 0xD1B54A32D192ED03ull, (unsigned)(seq*17u), 0, &brng);
        float best_g = -FLT_MAX; int best_t = -1; bool found = false;
        for (int t=0; t<vocab; t++) {
            float p_target = target_probs[(size_t)k_max*vocab+t];
            if (p_target<=0.0f) continue;
            if (vob!=nullptr) {
                unsigned w=(unsigned)(t/32);
                if (w>=words_per_vob || ((vob[(size_t)k_max*words_per_vob+w]>>(t&31))&1u)==0u) continue;
            }
            float u=curand_uniform(&brng);
            float g=logf(fmaxf(p_target,1e-30f))-logf(-logf(fmaxf(u,1e-30f)));
            if (g>best_g){ best_g=g; best_t=t; found=true; }
        }
        // Fail-closed sentinel: if no legal bonus token exists (an empty allow-set at
        // position k_max, a degenerate mask), write UINT32_MAX so the host rejects
        // it, rather than silently injecting token 0. A valid bonus is always in
        // 0..vocab.
        continuation = found ? (uint32_t)best_t : UINT32_MAX;
    }

    out_num_accepted[seq]=num_accepted;
    out_continuation[seq]=continuation;
}

extern "C" {

void fused_reject(
    const float* target_probs,
    const float* draft_probs,
    const uint32_t* draft_tokens,
    uint32_t* out_num_accepted,
    uint32_t* out_continuation,
    const uint32_t* vob,
    uint32_t words_per_vob,
    int k_max, int vocab,
    uint64_t seed, int batch, int64_t stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    fused_reject_kernel<<<blocks, threads, 0, s>>>(
        target_probs, draft_probs, draft_tokens,
        out_num_accepted, out_continuation, vob,
        words_per_vob, k_max, vocab, seed, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[fused_reject] launch error: %s\n", cudaGetErrorString(err));
}

} // extern "C"