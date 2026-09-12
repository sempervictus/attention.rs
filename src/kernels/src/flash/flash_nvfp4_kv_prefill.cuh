/**
 * NVFP4 paged KV prefill: software dequant (E2M1 LUT * E4M3 scale) + paged attention.
 *
 * Analogous to call_flash_prefill_paged_fp8 but reads NVFP4 paged cache.
 * Handless ragged query (multiple tokens per sequence) attending to paged KV.
 *
 * Paged layout (matching flash_nvfp4_kv_store):
 *   K_fp4: [num_blocks, block_size, num_kv_heads, head_dim/2] U8
 *   K_sf:  [num_blocks, block_size, num_kv_heads, head_dim/16] U8 (E4M3)
 *   V_fp4, V_sf: same shapes
 *
 * Query is WHT-rotated on-the-fly (if rotate=true) to match rotated K in cache.
 * V is unrotated.
 *
 * Grid: (num_q_tiles, num_kv_heads, num_seqs)
 * Block: 128 threads (4 warps)
 * Each warp handles a chunk of the KV sequence (online softmax).
 */

#include "flash_sm_compat.cuh"
// wht_transform, get_sign_flip from flash_turboquant.cuh
// nvfp4_e2m1_to_float, e4m3_to_float_direct from flash_nvfp4_kv_store.cuh

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

#define NVDEC_VEC (HDIM / WARP_SIZE)
#define NVDEC_LANES_PER_GROUP (16 / NVDEC_VEC)

// Prefill tile: 32 query tokens per tile (fits in 99 KB SMEM on SM120)
#define NVFP4_PREFILL_TILE 32
#define NVFP4_PREFILL_WARPS 4
#define NVFP4_PREFILL_THREADS (NVFP4_PREFILL_WARPS * WARP_SIZE)

template<typename HalfT>
__global__ void flash_nvfp4_kv_prefill(
    const HalfT* __restrict__ Q,
    const unsigned char* __restrict__ K_fp4,
    const unsigned char* __restrict__ K_sf,
    const unsigned char* __restrict__ V_fp4,
    const unsigned char* __restrict__ V_sf,
    HalfT* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_seqs,
    const unsigned int q_stride,
    const unsigned int kv_stride,
    const float softcap,
    const unsigned int sliding_window,
    const bool rotate
) {
    const unsigned int q_tile = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    const unsigned int q_head = kv_head * (num_q_heads / num_kv_heads) + (tid / (NVFP4_PREFILL_TILE * (num_q_heads / num_kv_heads)));
    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    // This warp's chunk of the KV sequence
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NVFP4_PREFILL_WARPS - 1) / NVFP4_PREFILL_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Load Q for this tile, apply WHT rotation
    const unsigned int q_base = q_tile * NVFP4_PREFILL_TILE;
    const unsigned int q_vec_off = lane_id * NVDEC_VEC;
    float q_reg[NVDEC_VEC];
    {
        const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                           + (unsigned long long)q_head * head_dim + q_base * head_dim + q_vec_off);
        #pragma unroll
        for (int i = 0; i < NVDEC_VEC / 2; i++) {
            unsigned int packed = __ldg(q32 + i);
            const HalfT* hp = reinterpret_cast<const HalfT*>(&packed);
            q_reg[2*i]   = FLASH_TO_FLOAT(hp[0]);
            q_reg[2*i+1] = FLASH_TO_FLOAT(hp[1]);
        }
        if (rotate) {
            #pragma unroll
            for (int i = 0; i < NVDEC_VEC; i++) {
                unsigned int ch = lane_id * NVDEC_VEC + i;
                q_reg[i] *= get_sign_flip(kv_head, ch);
            }
            wht_transform(q_reg, lane_id);
        }
    }

    // Paged strides
    const unsigned int hd_half = head_dim / 2;
    const unsigned int hd_groups = head_dim / 16;
    const unsigned long long fp4_page = (unsigned long long)block_size * num_kv_heads * hd_half;
    const unsigned long long sf_page  = (unsigned long long)block_size * num_kv_heads * hd_groups;
    const unsigned int lane_fp4_off = lane_id * (NVDEC_VEC / 2);
    const unsigned int lane_sf_off  = lane_id / NVDEC_LANES_PER_GROUP;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[NVDEC_VEC];
    #pragma unroll
    for (int i = 0; i < NVDEC_VEC; i++) o_reg[i] = 0.f;

    // Iterate over paged KV
    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        const unsigned char* k_fp4_base = K_fp4 + (unsigned long long)physical_block * fp4_page
                                         + (unsigned long long)kv_head * hd_half;
        const unsigned char* k_sf_base  = K_sf  + (unsigned long long)physical_block * sf_page
                                         + (unsigned long long)kv_head * hd_groups;
        const unsigned char* v_fp4_base = V_fp4 + (unsigned long long)physical_block * fp4_page
                                         + (unsigned long long)kv_head * hd_half;
        const unsigned char* v_sf_base  = V_sf  + (unsigned long long)physical_block * sf_page
                                         + (unsigned long long)kv_head * hd_groups;

        for (unsigned int i = 0; i < batch_count; i++) {
            unsigned int bo = block_offset + i;
            const unsigned char* kp = k_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
            const unsigned char* ks = k_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;

            // Dequant K: 2 bytes FP4 + 1 byte E4M3 scale
            float k_scale = e4m3_to_float_direct(*ks);
            unsigned char p0 = *kp;
            unsigned char p1 = *(kp + 1);
            float k0 = nvfp4_e2m1_to_float(p0 & 0xF) * k_scale;
            float k1 = nvfp4_e2m1_to_float((p0 >> 4) & 0xF) * k_scale;
            float k2 = nvfp4_e2m1_to_float(p1 & 0xF) * k_scale;
            float k3 = nvfp4_e2m1_to_float((p1 >> 4) & 0xF) * k_scale;

            float dot = q_reg[0]*k0 + q_reg[1]*k1 + q_reg[2]*k2 + q_reg[3]*k3;
            #pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);

            float score = dot * inv_sqrt_d;
            if (softcap > 0.f) score = softcap * tanhf(score / softcap);

            float m_new = fmaxf(m_val, score);
            float exp_old = __expf(m_val - m_new), exp_new = __expf(score - m_new);
            l_val = l_val * exp_old + exp_new;
            #pragma unroll
            for (int j = 0; j < NVDEC_VEC; j++) o_reg[j] *= exp_old;

            // Dequant V and accumulate
            const unsigned char* vp = v_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
            const unsigned char* vs = v_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;
            float v_scale = e4m3_to_float_direct(*vs);
            float w = exp_new * v_scale;
            unsigned char vp0 = *vp;
            unsigned char vp1 = *(vp + 1);
            o_reg[0] += w * nvfp4_e2m1_to_float(vp0 & 0xF);
            o_reg[1] += w * nvfp4_e2m1_to_float((vp0 >> 4) & 0xF);
            o_reg[2] += w * nvfp4_e2m1_to_float(vp1 & 0xF);
            o_reg[3] += w * nvfp4_e2m1_to_float((vp1 >> 4) & 0xF);

            m_val = m_new;
        }
        pos += batch_count;
    }

    // Inter-warp reduction
    __shared__ float smem_m[NVFP4_PREFILL_WARPS];
    __shared__ float smem_l[NVFP4_PREFILL_WARPS];
    __shared__ float smem_o[NVFP4_PREFILL_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < NVDEC_VEC; i++) smem_o[warp_id][q_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = NVFP4_PREFILL_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float mn = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - mn), scale_w = __expf(mw - mn);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = mn;
                #pragma unroll
                for (int i = 0; i < NVDEC_VEC; i++)
                    smem_o[warp_id][q_vec_off + i] =
                        smem_o[warp_id][q_vec_off + i] * scale_me +
                        smem_o[other][q_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.f) ? (1.f / final_l) : 0.f;
        HalfT* o_ptr = O + (unsigned long long)seq_idx * q_stride
                       + (unsigned long long)q_head * head_dim
                       + (unsigned long long)q_base * head_dim
                       + q_vec_off;
        #pragma unroll
        for (int i = 0; i < NVDEC_VEC; i++) {
            o_ptr[i] = FLASH_FROM_FLOAT(smem_o[0][q_vec_off + i] * inv_l);
        }
    }
}