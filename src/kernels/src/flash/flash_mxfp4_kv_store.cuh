/**
 * MXFP4 KV cache store: FP4 E2M1 + UE8M0 group-32 scales.
 *
 * SimplQuant paper format (arXiv 2606.20474):
 *   K: sign_flip -> WHT -> per-group-32 absmax -> UE8M0 -> FP4 E2M1
 *   V: per-group-32 absmax -> UE8M0 -> FP4 E2M1 (no rotation)
 *
 * Simplified v1: per-lane absmax (no group shuffle) to avoid
 * CUDA_ERROR_ILLEGAL_INSTRUCTION on SM120. Group-32 reduction
 * via __shfl_xor_sync is added in v2 after verifying basic kernel.
 */

#include "flash_sm_compat.cuh"

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

#define MXFP4_GROUP 32
#define MXFP4_GROUPS (HDIM / MXFP4_GROUP)
#define MXFP4_VEC (HDIM / WARP_SIZE)
#define MXFP4_LANES_PER_GROUP (WARP_SIZE / MXFP4_VEC)

#ifndef MXFP4_HELPERS_DEFINED
#define MXFP4_HELPERS_DEFINED

__device__ __forceinline__ uint8_t mxfp4_float_to_e2m1(float val) {
    float abs_val = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t code;
    if (abs_val < 0.25f)       code = 0x0;
    else if (abs_val < 0.75f) code = 0x1;
    else if (abs_val < 1.25f) code = 0x2;
    else if (abs_val < 1.75f) code = 0x3;
    else if (abs_val < 2.5f)  code = 0x4;
    else if (abs_val < 3.5f)  code = 0x5;
    else if (abs_val < 5.0f)  code = 0x6;
    else                      code = 0x7;
    return sign | code;
}

__device__ __forceinline__ float mxfp4_e2m1_to_float(uint8_t code) {
    static const float vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = vals[code & 0x7];
    return (code & 0x8) ? -v : v;
}

__device__ __forceinline__ uint8_t float_to_ue8m0(float val, float c) {
    float scaled = c * val;
    if (scaled <= 0.0f) return 0;
    int exp;
    frexpf(scaled, &exp);
    int log2r = (exp >= 1) ? exp : exp + 1;
    int byte_val = log2r + 127;
    if (byte_val < 0) byte_val = 0;
    if (byte_val > 255) byte_val = 255;
    return (uint8_t)byte_val;
}

__device__ __forceinline__ float ue8m0_to_float(uint8_t byte) {
    int exp = (int)byte - 127;
    if (exp < -126) return 0.0f;
    if (exp > 127) return 3.4028235e38f;
    return ldexpf(1.0f, exp);
}

#endif // MXFP4_HELPERS_DEFINED

template<typename HalfT>
__global__ void flash_mxfp4_kv_store(
    const HalfT* __restrict__ K,
    const HalfT* __restrict__ V,
    unsigned char* K_fp4,
    unsigned char* K_sf,
    unsigned char* V_fp4,
    unsigned char* V_sf,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float c_k,
    const float c_v,
    const bool rotate
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)(slot / block_size);
    unsigned int block_off = (unsigned int)(slot % block_size);

    unsigned int base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    unsigned long long fp4_k_off = (unsigned long long)block_idx * block_size * num_kv_heads * (head_dim / 2)
        + (unsigned long long)block_off * num_kv_heads * (head_dim / 2)
        + (unsigned long long)head_idx * (head_dim / 2);
    unsigned long long sf_k_off = (unsigned long long)block_idx * block_size * num_kv_heads * MXFP4_GROUPS
        + (unsigned long long)block_off * num_kv_heads * MXFP4_GROUPS
        + (unsigned long long)head_idx * MXFP4_GROUPS;

    // K: load, optionally rotate
    float k_reg[MXFP4_VEC];
    #pragma unroll
    for (int i = 0; i < MXFP4_VEC; i++) {
        unsigned int ch = lane_id * MXFP4_VEC + i;
        k_reg[i] = FLASH_TO_FLOAT(K[base + ch]);
        if (rotate) {
            k_reg[i] *= get_sign_flip(head_idx, ch);
        }
    }
    if (rotate) {
        wht_transform(k_reg, lane_id);
    }

    // Per-group-32 quantization for K (per-lane absmax, no shuffle)
    #pragma unroll
    for (int g = 0; g < MXFP4_GROUPS; g++) {
        unsigned int lane_start = g * MXFP4_LANES_PER_GROUP;
        if (lane_id < lane_start || lane_id >= lane_start + MXFP4_LANES_PER_GROUP) continue;

        int local = lane_id - lane_start;
        int elem_base = local * MXFP4_VEC;

        // Per-lane absmax (approximation: each lane computes its own max)
        float local_max = 0.f;
        #pragma unroll
        for (int i = 0; i < MXFP4_VEC; i++)
            local_max = fmaxf(local_max, fabsf(k_reg[i]));

        // Lane 0 of each group writes the SF byte
        if (local == 0) {
            uint8_t sf_byte = float_to_ue8m0(local_max, c_k);
            K_sf[sf_k_off + g] = sf_byte;
        }

        // All lanes in the group quantize using their local_max
        // (approximation: each lane uses its own max, not the group max)
        float scale = ue8m0_to_float(float_to_ue8m0(local_max, c_k));
        float inv_scale = (scale > 0.f) ? (1.0f / scale) : 0.f;

        #pragma unroll
        for (int i = 0; i < MXFP4_VEC; i += 2) {
            uint8_t lo = mxfp4_float_to_e2m1(k_reg[i] * inv_scale);
            uint8_t hi = mxfp4_float_to_e2m1(k_reg[i+1] * inv_scale);
            unsigned int byte_idx = (elem_base + i) / 2;
            K_fp4[fp4_k_off + byte_idx] = (hi << 4) | (lo & 0xF);
        }
    }

    // V: load, per-group-32 quantization (no rotation)
    float v_reg[MXFP4_VEC];
    #pragma unroll
    for (int i = 0; i < MXFP4_VEC; i++) {
        unsigned int ch = lane_id * MXFP4_VEC + i;
        v_reg[i] = FLASH_TO_FLOAT(V[base + ch]);
    }

    unsigned long long fp4_v_off = fp4_k_off;
    unsigned long long sf_v_off = sf_k_off;

    #pragma unroll
    for (int g = 0; g < MXFP4_GROUPS; g++) {
        unsigned int lane_start = g * MXFP4_LANES_PER_GROUP;
        if (lane_id < lane_start || lane_id >= lane_start + MXFP4_LANES_PER_GROUP) continue;

        int local = lane_id - lane_start;
        int elem_base = local * MXFP4_VEC;

        float local_max = 0.f;
        #pragma unroll
        for (int i = 0; i < MXFP4_VEC; i++)
            local_max = fmaxf(local_max, fabsf(v_reg[i]));

        if (local == 0) {
            uint8_t sf_byte = float_to_ue8m0(local_max, c_v);
            V_sf[sf_v_off + g] = sf_byte;
        }

        float scale = ue8m0_to_float(float_to_ue8m0(local_max, c_v));
        float inv_scale = (scale > 0.f) ? (1.0f / scale) : 0.f;

        #pragma unroll
        for (int i = 0; i < MXFP4_VEC; i += 2) {
            uint8_t lo = mxfp4_float_to_e2m1(v_reg[i] * inv_scale);
            uint8_t hi = mxfp4_float_to_e2m1(v_reg[i+1] * inv_scale);
            unsigned int byte_idx = (elem_base + i) / 2;
            V_fp4[fp4_v_off + byte_idx] = (hi << 4) | (lo & 0xF);
        }
    }
}