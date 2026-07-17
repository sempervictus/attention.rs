// attention.rs/src/kernels/src/flash/flash_tq_store_k8v4.cu
// TurboQuant K8V4 store kernel for KV cache quantization

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// TurboQuant K8V4 store kernel
// K: [num_tokens, num_kv_heads, head_dim] float32
// V: [num_tokens, num_kv_heads, head_dim] float32
// K_quant: [num_tokens, num_kv_heads, head_dim] uint8 (8-bit K)
// V_quant: [num_tokens, num_kv_heads, head_dim] uint8 (4-bit V)
// K_absmax: [num_tokens, num_kv_heads] float32 (absmax scales for K)
// V_absmax: [num_tokens, num_kv_heads] float32 (absmax scales for V)
__global__ void flash_tq_store_k8v4_kernel(
    const float* __restrict__ K,
    const float* __restrict__ V,
    uint8_t* __restrict__ K_quant,
    uint8_t* __restrict__ V_quant,
    float* __restrict__ K_absmax,
    float* __restrict__ V_absmax,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    float* k_scale,
    float* v_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * num_kv_heads * head_dim;
    
    if (idx >= total_elements) return;
    
    int token_idx = idx / (num_kv_heads * head_dim);
    int head_idx = (idx / head_dim) % num_kv_heads;
    int dim_idx = idx % head_dim;
    
    float k_val = K[idx];
    float v_val = V[idx];
    
    // Quantize K to 8-bit
    uint8_t k_quant = quantize_k8(k_val, k_scale[head_idx]);
    K_quant[idx] = k_quant;
    
    // Quantize V to 4-bit
    uint8_t v_quant = quantize_v4(v_val, v_scale[head_idx]);
    V_quant[idx] = v_quant;
    
    // Store absmax for dequantization
    K_absmax[token_idx * num_kv_heads + head_idx] = fabsf(k_val);
    V_absmax[token_idx * num_kv_heads + head_idx] = fabsf(v_val);
}

// K8 quantization function (8-bit)
__device__ __forceinline__ uint8_t quantize_k8(float val, float scale) {
    if (scale == 0.0f) scale = 1.0f;
    int quant_val = __float2int_rn(val / scale);
    return __min(__max(quant_val, 0), 255);
}

// V4 quantization function (4-bit)
__device__ __forceinline__ uint8_t quantize_v4(float val, float scale) {
    if (scale == 0.0f) scale = 1.0f;
    int quant_val = __float2int_rn(val / scale);
    return __min(__max(quant_val, 0), 15);
}

// Dequantization functions
__device__ __forceinline__ float dequant_k8(uint8_t quant_val, float scale) {
    return quant_val * scale;
}

__device__ __forceinline__ float dequant_v4(uint8_t quant_val, float scale) {
    return quant_val * scale;
}

// Kernel launcher
extern "C" void launch_flash_tq_store_k8v4(
    const float* K,
    const float* V,
    uint8_t* K_quant,
    uint8_t* V_quant,
    float* K_absmax,
    float* V_absmax,
    const int* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    float* k_scale,
    float* v_scale,
    cudaStream_t stream
) {
    int total_elements = num_tokens * num_kv_heads * head_dim;
    int num_blocks = (total_elements + 255) / 256;
    
    flash_tq_store_k8v4_kernel<<<num_blocks, 256, 0, stream>>>(
        K, V, K_quant, V_quant, K_absmax, V_absmax,
        slot_mapping, num_tokens, num_kv_heads, head_dim,
        k_scale, v_scale
    );
}