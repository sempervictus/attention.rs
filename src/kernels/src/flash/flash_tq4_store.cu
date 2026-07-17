// attention.rs/src/kernels/src/flash/flash_tq4_store.cu
// TurboQuant 4-bit store kernel for KV cache quantization

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// TurboQuant 4-bit store kernel
__global__ void flash_tq4_store_kernel(
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
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * num_kv_heads * head_dim;
    
    if (idx >= total_elements) return;
    
    int token_idx = idx / (num_kv_heads * head_dim);
    int head_idx = (idx / head_dim) % num_kv_heads;
    int dim_idx = idx % head_dim;
    
    float k_val = K[idx];
    float v_val = V[idx];
    
    // Find block and offset within block
    int block_idx = token_idx / block_size;
    int block_offset = token_idx % block_size;
    
    // Quantize K to 4-bit
    uint8_t k_quant = quantize_tq4(k_val);
    K_quant[idx] = k_quant;
    
    // Quantize V to 4-bit
    uint8_t v_quant = quantize_tq4(v_val);
    V_quant[idx] = v_quant;
    
    // Store absmax for dequantization
    int absmax_idx = token_idx * num_kv_heads + head_idx;
    K_absmax[absmax_idx] = fmaxf(K_absmax[absmax_idx], fabsf(k_val));
    V_absmax[absmax_idx] = fmaxf(V_absmax[absmax_idx], fabsf(v_val));
}

// TQ4 quantization function (4-bit symmetric)
__device__ __forceinline__ uint8_t quantize_tq4(float val) {
    // Scale to [-7.5, 7.5] range for 4-bit
    float scaled = val * 8.0f;
    int quant_val = __float2int_rn(scaled);
    quant_val = __max(__min(quant_val, 7), -8);
    // Convert to unsigned 4-bit (0-15)
    return (uint8_t)(quant_val + 8);
}

// TQ4 dequantization function
__device__ __forceinline__ float dequant_tq4(uint8_t quant_val, float absmax) {
    // Convert from unsigned 4-bit (0-15) to signed (-8 to 7)
    int signed_val = (int)quant_val - 8;
    // Scale back to original range
    return (float)signed_val * absmax / 8.0f;
}

// Kernel launcher
extern "C" void launch_flash_tq4_store(
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
    int block_size,
    cudaStream_t stream
) {
    int total_elements = num_tokens * num_kv_heads * head_dim;
    int num_blocks = (total_elements + 255) / 256;
    
    flash_tq4_store_kernel<<<num_blocks, 256, 0, stream>>>(
        K, V, K_quant, V_quant, K_absmax, V_absmax,
        slot_mapping, num_tokens, num_kv_heads, head_dim, block_size
    );
}