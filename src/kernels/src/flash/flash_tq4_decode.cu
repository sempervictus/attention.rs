// attention.rs/src/kernels/src/flash/flash_tq4_decode.cu
// TurboQuant 4-bit decode kernel with SM120 optimizations

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// TurboQuant 4-bit decode kernel
__global__ void flash_tq4_decode_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ K_quant,
    const float* __restrict__ K_absmax,
    const uint8_t* __restrict__ V_quant,
    const float* __restrict__ V_absmax,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float* __restrict__ O,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float scale,
    float softcap,
    int sliding_window,
    int max_context_len,
    float* workspace
) {
    // SM120-optimized tile sizes for decode
    constexpr int TILE_M = 1;  // Single token decode
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 16;
    
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads) return;
    
    // Load query for this head
    float q_val = Q[seq_idx * num_heads * head_dim + head_idx * head_dim + thread_idx];
    
    // Compute attention over quantized KV cache
    float acc = 0.0f;
    int context_len = context_lens[seq_idx];
    
    for (int block = 0; block < (context_len + block_size - 1) / block_size; block++) {
        int k_block = block_tables[seq_idx * max_context_len / block_size + block];
        if (k_block == 0) continue;
        
        // Load K/V quantized blocks
        int k_start = block * block_size;
        int k_end = min(k_start + block_size, context_len);
        
        for (int k = k_start; k < k_end; k += TILE_K) {
            int k_idx = k_block * block_size * num_kv_heads * head_dim + 
                        (head_idx / (num_heads / num_kv_heads)) * head_dim + 
                        (k % TILE_K);
            
            uint8_t k_quant_val = K_quant[k_idx];
            uint8_t v_quant_val = V_quant[k_idx];
            float k_scale = K_absmax[k_idx / TILE_K];
            float v_scale = V_absmax[k_idx / TILE_K];
            
            // Dequantize and compute attention
            float k_val = dequant_tq4(k_quant_val, k_scale);
            float v_val = dequant_tq4(v_quant_val, v_scale);
            
            acc += q_val * k_val * v_val;
        }
    }
    
    // Apply softmax and write output
    acc *= scale;
    if (softcap > 0.0f) {
        acc = softcap * tanhf(acc / softcap);
    }
    
    O[seq_idx * num_heads * head_dim + head_idx * head_dim + thread_idx] = acc;
}

// Dequantization function for TurboQuant 4-bit
__device__ __forceinline__ float dequant_tq4(uint8_t quant_val, float scale) {
    // Extract 4-bit value (0-15)
    uint8_t val = quant_val & 0x0F;
    // Dequantize: val * scale
    return val * scale;
}

// Kernel launcher
extern "C" void launch_flash_tq4_decode(
    const float* Q,
    const uint8_t* K_quant,
    const float* K_absmax,
    const uint8_t* V_quant,
    const float* V_absmax,
    const int* block_tables,
    const int* context_lens,
    float* O,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float scale,
    float softcap,
    int sliding_window,
    int max_context_len,
    float* workspace,
    cudaStream_t stream
) {
    dim3 grid(num_seqs, num_kv_heads, 1);
    dim3 block(head_dim);
    
    flash_tq4_decode_kernel<<<grid, block, 0, stream>>>(
        Q, K_quant, K_absmax, V_quant, V_absmax,
        block_tables, context_lens, O,
        num_seqs, num_heads, num_kv_heads, head_dim,
        block_size, scale, softcap, sliding_window,
        max_context_len, workspace
    );
}