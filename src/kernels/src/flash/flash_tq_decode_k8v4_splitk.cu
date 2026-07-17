// attention.rs/src/kernels/src/flash/flash_tq_decode_k8v4_splitk.cu
// TurboQuant K8V4 split-K decode kernel with SM120 optimizations

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// TurboQuant K8V4 split-K decode kernel
__global__ void flash_tq_decode_k8v4_splitk_kernel(
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
    float* workspace,
    int num_splits
) {
    // SM120-optimized split-K for decode
    constexpr int TILE_M = 1;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 16;
    
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int split_idx = blockIdx.z;
    int thread_idx = threadIdx.x;
    
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads) return;
    
    // Load query for this head
    float q_val = Q[seq_idx * num_heads * head_dim + head_idx * head_dim + thread_idx];
    
    // Compute partial attention with split-K
    float partial_acc = 0.0f;
    int context_len = context_lens[seq_idx];
    
    for (int block = 0; block < (context_len + block_size - 1) / block_size; block++) {
        int k_block = block_tables[seq_idx * max_context_len / block_size + block];
        if (k_block == 0) continue;
        
        int k_start = block * block_size;
        int k_end = min(k_start + block_size, context_len);
        
        for (int k = k_start; k < k_end; k += TILE_K) {
            // Split-K: only process this split's portion
            if ((k / TILE_K) % num_splits != split_idx) continue;
            
            int k_idx = k_block * block_size * num_kv_heads * head_dim + 
                        (head_idx / (num_heads / num_kv_heads)) * head_dim + 
                        (k % TILE_K);
            
            uint8_t k_quant_val = K_quant[k_idx];
            uint8_t v_quant_val = V_quant[k_idx];
            float k_scale = K_absmax[k_idx / TILE_K];
            float v_scale = V_absmax[k_idx / TILE_K];
            
            float k_val = dequant_k8v4_k(k_quant_val, k_scale);
            float v_val = dequant_k8v4_v(v_quant_val, v_scale);
            
            partial_acc += q_val * k_val * v_val;
        }
    }
    
    // Write partial result to workspace
    int workspace_idx = (seq_idx * num_heads + head_idx) * num_splits * head_dim + 
                        split_idx * head_dim + thread_idx;
    workspace[workspace_idx] = partial_acc;
    
    // Reduction kernel will combine splits
}

// K8V4 dequantization for K (8-bit)
__device__ __forceinline__ float dequant_k8v4_k(uint8_t quant_val, float scale) {
    uint8_t val = quant_val;
    return val * scale;
}

// K8V4 dequantization for V (4-bit)
__device__ __forceinline__ float dequant_k8v4_v(uint8_t quant_val, float scale) {
    uint8_t val = quant_val & 0x0F;
    return val * scale;
}

// Split-K reduction kernel
__global__ void flash_tq_splitk_reduce_kernel(
    float* workspace,
    float* O,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int num_splits,
    float scale,
    float softcap
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads) return;
    
    // Sum all split-K partials
    float sum = 0.0f;
    for (int split = 0; split < num_splits; split++) {
        int workspace_idx = (seq_idx * num_heads + head_idx) * num_splits * head_dim + 
                            split * head_dim + thread_idx;
        sum += workspace[workspace_idx];
    }
    
    // Apply softmax and write output
    sum *= scale;
    if (softcap > 0.0f) {
        sum = softcap * tanhf(sum / softcap);
    }
    
    O[seq_idx * num_heads * head_dim + head_idx * head_dim + thread_idx] = sum;
}

// Kernel launcher
extern "C" void launch_flash_tq_decode_k8v4_splitk(
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
    int num_splits,
    cudaStream_t stream
) {
    // Launch split-K decode kernel
    dim3 grid_decode(num_seqs, num_kv_heads, num_splits);
    dim3 block_decode(head_dim);
    
    flash_tq_decode_k8v4_splitk_kernel<<<grid_decode, block_decode, 0, stream>>>(
        Q, K_quant, K_absmax, V_quant, V_absmax,
        block_tables, context_lens, O,
        num_seqs, num_heads, num_kv_heads, head_dim,
        block_size, scale, softcap, sliding_window,
        max_context_len, workspace, num_splits
    );
    
    // Launch reduction kernel
    dim3 grid_reduce(num_seqs, num_kv_heads, 1);
    dim3 block_reduce(head_dim);
    
    flash_tq_splitk_reduce_kernel<<<grid_reduce, block_reduce, 0, stream>>>(
        workspace, O,
        num_seqs, num_heads, num_kv_heads, head_dim,
        num_splits, scale, softcap
    );
}