//! Fused Rotary Position Embedding (RoPE) CUDA kernel interface
//!
//! This module provides a high-performance fused rotary embedding implementation
//! that fuses two operations:
//!   1. Position-based cos/sin selection (eliminates index_select kernel)
//!   2. Rotary position embedding application
//!
//! Supports Grouped Query Attention (GQA) where Q and K have different head counts.

use candle_core::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
use kernels::ffi;

/// Fused Rotary Position Embedding
///
/// Applies rotary position embedding to Q and K tensors using optimized CUDA kernels.
/// Fuses the position-based cos/sin selection with the RoPE computation.
pub struct FusedRope;

impl FusedRope {
    /// Apply fused rotary embedding with position-based cos/sin selection.
    ///
    /// This fuses index_select + RoPE into a single kernel, eliminating one kernel launch.
    ///
    /// # Arguments
    /// * `q` - Query tensor, shape [batch, num_q_heads, seq_len, head_dim]
    /// * `k` - Key tensor, shape [batch, num_kv_heads, seq_len, head_dim]
    /// * `cos` - FULL cosine table, shape [max_seq_len, head_dim/2]
    /// * `sin` - FULL sine table, shape [max_seq_len, head_dim/2]
    /// * `positions` - Position indices, shape [seq_len] (i64)
    /// * `is_interleaved` - If true, uses interleaved layout (adjacent pairs)
    ///
    /// # Returns
    /// Result with (q_embed, k_embed) tensors with rotary embedding applied
    #[cfg(feature = "cuda")]
    pub fn apply(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
    ) -> Result<(Tensor, Tensor)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;

        // Validate inputs - Q and K can have different head counts (GQA)
        let (b, q_h, seq_len, d) = q.dims4()?;
        let (kb, k_h, k_seq_len, kd) = k.dims4()?;

        if b != kb || seq_len != k_seq_len || d != kd {
            candle_core::bail!(
                "Q and K batch/seq_len/head_dim must match, got Q: {:?}, K: {:?}",
                q.shape(),
                k.shape()
            );
        }

        // Positions should be 1D with length seq_len
        let pos_shape = positions.dims();
        if pos_shape.len() != 1 || pos_shape[0] != seq_len {
            candle_core::bail!(
                "positions should be [seq_len], got {:?}, expected [{}]",
                pos_shape,
                seq_len
            );
        }

        // Ensure positions is i64
        let positions = if positions.dtype() != DType::I64 {
            positions.to_dtype(DType::I64)?
        } else {
            positions.clone()
        };

        // Ensure contiguity
        let q = if !q.is_contiguous() {
            q.contiguous()?
        } else {
            q.clone()
        };
        let k = if !k.is_contiguous() {
            k.contiguous()?
        } else {
            k.clone()
        };
        let cos = if !cos.is_contiguous() {
            cos.contiguous()?
        } else {
            cos.clone()
        };
        let sin = if !sin.is_contiguous() {
            sin.contiguous()?
        } else {
            sin.clone()
        };
        let positions = if !positions.is_contiguous() {
            positions.contiguous()?
        } else {
            positions
        };

        // Validate dtypes match (except positions which is always i64)
        let dtype = q.dtype();
        if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
            candle_core::bail!(
                "Q, K, cos, sin must have same dtype, got Q: {:?}, K: {:?}, cos: {:?}, sin: {:?}",
                q.dtype(),
                k.dtype(),
                cos.dtype(),
                sin.dtype()
            );
        }

        // Get device
        let dev = q.device().as_cuda_device()?;
        let stream = *dev.cu_stream() as i64;

        // Calculate kernel parameters
        let q_bh = (b * q_h) as u32;
        let k_bh = (b * k_h) as u32;
        let seq_len_u32 = seq_len as u32;
        let d_u32 = d as u32;

        // Clone for output
        let q_out = q.clone();
        let k_out = k.clone();

        // Get storage
        let q_out_storage = q_out.storage_and_layout().0;
        let k_out_storage = k_out.storage_and_layout().0;
        let cos_storage = cos.storage_and_layout().0;
        let sin_storage = sin.storage_and_layout().0;
        let pos_storage = positions.storage_and_layout().0;

        let q_out_cuda = match &*q_out_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Q must be on CUDA"),
        };
        let k_out_cuda = match &*k_out_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("K must be on CUDA"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("cos must be on CUDA"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("sin must be on CUDA"),
        };
        let pos_cuda = match &*pos_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("positions must be on CUDA"),
        };

        // Get positions pointer
        let pos_ptr = match &pos_cuda.slice {
            CudaStorageSlice::I64(s) => *s.device_ptr() as *const i64,
            _ => candle_core::bail!("positions must be I64"),
        };

        match dtype {
            DType::F32 => {
                let q_ptr = match &q_out_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32"),
                };
                let k_ptr = match &k_out_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f32(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    } else {
                        ffi::fused_rope_f32(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    }
                }
            }
            DType::F16 => {
                let q_ptr = match &q_out_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };
                let k_ptr = match &k_out_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    } else {
                        ffi::fused_rope_f16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    }
                }
            }
            DType::BF16 => {
                let q_ptr = match &q_out_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };
                let k_ptr = match &k_out_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_bf16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    } else {
                        ffi::fused_rope_bf16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    }
                }
            }
            _ => candle_core::bail!("FusedRope only supports F32, F16, BF16, got {:?}", dtype),
        }

        Ok((q_out, k_out))
    }

    /// Apply fused rotary embedding in-place.
    ///
    /// Same as `apply` but modifies Q and K directly.
    #[cfg(feature = "cuda")]
    pub fn apply_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;

        let (b, q_h, seq_len, d) = q.dims4()?;
        let (kb, k_h, k_seq_len, kd) = k.dims4()?;

        if b != kb || seq_len != k_seq_len || d != kd {
            candle_core::bail!(
                "Q and K batch/seq_len/head_dim must match, got Q: {:?}, K: {:?}",
                q.shape(),
                k.shape()
            );
        }

        // Tensors must be contiguous for in-place
        if !q.is_contiguous() || !k.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous()
        {
            candle_core::bail!("All tensors must be contiguous for in-place operation");
        }

        let positions = if positions.dtype() != DType::I64 {
            positions.to_dtype(DType::I64)?
        } else {
            positions.clone()
        };
        let positions = if !positions.is_contiguous() {
            positions.contiguous()?
        } else {
            positions
        };

        let dtype = q.dtype();
        if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
            candle_core::bail!("Q, K, cos, sin must have same dtype");
        }

        let dev = q.device().as_cuda_device()?;
        let stream = *dev.cu_stream() as i64;

        let q_bh = (b * q_h) as u32;
        let k_bh = (b * k_h) as u32;
        let seq_len_u32 = seq_len as u32;
        let d_u32 = d as u32;

        let q_storage = q.storage_and_layout().0;
        let k_storage = k.storage_and_layout().0;
        let cos_storage = cos.storage_and_layout().0;
        let sin_storage = sin.storage_and_layout().0;
        let pos_storage = positions.storage_and_layout().0;

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Q must be on CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("K must be on CUDA"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("cos must be on CUDA"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("sin must be on CUDA"),
        };
        let pos_cuda = match &*pos_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("positions must be on CUDA"),
        };

        let pos_ptr = match &pos_cuda.slice {
            CudaStorageSlice::I64(s) => *s.device_ptr() as *const i64,
            _ => candle_core::bail!("positions must be I64"),
        };

        match dtype {
            DType::F32 => {
                let q_ptr = match &q_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32"),
                };
                let k_ptr = match &k_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f32(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    } else {
                        ffi::fused_rope_f32(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    }
                }
            }
            DType::F16 => {
                let q_ptr = match &q_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };
                let k_ptr = match &k_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    } else {
                        ffi::fused_rope_f16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    }
                }
            }
            DType::BF16 => {
                let q_ptr = match &q_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };
                let k_ptr = match &k_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_bf16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    } else {
                        ffi::fused_rope_bf16(
                            q_ptr,
                            k_ptr,
                            cos_ptr,
                            sin_ptr,
                            pos_ptr,
                            q_bh,
                            k_bh,
                            seq_len_u32,
                            d_u32,
                            stream,
                        );
                    }
                }
            }
            _ => candle_core::bail!("FusedRope only supports F32, F16, BF16, got {:?}", dtype),
        }

        Ok(())
    }

    /// Convenience: non-interleaved RoPE
    #[cfg(feature = "cuda")]
    pub fn apply_rope(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, positions, false)
    }

    /// Convenience: interleaved RoPE
    #[cfg(feature = "cuda")]
    pub fn apply_rope_i(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, positions, true)
    }

    /// Convenience: non-interleaved RoPE in-place
    #[cfg(feature = "cuda")]
    pub fn apply_rope_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, positions, false)
    }

    /// Convenience: interleaved RoPE in-place
    #[cfg(feature = "cuda")]
    pub fn apply_rope_i_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, positions, true)
    }
}
