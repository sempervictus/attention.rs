//! Fused Rotary Position Embedding (RoPE) CUDA kernel interface
//!
//! This module provides a high-performance fused rotary embedding implementation
//! that operates on both Q and K tensors simultaneously in-place.

use candle_core::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
use kernels::ffi;

/// Fused Rotary Position Embedding
///
/// Applies rotary position embedding to Q and K tensors in-place using optimized CUDA kernels.
/// Supports both interleaved and non-interleaved layouts.
pub struct FusedRope;

impl FusedRope {
    /// Apply fused rotary embedding to Q and K tensors in-place.
    ///
    /// # Arguments
    /// * `q` - Query tensor, shape [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor, shape [batch, num_heads, seq_len, head_dim]
    /// * `cos` - Cosine values, shape [seq_len, head_dim/2]
    /// * `sin` - Sine values, shape [seq_len, head_dim/2]
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
        is_interleaved: bool,
    ) -> Result<(Tensor, Tensor)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;

        // Validate inputs
        let (b, h, t, d) = q.dims4()?;
        let (kb, kh, kt, kd) = k.dims4()?;

        if (b, h, t, d) != (kb, kh, kt, kd) {
            candle_core::bail!(
                "Q and K shapes must match, got Q: {:?}, K: {:?}",
                q.shape(),
                k.shape()
            );
        }

        // Check cos/sin dimensions
        let cos_shape = cos.dims();
        let sin_shape = sin.dims();

        if cos_shape != sin_shape {
            candle_core::bail!(
                "cos and sin shapes must match, got cos: {:?}, sin: {:?}",
                cos_shape,
                sin_shape
            );
        }

        // cos/sin should be [seq_len, head_dim/2] or similar
        let expected_last_dim = d / 2;
        if cos_shape.len() < 1 || cos_shape[cos_shape.len() - 1] != expected_last_dim {
            candle_core::bail!(
                "cos/sin last dimension should be {}, got {:?}",
                expected_last_dim,
                cos_shape
            );
        }

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

        // Validate dtypes match
        let dtype = q.dtype();
        if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
            candle_core::bail!(
                "All tensors must have the same dtype, got Q: {:?}, K: {:?}, cos: {:?}, sin: {:?}",
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
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d_param = d as u32;
        let stride_b = 0u32; // No batch stride for now (shared cos/sin across batch)

        // Get storage and call appropriate kernel
        let q_storage = q.storage_and_layout().0;
        let k_storage = k.storage_and_layout().0;
        let cos_storage = cos.storage_and_layout().0;
        let sin_storage = sin.storage_and_layout().0;

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Q must be on CUDA device"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("K must be on CUDA device"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("cos must be on CUDA device"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("sin must be on CUDA device"),
        };

        // Note: The kernel operates in-place on q and k
        // We need to clone them first to avoid modifying the originals
        // unless we explicitly want in-place operation
        let q_out = q.clone();
        let k_out = k.clone();

        // Get mutable storage for output
        let q_out_storage = q_out.storage_and_layout().0;
        let k_out_storage = k_out.storage_and_layout().0;

        let q_out_cuda = match &*q_out_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Q output must be on CUDA device"),
        };
        let k_out_cuda = match &*k_out_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("K output must be on CUDA device"),
        };

        match dtype {
            DType::F32 => {
                let q_ptr = match &q_out_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };
                let k_ptr = match &k_out_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f32(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, stride_b, stream,
                        );
                    } else {
                        ffi::fused_rope_f32(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, d_param, stride_b, stream,
                        );
                    }
                }
            }
            DType::F16 => {
                let q_ptr = match &q_out_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };
                let k_ptr = match &k_out_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, stride_b, stream,
                        );
                    } else {
                        ffi::fused_rope_f16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, d_param, stride_b, stream,
                        );
                    }
                }
            }
            DType::BF16 => {
                let q_ptr = match &q_out_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };
                let k_ptr = match &k_out_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_bf16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, stride_b, stream,
                        );
                    } else {
                        ffi::fused_rope_bf16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, d_param, stride_b, stream,
                        );
                    }
                }
            }
            _ => candle_core::bail!(
                "FusedRope only supports F32, F16, and BF16 dtypes, got {:?}",
                dtype
            ),
        }

        Ok((q_out, k_out))
    }

    /// Apply fused rotary embedding to Q and K tensors truly in-place.
    /// This modifies the input tensors directly and returns Result<()>.
    ///
    /// # Arguments
    /// * `q` - Query tensor, shape [batch, num_heads, seq_len, head_dim] (modified in-place)
    /// * `k` - Key tensor, shape [batch, num_heads, seq_len, head_dim] (modified in-place)
    /// * `cos` - Cosine values, shape [seq_len, head_dim/2]
    /// * `sin` - Sine values, shape [seq_len, head_dim/2]
    /// * `is_interleaved` - If true, uses interleaved layout (adjacent pairs)
    ///
    /// # Returns
    /// Result<()> indicating success or failure
    #[cfg(feature = "cuda")]
    pub fn apply_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        is_interleaved: bool,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;

        // Validate inputs
        let (b, h, t, d) = q.dims4()?;
        let (kb, kh, kt, kd) = k.dims4()?;

        if (b, h, t, d) != (kb, kh, kt, kd) {
            candle_core::bail!(
                "Q and K shapes must match, got Q: {:?}, K: {:?}",
                q.shape(),
                k.shape()
            );
        }

        // Check cos/sin dimensions
        let cos_shape = cos.dims();
        let sin_shape = sin.dims();

        if cos_shape != sin_shape {
            candle_core::bail!(
                "cos and sin shapes must match, got cos: {:?}, sin: {:?}",
                cos_shape,
                sin_shape
            );
        }

        // Tensors should already be contiguous for in-place ops
        if !q.is_contiguous() {
            candle_core::bail!("Q must be contiguous for in-place operation");
        }
        if !k.is_contiguous() {
            candle_core::bail!("K must be contiguous for in-place operation");
        }
        if !cos.is_contiguous() {
            candle_core::bail!("cos must be contiguous for in-place operation");
        }
        if !sin.is_contiguous() {
            candle_core::bail!("sin must be contiguous for in-place operation");
        }

        // Validate dtypes match
        let dtype = q.dtype();
        if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
            candle_core::bail!(
                "All tensors must have the same dtype, got Q: {:?}, K: {:?}, cos: {:?}, sin: {:?}",
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
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d_param = d as u32;
        let stride_b = 0u32;

        // Get storage - directly use input tensors (in-place)
        let q_storage = q.storage_and_layout().0;
        let k_storage = k.storage_and_layout().0;
        let cos_storage = cos.storage_and_layout().0;
        let sin_storage = sin.storage_and_layout().0;

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Q must be on CUDA device"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("K must be on CUDA device"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("cos must be on CUDA device"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("sin must be on CUDA device"),
        };

        match dtype {
            DType::F32 => {
                let q_ptr = match &q_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };
                let k_ptr = match &k_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                    _ => candle_core::bail!("Expected F32 storage"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f32(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, stride_b, stream,
                        );
                    } else {
                        ffi::fused_rope_f32(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, d_param, stride_b, stream,
                        );
                    }
                }
            }
            DType::F16 => {
                let q_ptr = match &q_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };
                let k_ptr = match &k_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected F16 storage"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_f16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, stride_b, stream,
                        );
                    } else {
                        ffi::fused_rope_f16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, d_param, stride_b, stream,
                        );
                    }
                }
            }
            DType::BF16 => {
                let q_ptr = match &q_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };
                let k_ptr = match &k_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };
                let cos_ptr = match &cos_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };
                let sin_ptr = match &sin_cuda.slice {
                    CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Expected BF16 storage"),
                };

                unsafe {
                    if is_interleaved {
                        ffi::fused_rope_i_bf16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, stride_b, stream,
                        );
                    } else {
                        ffi::fused_rope_bf16(
                            q_ptr, k_ptr, cos_ptr, sin_ptr, bh, td, d_param, stride_b, stream,
                        );
                    }
                }
            }
            _ => candle_core::bail!(
                "FusedRope only supports F32, F16, and BF16 dtypes, got {:?}",
                dtype
            ),
        }

        Ok(())
    }

    /// Convenience method for non-interleaved rotary embedding
    #[cfg(feature = "cuda")]
    pub fn apply_rope(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, false)
    }

    /// Convenience method for interleaved rotary embedding
    #[cfg(feature = "cuda")]
    pub fn apply_rope_i(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, true)
    }

    /// Convenience method for non-interleaved rotary embedding (in-place)
    #[cfg(feature = "cuda")]
    pub fn apply_rope_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, false)
    }

    /// Convenience method for interleaved rotary embedding (in-place)
    #[cfg(feature = "cuda")]
    pub fn apply_rope_i_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, true)
    }
}
