#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Tensor};

/// FP8 Matrix Multiplication: C = A * B^T
///
/// # Arguments
/// * `input` - Input tensor A of shape [M, K]
/// * `weight` - Weight tensor B of shape [N, K] (stored as u8)
/// * `weight_scale` - Scales for weight tensor
/// * `block_size` - [block_size_y, block_size_x] for scaling
///
/// The weight tensor is expected to be in FP8 format (e4m3).
#[allow(unused)]
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize], // [block_size_y, block_size_x]
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (n, k_w) = weight.dims2()?;

    if k != k_w {
        candle_core::bail!(
            "Shape mismatch in fp8_matmul: input [{}, {}], weight [{}, {}]",
            m,
            k,
            n,
            k_w
        );
    }

    let dev = input.device();
    let dtype = input.dtype();
    assert!(
        weight_scale.dtype() == DType::F32,
        "fp8_matmul expects f32 scales, got {:?}",
        weight_scale.dtype()
    );
    let scale_row_stride = (k_w + block_size[1] - 1) / block_size[1];

    let output = Tensor::zeros((m, n), dtype, dev)?;

    match (dev, dtype) {
        #[cfg(feature = "cuda")]
        (Device::Cuda(dev), DType::F16) => {
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };
            let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;

            let (weight_storage, _) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                _ => candle_core::bail!("weight must be a cuda tensor"),
            };
            let weight_ptr = *weight_slice.device_ptr() as *const u8;

            let (scale_storage, _) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };
            let weight_scale_ptr = *scale_slice.device_ptr() as *const f32;

            let (output_storage, _) = output.storage_and_layout();
            let output_slice = match &*output_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => candle_core::bail!("output allocation failed"),
            };
            let output_ptr = *output_slice.device_ptr() as *mut core::ffi::c_void;

            let stream = *dev.cu_stream() as i64;

            unsafe {
                ffi::fp8_matmul_f16(
                    input_ptr,
                    weight_ptr,
                    weight_scale_ptr,
                    output_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    scale_row_stride as i32,
                    block_size[0] as i32,
                    block_size[1] as i32,
                    stream,
                )
            }
        }
        #[cfg(feature = "cuda")]
        (Device::Cuda(dev), DType::BF16) => {
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };
            let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;

            let (weight_storage, _) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                _ => candle_core::bail!("weight must be a cuda tensor"),
            };
            let weight_ptr = *weight_slice.device_ptr() as *const u8;

            let (scale_storage, _) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };
            let weight_scale_ptr = *scale_slice.device_ptr() as *const f32;

            let (output_storage, _) = output.storage_and_layout();
            let output_slice = match &*output_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
                _ => candle_core::bail!("output allocation failed"),
            };
            let output_ptr = *output_slice.device_ptr() as *mut core::ffi::c_void;

            let stream = *dev.cu_stream() as i64;

            unsafe {
                ffi::fp8_matmul_bf16(
                    input_ptr,
                    weight_ptr,
                    weight_scale_ptr,
                    output_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    scale_row_stride as i32,
                    block_size[0] as i32,
                    block_size[1] as i32,
                    stream,
                )
            }
        }
        (Device::Cuda(_), _) => candle_core::bail!("fp8_matmul requires f16 or bf16 input"),
        #[cfg(feature = "metal")]
        (Device::Metal(dev), _) => {
            let (input_storage, input_layout) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("input must be a metal tensor"),
            };
            let input_offset = input_layout.start_offset() * input.dtype().size_in_bytes();

            let (weight_storage, weight_layout) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("weight must be a metal tensor"),
            };
            let weight_offset = weight_layout.start_offset() * weight.dtype().size_in_bytes();

            let (scale_storage, scale_layout) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("weight_scale must be a metal tensor"),
            };
            let scale_offset = scale_layout.start_offset() * weight_scale.dtype().size_in_bytes();

            let (output_storage, output_layout) = output.storage_and_layout();
            let output_slice = match &*output_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("output allocation failed"),
            };
            let output_offset = output_layout.start_offset() * output.dtype().size_in_bytes();

            let command_buffer = dev.command_buffer()?;

            metal_kernels::call_fp8_matmul(
                dev.device(),
                &command_buffer,
                metal_kernels::Kernels::default(),
                dtype,
                input_slice.buffer(),
                input_offset,
                weight_slice.buffer(),
                weight_offset,
                scale_slice.buffer(),
                scale_offset,
                output_slice.buffer(),
                output_offset,
                m as i32,
                n as i32,
                k as i32,
                scale_row_stride as i32,
                block_size[0] as i32,
                block_size[1] as i32,
            )
            .map_err(candle_core::Error::wrap)?;
        }
        _ => candle_core::bail!("fp8_matmul only supports CUDA and Metal"), // Adjusted error message
    }

    Ok(output)
}
