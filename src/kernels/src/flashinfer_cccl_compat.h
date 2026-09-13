#pragma once
// FlashInfer TRT-LLM quantization.cu uses `cuda::maximum<>` without including it.
//
// CUDA 13: nvcc usually finds <cuda/functional> and CUB pulls the same header.
// CUDA 12.8: libcudacxx lives under include/cccl, which is often *not* on the
// nvcc include path when Cutlass's bundled CUB is used. That CUB no longer
// injects `cuda::maximum`, so SM120 builds fail unless we provide it.
#if __has_include(<cuda/functional>)
#include <cuda/functional>
#endif

#ifndef _CUDA_FUNCTIONAL_MAXIMUM_H
#define _CUDA_FUNCTIONAL_MAXIMUM_H
namespace cuda {
template <class T = void>
struct maximum {
  __host__ __device__ constexpr T operator()(T const& a, T const& b) const {
    return a < b ? b : a;
  }
};

template <>
struct maximum<void> {
  template <class T1, class T2>
  __host__ __device__ constexpr auto operator()(T1 const& a, T2 const& b) const
      -> decltype(a < b ? b : a) {
    return a < b ? b : a;
  }
};
}  // namespace cuda
#endif

// FlashInfer fastdiv.cuh (commit 2bfb9334+) uses cuda::fast_mod_div<uint32_t>
// from CCCL internals. This type is not publicly available in CUDA 12.x and
// the header is not on the include path when using CUTLASS's bundled CUB.
// Provide a minimal polyfill that falls back to regular integer division.
// On CUDA 13+ (defined by -DCUDA_VERSION_13) the native CCCL type is
// available; skip the polyfill.
#if !defined(CUDA_VERSION_13) && !defined(_CUDA_STD_DETAIL_FAST_MATH_H) && !defined(CUDA_HAS_FAST_MOD_DIV)
#define FLASHINFER_CCCL_COMPAT_FAST_MOD_DIV
namespace cuda {
template <typename T>
struct fast_mod_div {
  T divisor;
  __host__ __device__ explicit fast_mod_div(T d) : divisor(d ? d : 1) {}
};
template <typename T>
__host__ __device__ inline T operator/(T n, const fast_mod_div<T>& f) {
  return n / f.divisor;
}
template <typename T>
__host__ __device__ inline T operator%(T n, const fast_mod_div<T>& f) {
  return n % f.divisor;
}
}  // namespace cuda
#endif
