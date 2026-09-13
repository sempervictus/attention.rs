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
// from CCCL internals. This type is NOT on the default include path when
// using CUTLASS's bundled CUB (which shadows the system CCCL headers),
// regardless of CUDA version. Provide a minimal polyfill that falls back
// to regular integer division.
//
// The guard checks for the CCCL internal header that defines the real type.
// If it's already been included (e.g. CUDA <cuda/std/...> on CUDA 13+ with
// full CCCL), the polyfill is skipped.
#if !defined(_CUDA_STD_DETAIL_CORE_CORE_DEFS_H_) && !defined(FLASHINFER_CCCL_COMPAT_FAST_MOD_DIV)
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
