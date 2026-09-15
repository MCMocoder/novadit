#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "common/cuda_check.h"
#include "common/cutlass_config.h"

// FP8 support (requires CUDA 11.8+)
#if defined(__CUDACC__) && CUDART_VERSION >= 11080
#define NOVADIT_HAS_FP8 1
#include <cuda_fp8.h>
#endif

namespace {

constexpr int kAdalnThreads = 256;
constexpr int kWarpSize = 32;

template <typename scalar_t> struct VecWidth {
  static constexpr int kElements = sizeof(scalar_t) >= 4 ? 4 : 8;
};

template <typename scalar_t>
using VecType = cutlass::AlignedArray<scalar_t, VecWidth<scalar_t>::kElements>;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

template <typename scalar_t>
__device__ __forceinline__ VecType<scalar_t> const *
as_vec_ptr(scalar_t const *ptr) {
  return reinterpret_cast<VecType<scalar_t> const *>(ptr);
}

template <typename scalar_t>
__device__ __forceinline__ VecType<scalar_t> *as_vec_ptr(scalar_t *ptr) {
  return reinterpret_cast<VecType<scalar_t> *>(ptr);
}

#ifdef NOVADIT_HAS_FP8
template <> struct VecWidth<__nv_fp8_e4m3> {
  static constexpr int kElements = 16;
};

template <> struct VecWidth<__nv_fp8_e5m2> {
  static constexpr int kElements = 16;
};

template <>
__device__ __forceinline__ float to_float(__nv_fp8_e4m3 value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 from_float(float value) {
  return static_cast<__nv_fp8_e4m3>(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_fp8_e5m2 value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 from_float(float value) {
  return static_cast<__nv_fp8_e5m2>(value);
}
#endif  // NOVADIT_HAS_FP8

template <typename scalar_t>
__device__ __forceinline__ void
accumulate_fragment(VecType<scalar_t> const &fragment, float &sum,
                    float &sumsq) {
#pragma unroll
  for (int i = 0; i < VecWidth<scalar_t>::kElements; ++i) {
    float x = to_float(fragment[i]);
    sum += x;
    sumsq += x * x;
  }
}

template <typename scalar_t>
__device__ __forceinline__ bool is_aligned_ptr_device(scalar_t const *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignof(VecType<scalar_t>)) == 0;
}

template <int Threads>
__device__ __forceinline__ void block_reduce_pair(float &sum, float &sumsq,
                                                  float *shared_sum,
                                                  float *shared_sumsq) {
  static_assert(
      (Threads + kWarpSize - 1) / kWarpSize <= kWarpSize,
      "block_reduce_pair requires at most one warp of partial reducers");

  int lane = threadIdx.x & (kWarpSize - 1);
  int warp = threadIdx.x / kWarpSize;
  constexpr int kWarpCount = (Threads + kWarpSize - 1) / kWarpSize;

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
    sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
  }

  if (lane == 0) {
    shared_sum[warp] = sum;
    shared_sumsq[warp] = sumsq;
  }
  __syncthreads();

  if (warp == 0) {
    sum = lane < kWarpCount ? shared_sum[lane] : 0.0f;
    sumsq = lane < kWarpCount ? shared_sumsq[lane] : 0.0f;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
      sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
    if (lane == 0) {
      shared_sum[0] = sum;
      shared_sumsq[0] = sumsq;
    }
  }
  __syncthreads();

  sum = shared_sum[0];
  sumsq = shared_sumsq[0];
}

template <typename scalar_t, int HiddenSize>
__global__ void fused_adaln_ungated_aligned_kernel(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, float *mean_out, float *invstd_out, int64_t batch_size,
    int64_t seq_len, float eps) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kVecCount = HiddenSize / kVecElems;
  constexpr int kWarpCount = (kAdalnThreads + kWarpSize - 1) / kWarpSize;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) {
    return;
  }

  int64_t batch_idx = row_idx / seq_len;
  int64_t x_offset = row_idx * HiddenSize;
  int64_t cond_offset = batch_idx * HiddenSize;

  scalar_t const *x_row = x + x_offset;
  scalar_t const *scale_row = scale + cond_offset;
  scalar_t const *shift_row = shift + cond_offset;
  scalar_t *y_row = y + x_offset;

  auto x_vec = as_vec_ptr(x_row);

  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int vec_idx = threadIdx.x; vec_idx < kVecCount; vec_idx += blockDim.x) {
    VecType<scalar_t> fragment = x_vec[vec_idx];
    accumulate_fragment(fragment, sum, sumsq);
  }

  __shared__ float shared_sum[kWarpCount];
  __shared__ float shared_sumsq[kWarpCount];
  block_reduce_pair<kAdalnThreads>(sum, sumsq, shared_sum, shared_sumsq);

  float inv_hidden = 1.0f / static_cast<float>(HiddenSize);
  float mean = sum * inv_hidden;
  float var = fmaxf(sumsq * inv_hidden - mean * mean, 0.0f);
  float inv_std = rsqrtf(var + eps);
  if (threadIdx.x == 0) {
    mean_out[row_idx] = mean;
    invstd_out[row_idx] = inv_std;
  }

  auto scale_vec = as_vec_ptr(scale_row);
  auto shift_vec = as_vec_ptr(shift_row);
  auto y_vec = as_vec_ptr(y_row);

  for (int vec_idx = threadIdx.x; vec_idx < kVecCount; vec_idx += blockDim.x) {
    VecType<scalar_t> x_fragment = x_vec[vec_idx];
    VecType<scalar_t> scale_fragment = scale_vec[vec_idx];
    VecType<scalar_t> shift_fragment = shift_vec[vec_idx];
    VecType<scalar_t> out_fragment;

#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      float x_val = to_float(x_fragment[i]);
      float scale_val = to_float(scale_fragment[i]);
      float shift_val = to_float(shift_fragment[i]);
      float xhat = (x_val - mean) * inv_std;
      out_fragment[i] =
          from_float<scalar_t>(xhat * (1.0f + scale_val) + shift_val);
    }

    y_vec[vec_idx] = out_fragment;
  }
}

template <typename scalar_t>
__global__ void fused_adaln_ungated_dynamic_kernel(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, float *mean_out, float *invstd_out, int64_t batch_size,
    int64_t seq_len, int64_t hidden_size, float eps) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kWarpCount = (kAdalnThreads + kWarpSize - 1) / kWarpSize;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) {
    return;
  }

  int64_t batch_idx = row_idx / seq_len;
  int64_t x_offset = row_idx * hidden_size;
  int64_t cond_offset = batch_idx * hidden_size;

  scalar_t const *x_row = x + x_offset;
  scalar_t const *scale_row = scale + cond_offset;
  scalar_t const *shift_row = shift + cond_offset;
  scalar_t *y_row = y + x_offset;

  int64_t vec_prefix_elems = (hidden_size / kVecElems) * kVecElems;
  int64_t vec_count = vec_prefix_elems / kVecElems;
  bool row_aligned = vec_prefix_elems > 0 && is_aligned_ptr_device(x_row) &&
                     is_aligned_ptr_device(scale_row) &&
                     is_aligned_ptr_device(shift_row) &&
                     is_aligned_ptr_device(y_row);
  // If alignment is lost for this row, fall back to scalar processing for the
  // full row instead of only handling the tail.
  int64_t scalar_start = row_aligned ? vec_prefix_elems : 0;

  float sum = 0.0f;
  float sumsq = 0.0f;

  if (row_aligned) {
    auto x_vec = as_vec_ptr(x_row);
    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> fragment = x_vec[vec_idx];
      accumulate_fragment(fragment, sum, sumsq);
    }
  }

  for (int64_t idx = scalar_start + threadIdx.x; idx < hidden_size;
       idx += blockDim.x) {
    float x_val = to_float(x_row[idx]);
    sum += x_val;
    sumsq += x_val * x_val;
  }

  __shared__ float shared_sum[kWarpCount];
  __shared__ float shared_sumsq[kWarpCount];
  block_reduce_pair<kAdalnThreads>(sum, sumsq, shared_sum, shared_sumsq);

  float inv_hidden = 1.0f / static_cast<float>(hidden_size);
  float mean = sum * inv_hidden;
  float var = fmaxf(sumsq * inv_hidden - mean * mean, 0.0f);
  float inv_std = rsqrtf(var + eps);
  if (threadIdx.x == 0) {
    mean_out[row_idx] = mean;
    invstd_out[row_idx] = inv_std;
  }

  if (row_aligned) {
    auto x_vec = as_vec_ptr(x_row);
    auto scale_vec = as_vec_ptr(scale_row);
    auto shift_vec = as_vec_ptr(shift_row);
    auto y_vec = as_vec_ptr(y_row);

    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> x_fragment = x_vec[vec_idx];
      VecType<scalar_t> scale_fragment = scale_vec[vec_idx];
      VecType<scalar_t> shift_fragment = shift_vec[vec_idx];
      VecType<scalar_t> out_fragment;

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float x_val = to_float(x_fragment[i]);
        float scale_val = to_float(scale_fragment[i]);
        float shift_val = to_float(shift_fragment[i]);
        float xhat = (x_val - mean) * inv_std;
        out_fragment[i] =
            from_float<scalar_t>(xhat * (1.0f + scale_val) + shift_val);
      }

      y_vec[vec_idx] = out_fragment;
    }
  }

  for (int64_t idx = scalar_start + threadIdx.x; idx < hidden_size;
       idx += blockDim.x) {
    float x_val = to_float(x_row[idx]);
    float scale_val = to_float(scale_row[idx]);
    float shift_val = to_float(shift_row[idx]);
    float xhat = (x_val - mean) * inv_std;
    y_row[idx] = from_float<scalar_t>(xhat * (1.0f + scale_val) + shift_val);
  }
}


// ---- Inference-only forward kernels (no mean/invstd output) ----

template <typename scalar_t, int HiddenSize>
__global__ void fused_adaln_ungated_inference_aligned_kernel(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, int64_t batch_size, int64_t seq_len, float eps) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kVecCount = HiddenSize / kVecElems;
  constexpr int kWarpCount = (kAdalnThreads + kWarpSize - 1) / kWarpSize;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) return;

  int64_t batch_idx = row_idx / seq_len;
  int64_t x_offset = row_idx * HiddenSize;
  int64_t cond_offset = batch_idx * HiddenSize;

  scalar_t const *x_row = x + x_offset;
  scalar_t const *scale_row = scale + cond_offset;
  scalar_t const *shift_row = shift + cond_offset;
  scalar_t *y_row = y + x_offset;

  auto x_vec = as_vec_ptr(x_row);

  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int vec_idx = threadIdx.x; vec_idx < kVecCount; vec_idx += blockDim.x) {
    VecType<scalar_t> fragment = x_vec[vec_idx];
    accumulate_fragment(fragment, sum, sumsq);
  }

  __shared__ float shared_sum[kWarpCount];
  __shared__ float shared_sumsq[kWarpCount];
  block_reduce_pair<kAdalnThreads>(sum, sumsq, shared_sum, shared_sumsq);

  float inv_hidden = 1.0f / static_cast<float>(HiddenSize);
  float mean = sum * inv_hidden;
  float var = fmaxf(sumsq * inv_hidden - mean * mean, 0.0f);
  float inv_std = rsqrtf(var + eps);

  auto scale_vec = as_vec_ptr(scale_row);
  auto shift_vec = as_vec_ptr(shift_row);
  auto y_vec = as_vec_ptr(y_row);

  for (int vec_idx = threadIdx.x; vec_idx < kVecCount; vec_idx += blockDim.x) {
    VecType<scalar_t> x_fragment = x_vec[vec_idx];
    VecType<scalar_t> scale_fragment = scale_vec[vec_idx];
    VecType<scalar_t> shift_fragment = shift_vec[vec_idx];
    VecType<scalar_t> out_fragment;

#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      float x_val = to_float(x_fragment[i]);
      float scale_val = to_float(scale_fragment[i]);
      float shift_val = to_float(shift_fragment[i]);
      float xhat = (x_val - mean) * inv_std;
      out_fragment[i] =
          from_float<scalar_t>(xhat * (1.0f + scale_val) + shift_val);
    }
    y_vec[vec_idx] = out_fragment;
  }
}

template <typename scalar_t>
__global__ void fused_adaln_ungated_inference_dynamic_kernel(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, int64_t batch_size, int64_t seq_len,
    int64_t hidden_size, float eps) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kWarpCount = (kAdalnThreads + kWarpSize - 1) / kWarpSize;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) return;

  int64_t batch_idx = row_idx / seq_len;
  int64_t x_offset = row_idx * hidden_size;
  int64_t cond_offset = batch_idx * hidden_size;

  scalar_t const *x_row = x + x_offset;
  scalar_t const *scale_row = scale + cond_offset;
  scalar_t const *shift_row = shift + cond_offset;
  scalar_t *y_row = y + x_offset;

  int64_t vec_prefix_elems = (hidden_size / kVecElems) * kVecElems;
  int64_t vec_count = vec_prefix_elems / kVecElems;
  bool row_aligned = vec_prefix_elems > 0 && is_aligned_ptr_device(x_row) &&
                     is_aligned_ptr_device(scale_row) &&
                     is_aligned_ptr_device(shift_row) &&
                     is_aligned_ptr_device(y_row);
  int64_t scalar_start = row_aligned ? vec_prefix_elems : 0;

  float sum = 0.0f;
  float sumsq = 0.0f;

  if (row_aligned) {
    auto x_vec = as_vec_ptr(x_row);
    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> fragment = x_vec[vec_idx];
      accumulate_fragment(fragment, sum, sumsq);
    }
  }

  for (int64_t idx = scalar_start + threadIdx.x; idx < hidden_size;
       idx += blockDim.x) {
    float x_val = to_float(x_row[idx]);
    sum += x_val;
    sumsq += x_val * x_val;
  }

  __shared__ float shared_sum[kWarpCount];
  __shared__ float shared_sumsq[kWarpCount];
  block_reduce_pair<kAdalnThreads>(sum, sumsq, shared_sum, shared_sumsq);

  float inv_hidden = 1.0f / static_cast<float>(hidden_size);
  float mean = sum * inv_hidden;
  float var = fmaxf(sumsq * inv_hidden - mean * mean, 0.0f);
  float inv_std = rsqrtf(var + eps);

  if (row_aligned) {
    auto x_vec = as_vec_ptr(x_row);
    auto scale_vec = as_vec_ptr(scale_row);
    auto shift_vec = as_vec_ptr(shift_row);
    auto y_vec = as_vec_ptr(y_row);

    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> x_fragment = x_vec[vec_idx];
      VecType<scalar_t> scale_fragment = scale_vec[vec_idx];
      VecType<scalar_t> shift_fragment = shift_vec[vec_idx];
      VecType<scalar_t> out_fragment;

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float x_val = to_float(x_fragment[i]);
        float scale_val = to_float(scale_fragment[i]);
        float shift_val = to_float(shift_fragment[i]);
        float xhat = (x_val - mean) * inv_std;
        out_fragment[i] =
            from_float<scalar_t>(xhat * (1.0f + scale_val) + shift_val);
      }
      y_vec[vec_idx] = out_fragment;
    }
  }

  for (int64_t idx = scalar_start + threadIdx.x; idx < hidden_size;
       idx += blockDim.x) {
    float x_val = to_float(x_row[idx]);
    float scale_val = to_float(scale_row[idx]);
    float shift_val = to_float(shift_row[idx]);
    float xhat = (x_val - mean) * inv_std;
    y_row[idx] = from_float<scalar_t>(xhat * (1.0f + scale_val) + shift_val);
  }
}

template <typename scalar_t>
__global__ void fused_adaln_backward_dx_kernel(
    scalar_t const *dy, scalar_t const *x, scalar_t const *scale,
    float const *mean, float const *invstd, scalar_t *dx, int64_t batch_size,
    int64_t seq_len, int64_t hidden_size) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kWarpCount = (kAdalnThreads + kWarpSize - 1) / kWarpSize;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) {
    return;
  }

  int64_t batch_idx = row_idx / seq_len;
  int64_t row_offset = row_idx * hidden_size;
  int64_t cond_offset = batch_idx * hidden_size;

  scalar_t const *dy_row = dy + row_offset;
  scalar_t const *x_row = x + row_offset;
  scalar_t const *scale_row = scale + cond_offset;
  scalar_t *dx_row = dx + row_offset;

  float row_mean = mean[row_idx];
  float row_invstd = invstd[row_idx];
  int64_t vec_prefix_elems = (hidden_size / kVecElems) * kVecElems;
  int64_t vec_count = vec_prefix_elems / kVecElems;
  bool row_aligned = vec_prefix_elems > 0 && is_aligned_ptr_device(dy_row) &&
                     is_aligned_ptr_device(x_row) &&
                     is_aligned_ptr_device(scale_row) &&
                     is_aligned_ptr_device(dx_row);
  int64_t scalar_start = row_aligned ? vec_prefix_elems : 0;

  float sum_dxhat = 0.0f;
  float sum_dxhat_xhat = 0.0f;

  if (row_aligned) {
    auto dy_vec = as_vec_ptr(dy_row);
    auto x_vec = as_vec_ptr(x_row);
    auto scale_vec = as_vec_ptr(scale_row);

    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> dy_fragment = dy_vec[vec_idx];
      VecType<scalar_t> x_fragment = x_vec[vec_idx];
      VecType<scalar_t> scale_fragment = scale_vec[vec_idx];

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float xhat = (to_float(x_fragment[i]) - row_mean) * row_invstd;
        float dxhat =
            to_float(dy_fragment[i]) * (1.0f + to_float(scale_fragment[i]));
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
      }
    }
  }

  for (int64_t idx = scalar_start + threadIdx.x; idx < hidden_size;
       idx += blockDim.x) {
    float xhat = (to_float(x_row[idx]) - row_mean) * row_invstd;
    float dxhat = to_float(dy_row[idx]) * (1.0f + to_float(scale_row[idx]));
    sum_dxhat += dxhat;
    sum_dxhat_xhat += dxhat * xhat;
  }

  __shared__ float shared_sum_dxhat[kWarpCount];
  __shared__ float shared_sum_dxhat_xhat[kWarpCount];
  block_reduce_pair<kAdalnThreads>(sum_dxhat, sum_dxhat_xhat, shared_sum_dxhat,
                                   shared_sum_dxhat_xhat);

  float inv_hidden = 1.0f / static_cast<float>(hidden_size);
  float mean_dxhat = sum_dxhat * inv_hidden;
  float mean_dxhat_xhat = sum_dxhat_xhat * inv_hidden;

  if (row_aligned) {
    auto dy_vec = as_vec_ptr(dy_row);
    auto x_vec = as_vec_ptr(x_row);
    auto scale_vec = as_vec_ptr(scale_row);
    auto dx_vec = as_vec_ptr(dx_row);

    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> dy_fragment = dy_vec[vec_idx];
      VecType<scalar_t> x_fragment = x_vec[vec_idx];
      VecType<scalar_t> scale_fragment = scale_vec[vec_idx];
      VecType<scalar_t> dx_fragment;

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float xhat = (to_float(x_fragment[i]) - row_mean) * row_invstd;
        float dxhat =
            to_float(dy_fragment[i]) * (1.0f + to_float(scale_fragment[i]));
        float dx_val =
            row_invstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
        dx_fragment[i] = from_float<scalar_t>(dx_val);
      }

      dx_vec[vec_idx] = dx_fragment;
    }
  }

  for (int64_t idx = scalar_start + threadIdx.x; idx < hidden_size;
       idx += blockDim.x) {
    float xhat = (to_float(x_row[idx]) - row_mean) * row_invstd;
    float dxhat = to_float(dy_row[idx]) * (1.0f + to_float(scale_row[idx]));
    float dx_val = row_invstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
    dx_row[idx] = from_float<scalar_t>(dx_val);
  }
}

template <typename scalar_t>
__global__ void fused_adaln_backward_dparam_kernel(
    scalar_t const *dy, scalar_t const *x, float const *mean,
    float const *invstd, float *dscale, float *dshift,
    int64_t seq_len, int64_t hidden_size) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kHiddenTile = kWarpSize * kWarpsPerBlock;

  int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  int64_t hidden_idx = static_cast<int64_t>(blockIdx.y) * kHiddenTile +
                       static_cast<int64_t>(threadIdx.y) * kWarpSize +
                       threadIdx.x;
  if (hidden_idx >= hidden_size) {
    return;
  }

  float dscale_acc = 0.0f;
  float dshift_acc = 0.0f;
  int64_t row = batch_idx * seq_len;

  for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx, ++row) {
    int64_t offset = row * hidden_size + hidden_idx;
    float dy_val = to_float(dy[offset]);
    float xhat = (to_float(x[offset]) - mean[row]) * invstd[row];
    dshift_acc += dy_val;
    dscale_acc += dy_val * xhat;
  }

  int64_t output_idx = batch_idx * hidden_size + hidden_idx;
  dscale[output_idx] = dscale_acc;
  dshift[output_idx] = dshift_acc;
}

template <typename scalar_t, int HiddenSize>
void launch_aligned_path(scalar_t const *x, scalar_t const *scale,
                         scalar_t const *shift, scalar_t *y, float *mean,
                         float *invstd, int64_t batch_size, int64_t seq_len,
                         float eps, cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_adaln_ungated_aligned_kernel<scalar_t, HiddenSize>
      <<<row_count, kAdalnThreads, 0, stream>>>(
          x, scale, shift, y, mean, invstd, batch_size, seq_len, eps);
}

template <typename scalar_t>
void launch_dynamic_path(scalar_t const *x, scalar_t const *scale,
                         scalar_t const *shift, scalar_t *y, float *mean,
                         float *invstd, int64_t batch_size, int64_t seq_len,
                         int64_t hidden_size, float eps, cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_adaln_ungated_dynamic_kernel<scalar_t>
      <<<row_count, kAdalnThreads, 0, stream>>>(x, scale, shift, y, mean,
                                                invstd, batch_size, seq_len,
                                                hidden_size, eps);
}

// ---- Inference launch helpers ----

template <typename scalar_t, int HiddenSize>
void launch_inference_aligned_path(scalar_t const *x, scalar_t const *scale,
                                   scalar_t const *shift, scalar_t *y,
                                   int64_t batch_size, int64_t seq_len,
                                   float eps, cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_adaln_ungated_inference_aligned_kernel<scalar_t, HiddenSize>
      <<<row_count, kAdalnThreads, 0, stream>>>(
          x, scale, shift, y, batch_size, seq_len, eps);
}

template <typename scalar_t>
void launch_inference_dynamic_path(scalar_t const *x, scalar_t const *scale,
                                   scalar_t const *shift, scalar_t *y,
                                   int64_t batch_size, int64_t seq_len,
                                   int64_t hidden_size, float eps,
                                   cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_adaln_ungated_inference_dynamic_kernel<scalar_t>
      <<<row_count, kAdalnThreads, 0, stream>>>(x, scale, shift, y,
                                                batch_size, seq_len,
                                                hidden_size, eps);
}

template <typename scalar_t>
bool base_ptrs_support_aligned_path(scalar_t const *x, scalar_t const *scale,
                                    scalar_t const *shift, scalar_t *y,
                                    int64_t hidden_size) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  return hidden_size % kVecElems == 0 &&
         (reinterpret_cast<uintptr_t>(x) % alignof(VecType<scalar_t>)) == 0 &&
         (reinterpret_cast<uintptr_t>(scale) % alignof(VecType<scalar_t>)) ==
             0 &&
         (reinterpret_cast<uintptr_t>(shift) % alignof(VecType<scalar_t>)) ==
             0 &&
         (reinterpret_cast<uintptr_t>(y) % alignof(VecType<scalar_t>)) == 0;
}

template <typename scalar_t>
void dispatch_aligned_or_dynamic(scalar_t const *x, scalar_t const *scale,
                                 scalar_t const *shift, scalar_t *y,
                                 float *mean, float *invstd, int64_t batch_size,
                                 int64_t seq_len, int64_t hidden_size,
                                 float eps, cudaStream_t stream) {
  if (!base_ptrs_support_aligned_path(x, scale, shift, y, hidden_size)) {
    launch_dynamic_path(x, scale, shift, y, mean, invstd, batch_size, seq_len,
                        hidden_size, eps, stream);
    return;
  }

  switch (hidden_size) {
  case 128:
    launch_aligned_path<scalar_t, 128>(x, scale, shift, y, mean, invstd,
                                       batch_size, seq_len, eps, stream);
    break;
  case 256:
    launch_aligned_path<scalar_t, 256>(x, scale, shift, y, mean, invstd,
                                       batch_size, seq_len, eps, stream);
    break;
  case 512:
    launch_aligned_path<scalar_t, 512>(x, scale, shift, y, mean, invstd,
                                       batch_size, seq_len, eps, stream);
    break;
  case 768:
    launch_aligned_path<scalar_t, 768>(x, scale, shift, y, mean, invstd,
                                       batch_size, seq_len, eps, stream);
    break;
  case 1024:
    launch_aligned_path<scalar_t, 1024>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  case 1152:
    launch_aligned_path<scalar_t, 1152>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  case 1536:
    launch_aligned_path<scalar_t, 1536>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  case 2048:
    launch_aligned_path<scalar_t, 2048>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  case 3072:
    launch_aligned_path<scalar_t, 3072>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  case 4096:
    launch_aligned_path<scalar_t, 4096>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  case 8192:
    launch_aligned_path<scalar_t, 8192>(x, scale, shift, y, mean, invstd,
                                        batch_size, seq_len, eps, stream);
    break;
  default:
    launch_dynamic_path(x, scale, shift, y, mean, invstd, batch_size, seq_len,
                        hidden_size, eps, stream);
    break;
  }
}

template <typename scalar_t>
void dispatch_inference_aligned_or_dynamic(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, int64_t batch_size, int64_t seq_len, int64_t hidden_size,
    float eps, cudaStream_t stream) {
  if (!base_ptrs_support_aligned_path(x, scale, shift, y, hidden_size)) {
    launch_inference_dynamic_path(x, scale, shift, y, batch_size, seq_len,
                                  hidden_size, eps, stream);
    return;
  }

  switch (hidden_size) {
  case 128:
    launch_inference_aligned_path<scalar_t, 128>(x, scale, shift, y,
                                                 batch_size, seq_len, eps,
                                                 stream);
    break;
  case 256:
    launch_inference_aligned_path<scalar_t, 256>(x, scale, shift, y,
                                                 batch_size, seq_len, eps,
                                                 stream);
    break;
  case 512:
    launch_inference_aligned_path<scalar_t, 512>(x, scale, shift, y,
                                                 batch_size, seq_len, eps,
                                                 stream);
    break;
  case 768:
    launch_inference_aligned_path<scalar_t, 768>(x, scale, shift, y,
                                                 batch_size, seq_len, eps,
                                                 stream);
    break;
  case 1024:
    launch_inference_aligned_path<scalar_t, 1024>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  case 1152:
    launch_inference_aligned_path<scalar_t, 1152>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  case 1536:
    launch_inference_aligned_path<scalar_t, 1536>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  case 2048:
    launch_inference_aligned_path<scalar_t, 2048>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  case 3072:
    launch_inference_aligned_path<scalar_t, 3072>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  case 4096:
    launch_inference_aligned_path<scalar_t, 4096>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  case 8192:
    launch_inference_aligned_path<scalar_t, 8192>(x, scale, shift, y,
                                                  batch_size, seq_len, eps,
                                                  stream);
    break;
  default:
    launch_inference_dynamic_path(x, scale, shift, y, batch_size, seq_len,
                                  hidden_size, eps, stream);
    break;
  }
}

template <typename scalar_t>
void launch_fused_adaln_ungated_forward_inference(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, int64_t batch_size, int64_t seq_len, int64_t hidden_size,
    float eps, cudaStream_t stream) {
  dispatch_inference_aligned_or_dynamic(x, scale, shift, y, batch_size, seq_len,
                                        hidden_size, eps, stream);
}

template <typename scalar_t>
void launch_fused_adaln_ungated_forward(
    scalar_t const *x, scalar_t const *scale, scalar_t const *shift,
    scalar_t *y, float *mean, float *invstd, int64_t batch_size,
    int64_t seq_len, int64_t hidden_size, float eps, cudaStream_t stream) {
  dispatch_aligned_or_dynamic(x, scale, shift, y, mean, invstd, batch_size,
                              seq_len, hidden_size, eps, stream);
}

template <typename scalar_t>
void launch_fused_adaln_backward_dx(scalar_t const *dy, scalar_t const *x,
                                    scalar_t const *scale, float const *mean,
                                    float const *invstd, scalar_t *dx,
                                    int64_t batch_size, int64_t seq_len,
                                    int64_t hidden_size, cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_adaln_backward_dx_kernel<scalar_t>
      <<<row_count, kAdalnThreads, 0, stream>>>(
          dy, x, scale, mean, invstd, dx, batch_size, seq_len, hidden_size);
}

template <typename scalar_t>
void launch_fused_adaln_backward_dparam(
    scalar_t const *dy, scalar_t const *x, float const *mean,
    float const *invstd, float *dscale, float *dshift,
    int64_t batch_size, int64_t seq_len, int64_t hidden_size,
    cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kHiddenTile = kWarpSize * kWarpsPerBlock;
  dim3 block(kWarpSize, kWarpsPerBlock);
  dim3 grid(batch_size, (hidden_size + kHiddenTile - 1) / kHiddenTile);
  fused_adaln_backward_dparam_kernel<scalar_t>
      <<<grid, block, 0, stream>>>(dy, x, mean, invstd, dscale, dshift, seq_len,
                                  hidden_size);
}

void check_adaln_inputs(at::Tensor const &x, at::Tensor const &scale,
                        at::Tensor const &shift) {
  NOVADIT_CHECK_INPUT(x);
  NOVADIT_CHECK_INPUT(scale);
  NOVADIT_CHECK_INPUT(shift);
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, H]");
  TORCH_CHECK(scale.dim() == 2 || scale.dim() == 3,
              "scale must have shape [B, H] or [B, 1, H]");
  TORCH_CHECK(shift.dim() == 2 || shift.dim() == 3,
              "shift must have shape [B, H] or [B, 1, H]");
  TORCH_CHECK(x.scalar_type() == scale.scalar_type(),
              "x and scale must have the same dtype");
  TORCH_CHECK(x.scalar_type() == shift.scalar_type(),
              "x and shift must have the same dtype");
  TORCH_CHECK(x.size(0) == scale.size(0), "scale batch dimension must match x");
  TORCH_CHECK(x.size(0) == shift.size(0), "shift batch dimension must match x");
  TORCH_CHECK(scale.size(-1) == x.size(-1),
              "scale hidden dimension must match x");
  TORCH_CHECK(shift.size(-1) == x.size(-1),
              "shift hidden dimension must match x");
  if (scale.dim() == 3) {
    TORCH_CHECK(scale.size(1) == 1, "3D scale must have shape [B, 1, H]");
  }
  if (shift.dim() == 3) {
    TORCH_CHECK(shift.size(1) == 1, "3D shift must have shape [B, 1, H]");
  }
}

void check_adaln_backward_dx_inputs(at::Tensor const &dy, at::Tensor const &x,
                                    at::Tensor const &scale,
                                    at::Tensor const &mean,
                                    at::Tensor const &invstd) {
  NOVADIT_CHECK_INPUT(dy);
  NOVADIT_CHECK_INPUT(x);
  NOVADIT_CHECK_INPUT(scale);
  NOVADIT_CHECK_INPUT(mean);
  NOVADIT_CHECK_INPUT(invstd);
  TORCH_CHECK(dy.dim() == 3, "dy must have shape [B, S, H]");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, H]");
  TORCH_CHECK(dy.sizes() == x.sizes(), "dy and x must have the same shape");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type(),
              "dy and x must have the same dtype");
  TORCH_CHECK(scale.dim() == 2 || scale.dim() == 3,
              "scale must have shape [B, H] or [B, 1, H]");
  TORCH_CHECK(scale.scalar_type() == x.scalar_type(),
              "scale and x must have the same dtype");
  TORCH_CHECK(scale.size(0) == x.size(0), "scale batch dimension must match x");
  TORCH_CHECK(scale.size(-1) == x.size(-1),
              "scale hidden dimension must match x");
  if (scale.dim() == 3) {
    TORCH_CHECK(scale.size(1) == 1, "3D scale must have shape [B, 1, H]");
  }
  TORCH_CHECK(mean.scalar_type() == at::kFloat, "mean must have dtype float32");
  TORCH_CHECK(invstd.scalar_type() == at::kFloat,
              "invstd must have dtype float32");
  TORCH_CHECK(mean.dim() == 2 && mean.size(0) == x.size(0) &&
                  mean.size(1) == x.size(1),
              "mean must have shape [B, S]");
  TORCH_CHECK(invstd.sizes() == mean.sizes(),
              "invstd must have the same shape as mean");
  TORCH_CHECK(dy.device() == x.device() && scale.device() == x.device() &&
                  mean.device() == x.device() && invstd.device() == x.device(),
              "all backward tensors must be on the same device");
}

void check_adaln_backward_dparam_inputs(
    at::Tensor const &dy, at::Tensor const &x, at::Tensor const &mean,
    at::Tensor const &invstd) {
  NOVADIT_CHECK_INPUT(dy);
  NOVADIT_CHECK_INPUT(x);
  NOVADIT_CHECK_INPUT(mean);
  NOVADIT_CHECK_INPUT(invstd);
  TORCH_CHECK(dy.dim() == 3, "dy must have shape [B, S, H]");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, H]");
  TORCH_CHECK(dy.sizes() == x.sizes(), "dy and x must have the same shape");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type(),
              "dy and x must have the same dtype");
  TORCH_CHECK(mean.scalar_type() == at::kFloat, "mean must have dtype float32");
  TORCH_CHECK(invstd.scalar_type() == at::kFloat,
              "invstd must have dtype float32");
  TORCH_CHECK(mean.dim() == 2 && mean.size(0) == x.size(0) &&
                  mean.size(1) == x.size(1),
              "mean must have shape [B, S]");
  TORCH_CHECK(invstd.sizes() == mean.sizes(),
              "invstd must have the same shape as mean");
  TORCH_CHECK(dy.device() == x.device() && mean.device() == x.device() &&
                  invstd.device() == x.device(),
              "all backward tensors must be on the same device");
}

void check_launch(cudaError_t error, char const *kernel_name) {
  TORCH_CHECK(error == cudaSuccess, kernel_name,
              " launch failed: ", cudaGetErrorString(error));
}

} // namespace

void fused_adaln_fp16_ungated_forward(at::Half const *x, at::Half const *scale,
                                      at::Half const *shift, at::Half *y,
                                      float *mean, float *invstd,
                                      int64_t batch_size, int64_t seq_len,
                                      int64_t hidden_size, float eps,
                                      cudaStream_t stream) {
  launch_fused_adaln_ungated_forward(x, scale, shift, y, mean, invstd,
                                     batch_size, seq_len, hidden_size, eps,
                                     stream);
}

void fused_adaln_bf16_ungated_forward(
    at::BFloat16 const *x, at::BFloat16 const *scale, at::BFloat16 const *shift,
    at::BFloat16 *y, float *mean, float *invstd, int64_t batch_size,
    int64_t seq_len, int64_t hidden_size, float eps, cudaStream_t stream) {
  launch_fused_adaln_ungated_forward(x, scale, shift, y, mean, invstd,
                                     batch_size, seq_len, hidden_size, eps,
                                     stream);
}

// ---- Inference-only forward entry points ----

void fused_adaln_fp16_ungated_forward_inference(
    at::Half const *x, at::Half const *scale, at::Half const *shift,
    at::Half *y, int64_t batch_size, int64_t seq_len, int64_t hidden_size,
    float eps, cudaStream_t stream) {
  launch_fused_adaln_ungated_forward_inference(x, scale, shift, y, batch_size,
                                               seq_len, hidden_size, eps,
                                               stream);
}

void fused_adaln_bf16_ungated_forward_inference(
    at::BFloat16 const *x, at::BFloat16 const *scale,
    at::BFloat16 const *shift, at::BFloat16 *y, int64_t batch_size,
    int64_t seq_len, int64_t hidden_size, float eps, cudaStream_t stream) {
  launch_fused_adaln_ungated_forward_inference(x, scale, shift, y, batch_size,
                                               seq_len, hidden_size, eps,
                                               stream);
}

#ifdef NOVADIT_HAS_FP8
void fused_adaln_fp8_e4m3_ungated_forward_inference(
    __nv_fp8_e4m3 const *x, __nv_fp8_e4m3 const *scale,
    __nv_fp8_e4m3 const *shift, __nv_fp8_e4m3 *y,
    int64_t batch_size, int64_t seq_len, int64_t hidden_size,
    float eps, cudaStream_t stream) {
  launch_fused_adaln_ungated_forward_inference(x, scale, shift, y, batch_size,
                                               seq_len, hidden_size, eps,
                                               stream);
}

void fused_adaln_fp8_e5m2_ungated_forward_inference(
    __nv_fp8_e5m2 const *x, __nv_fp8_e5m2 const *scale,
    __nv_fp8_e5m2 const *shift, __nv_fp8_e5m2 *y,
    int64_t batch_size, int64_t seq_len, int64_t hidden_size,
    float eps, cudaStream_t stream) {
  launch_fused_adaln_ungated_forward_inference(x, scale, shift, y, batch_size,
                                               seq_len, hidden_size, eps,
                                               stream);
}
#endif  // NOVADIT_HAS_FP8

std::vector<at::Tensor> fused_adaln_forward(const at::Tensor &x,
                                            const at::Tensor &scale,
                                            const at::Tensor &shift,
                                            double eps) {
  check_adaln_inputs(x, scale, shift);
  TORCH_CHECK(x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "fused_adaln_forward only supports fp16/bf16 tensors");

  at::Tensor x_contig = x.contiguous();
  at::Tensor scale_contig = scale.reshape({x.size(0), x.size(2)}).contiguous();
  at::Tensor shift_contig = shift.reshape({x.size(0), x.size(2)}).contiguous();
  at::Tensor y = at::empty_like(x_contig);
  at::TensorOptions stats_options =
      x_contig.options().dtype(at::kFloat).requires_grad(false);
  at::Tensor mean =
      at::empty({x_contig.size(0), x_contig.size(1)}, stats_options);
  at::Tensor invstd = at::empty_like(mean);

  c10::cuda::CUDAGuard device_guard(x_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x_contig.get_device()).stream();

  int64_t batch_size = x_contig.size(0);
  int64_t seq_len = x_contig.size(1);
  int64_t hidden_size = x_contig.size(2);
  float eps_f = static_cast<float>(eps);

  switch (x_contig.scalar_type()) {
  case at::kHalf:
    fused_adaln_fp16_ungated_forward(
        x_contig.data_ptr<at::Half>(), scale_contig.data_ptr<at::Half>(),
        shift_contig.data_ptr<at::Half>(), y.data_ptr<at::Half>(),
        mean.data_ptr<float>(), invstd.data_ptr<float>(), batch_size, seq_len,
        hidden_size, eps_f, stream);
    check_launch(cudaGetLastError(), "fused_adaln_forward<fp16>");
    break;
  case at::kBFloat16:
    fused_adaln_bf16_ungated_forward(
        x_contig.data_ptr<at::BFloat16>(),
        scale_contig.data_ptr<at::BFloat16>(),
        shift_contig.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>(),
        mean.data_ptr<float>(), invstd.data_ptr<float>(), batch_size, seq_len,
        hidden_size, eps_f, stream);
    check_launch(cudaGetLastError(), "fused_adaln_forward<bf16>");
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused_adaln_forward");
  }

  return {y, scale_contig, mean, invstd};
}

at::Tensor fused_adaln_forward_inference(const at::Tensor &x,
                                         const at::Tensor &scale,
                                         const at::Tensor &shift,
                                         double eps) {
  check_adaln_inputs(x, scale, shift);
  TORCH_CHECK(x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "fused_adaln_forward_inference only supports fp16/bf16 tensors");

  at::Tensor x_contig = x.contiguous();
  at::Tensor scale_contig = scale.reshape({x.size(0), x.size(2)}).contiguous();
  at::Tensor shift_contig = shift.reshape({x.size(0), x.size(2)}).contiguous();
  at::Tensor y = at::empty_like(x_contig);

  c10::cuda::CUDAGuard device_guard(x_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x_contig.get_device()).stream();

  int64_t batch_size = x_contig.size(0);
  int64_t seq_len = x_contig.size(1);
  int64_t hidden_size = x_contig.size(2);
  float eps_f = static_cast<float>(eps);

  switch (x_contig.scalar_type()) {
  case at::kHalf:
    fused_adaln_fp16_ungated_forward_inference(
        x_contig.data_ptr<at::Half>(), scale_contig.data_ptr<at::Half>(),
        shift_contig.data_ptr<at::Half>(), y.data_ptr<at::Half>(),
        batch_size, seq_len, hidden_size, eps_f, stream);
    check_launch(cudaGetLastError(), "fused_adaln_forward_inference<fp16>");
    break;
  case at::kBFloat16:
    fused_adaln_bf16_ungated_forward_inference(
        x_contig.data_ptr<at::BFloat16>(),
        scale_contig.data_ptr<at::BFloat16>(),
        shift_contig.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>(),
        batch_size, seq_len, hidden_size, eps_f, stream);
    check_launch(cudaGetLastError(), "fused_adaln_forward_inference<bf16>");
    break;
#ifdef NOVADIT_HAS_FP8
  case at::kFloat8_e4m3fn: {
    using fp8_t = __nv_fp8_e4m3;
    fused_adaln_fp8_e4m3_ungated_forward_inference(
        reinterpret_cast<fp8_t const *>(x_contig.data_ptr()),
        reinterpret_cast<fp8_t const *>(scale_contig.data_ptr()),
        reinterpret_cast<fp8_t const *>(shift_contig.data_ptr()),
        reinterpret_cast<fp8_t *>(y.data_ptr()),
        batch_size, seq_len, hidden_size, eps_f, stream);
    check_launch(cudaGetLastError(),
                 "fused_adaln_forward_inference<fp8_e4m3>");
    break;
  }
  case at::kFloat8_e5m2: {
    using fp8_t = __nv_fp8_e5m2;
    fused_adaln_fp8_e5m2_ungated_forward_inference(
        reinterpret_cast<fp8_t const *>(x_contig.data_ptr()),
        reinterpret_cast<fp8_t const *>(scale_contig.data_ptr()),
        reinterpret_cast<fp8_t const *>(shift_contig.data_ptr()),
        reinterpret_cast<fp8_t *>(y.data_ptr()),
        batch_size, seq_len, hidden_size, eps_f, stream);
    check_launch(cudaGetLastError(),
                 "fused_adaln_forward_inference<fp8_e5m2>");
    break;
  }
#endif  // NOVADIT_HAS_FP8
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused_adaln_forward_inference");
  }

  return y;
}

at::Tensor fused_adaln_backward_dx(const at::Tensor &dy, const at::Tensor &x,
                                   const at::Tensor &scale,
                                   const at::Tensor &mean,
                                   const at::Tensor &invstd) {
  check_adaln_backward_dx_inputs(dy, x, scale, mean, invstd);
  TORCH_CHECK(x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "fused_adaln_backward_dx only supports fp16/bf16 tensors");

  at::Tensor dy_contig = dy.contiguous();
  at::Tensor x_contig = x.contiguous();
  at::Tensor scale_contig = scale.reshape({x.size(0), x.size(2)}).contiguous();
  at::Tensor mean_contig = mean.contiguous();
  at::Tensor invstd_contig = invstd.contiguous();
  at::Tensor dx = at::empty_like(x_contig);

  c10::cuda::CUDAGuard device_guard(x_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x_contig.get_device()).stream();

  int64_t batch_size = x_contig.size(0);
  int64_t seq_len = x_contig.size(1);
  int64_t hidden_size = x_contig.size(2);

  switch (x_contig.scalar_type()) {
  case at::kHalf:
    launch_fused_adaln_backward_dx(
        dy_contig.data_ptr<at::Half>(), x_contig.data_ptr<at::Half>(),
        scale_contig.data_ptr<at::Half>(), mean_contig.data_ptr<float>(),
        invstd_contig.data_ptr<float>(), dx.data_ptr<at::Half>(), batch_size,
        seq_len, hidden_size, stream);
    check_launch(cudaGetLastError(), "fused_adaln_backward_dx<fp16>");
    break;
  case at::kBFloat16:
    launch_fused_adaln_backward_dx(
        dy_contig.data_ptr<at::BFloat16>(), x_contig.data_ptr<at::BFloat16>(),
        scale_contig.data_ptr<at::BFloat16>(), mean_contig.data_ptr<float>(),
        invstd_contig.data_ptr<float>(), dx.data_ptr<at::BFloat16>(),
        batch_size, seq_len, hidden_size, stream);
    check_launch(cudaGetLastError(), "fused_adaln_backward_dx<bf16>");
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused_adaln_backward_dx");
  }

  return dx;
}

std::vector<at::Tensor>
fused_adaln_backward_dparam(const at::Tensor &dy, const at::Tensor &x,
                            const at::Tensor &mean,
                            const at::Tensor &invstd) {
  check_adaln_backward_dparam_inputs(dy, x, mean, invstd);
  TORCH_CHECK(
      x.scalar_type() == at::kHalf ||
          x.scalar_type() == at::kBFloat16,
      "fused_adaln_backward_dparam only supports fp16/bf16 tensors");

  at::Tensor dy_contig = dy.contiguous();
  at::Tensor x_contig = x.contiguous();
  at::Tensor mean_contig = mean.contiguous();
  at::Tensor invstd_contig = invstd.contiguous();
  at::TensorOptions grad_options =
      x_contig.options().dtype(at::kFloat).requires_grad(false);
  at::Tensor dscale =
      at::empty({x_contig.size(0), x_contig.size(2)}, grad_options);
  at::Tensor dshift = at::empty_like(dscale);

  c10::cuda::CUDAGuard device_guard(x_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x_contig.get_device()).stream();

  int64_t batch_size = x_contig.size(0);
  int64_t seq_len = x_contig.size(1);
  int64_t hidden_size = x_contig.size(2);

  switch (x_contig.scalar_type()) {
  case at::kHalf:
    launch_fused_adaln_backward_dparam(
        dy_contig.data_ptr<at::Half>(), x_contig.data_ptr<at::Half>(),
        mean_contig.data_ptr<float>(), invstd_contig.data_ptr<float>(),
        dscale.data_ptr<float>(), dshift.data_ptr<float>(), batch_size, seq_len,
        hidden_size, stream);
    check_launch(cudaGetLastError(), "fused_adaln_backward_dparam<fp16>");
    break;
  case at::kBFloat16:
    launch_fused_adaln_backward_dparam(
        dy_contig.data_ptr<at::BFloat16>(), x_contig.data_ptr<at::BFloat16>(),
        mean_contig.data_ptr<float>(), invstd_contig.data_ptr<float>(),
        dscale.data_ptr<float>(), dshift.data_ptr<float>(), batch_size, seq_len,
        hidden_size, stream);
    check_launch(cudaGetLastError(), "fused_adaln_backward_dparam<bf16>");
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused_adaln_backward_dparam");
  }

  return {dscale, dshift};
}
