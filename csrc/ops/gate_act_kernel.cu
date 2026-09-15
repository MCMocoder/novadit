#include <torch/extension.h>
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

namespace {

constexpr int kGateActThreads = 256;
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

template <typename scalar_t>
__device__ __forceinline__ bool is_aligned_ptr_device(scalar_t const *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignof(VecType<scalar_t>)) == 0;
}

// ---- Activation functions (device-only) ----

struct SiluActivation {
  __device__ __forceinline__ static float fwd(float t) {
    float sig = 1.0f / (1.0f + expf(-t));
    return t * sig;
  }

  __device__ __forceinline__ static float bwd(float t) {
    // SiLU'(t) = σ(t) * (1 + t * (1 - σ(t)))
    float sig = 1.0f / (1.0f + expf(-t));
    return sig * (1.0f + t * (1.0f - sig));
  }
};

struct GeluActivation {
  __device__ __forceinline__ static float fwd(float t) {
    // tanh approximation: GELU(t) ≈ 0.5 * t * (1 + tanh(√(2/π) * (t + 0.044715*t³)))
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    float t_cube = t * t * t;
    float inner = kSqrt2OverPi * (t + 0.044715f * t_cube);
    return 0.5f * t * (1.0f + tanhf(inner));
  }

  __device__ __forceinline__ static float bwd(float t) {
    // GELU'(t) = 0.5*(1+tanh(g)) + 0.5*t*(1-tanh²(g))*g'
    // where g = α*(t+β*t³), g' = α*(1+3β*t²)
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    float t2 = t * t;
    float g = kAlpha * (t + kBeta * t * t2);
    float tanh_g = tanhf(g);
    float sech2 = 1.0f - tanh_g * tanh_g;
    float g_prime = kAlpha * (1.0f + 3.0f * kBeta * t2);
    return 0.5f * (1.0f + tanh_g) + 0.5f * t * sech2 * g_prime;
  }
};

// ---- Kernels ----

template <typename scalar_t, typename Activation, int InterSize>
__global__ void fused_gate_act_aligned_kernel(
    scalar_t const *z, scalar_t *y,
    int64_t batch_size, int64_t seq_len) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kVecCount = InterSize / kVecElems;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) return;

  int64_t z_offset = row_idx * 2 * InterSize;
  int64_t y_offset = row_idx * InterSize;

  scalar_t const *z_up = z + z_offset;
  scalar_t const *z_gate = z + z_offset + InterSize;
  scalar_t *y_row = y + y_offset;

  auto z_up_vec = as_vec_ptr(z_up);
  auto z_gate_vec = as_vec_ptr(z_gate);
  auto y_vec = as_vec_ptr(y_row);

  for (int vec_idx = threadIdx.x; vec_idx < kVecCount;
       vec_idx += blockDim.x) {
    VecType<scalar_t> up_frag = z_up_vec[vec_idx];
    VecType<scalar_t> gate_frag = z_gate_vec[vec_idx];
    VecType<scalar_t> out_frag;

#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      float up_val = to_float(up_frag[i]);
      float gate_val = to_float(gate_frag[i]);
      float activated = Activation::fwd(gate_val);
      out_frag[i] = from_float<scalar_t>(up_val * activated);
    }

    y_vec[vec_idx] = out_frag;
  }
}

template <typename scalar_t, typename Activation>
__global__ void fused_gate_act_dynamic_kernel(
    scalar_t const *z, scalar_t *y,
    int64_t batch_size, int64_t seq_len, int64_t inter_size) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) return;

  int64_t z_offset = row_idx * 2 * inter_size;
  int64_t y_offset = row_idx * inter_size;

  scalar_t const *z_up = z + z_offset;
  scalar_t const *z_gate = z + z_offset + inter_size;
  scalar_t *y_row = y + y_offset;

  int64_t vec_prefix_elems = (inter_size / kVecElems) * kVecElems;
  int64_t vec_count = vec_prefix_elems / kVecElems;
  bool row_aligned = vec_prefix_elems > 0 &&
                     is_aligned_ptr_device(z_up) &&
                     is_aligned_ptr_device(z_gate) &&
                     is_aligned_ptr_device(y_row);
  int64_t scalar_start = row_aligned ? vec_prefix_elems : 0;

  if (row_aligned) {
    auto z_up_vec = as_vec_ptr(z_up);
    auto z_gate_vec = as_vec_ptr(z_gate);
    auto y_vec = as_vec_ptr(y_row);

    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> up_frag = z_up_vec[vec_idx];
      VecType<scalar_t> gate_frag = z_gate_vec[vec_idx];
      VecType<scalar_t> out_frag;

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float up_val = to_float(up_frag[i]);
        float gate_val = to_float(gate_frag[i]);
        float activated = Activation::fwd(gate_val);
        out_frag[i] = from_float<scalar_t>(up_val * activated);
      }

      y_vec[vec_idx] = out_frag;
    }
  }

  for (int64_t i = scalar_start + threadIdx.x; i < inter_size;
       i += blockDim.x) {
    float up_val = to_float(z_up[i]);
    float gate_val = to_float(z_gate[i]);
    y_row[i] = from_float<scalar_t>(up_val * Activation::fwd(gate_val));
  }
}

// ---- Backward kernels ----

template <typename scalar_t, typename Activation, int InterSize>
__global__ void fused_gate_act_backward_aligned_kernel(
    scalar_t const *dy, scalar_t const *z,
    scalar_t *dz, int64_t batch_size, int64_t seq_len) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  constexpr int kVecCount = InterSize / kVecElems;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) return;

  int64_t z_offset = row_idx * 2 * InterSize;
  int64_t dy_offset = row_idx * InterSize;
  int64_t dz_offset = row_idx * 2 * InterSize;

  scalar_t const *dy_row = dy + dy_offset;
  scalar_t const *z_up = z + z_offset;
  scalar_t const *z_gate = z + z_offset + InterSize;
  scalar_t *dz_up = dz + dz_offset;
  scalar_t *dz_gate = dz + dz_offset + InterSize;

  auto dy_vec = as_vec_ptr(dy_row);
  auto up_vec = as_vec_ptr(z_up);
  auto gate_vec = as_vec_ptr(z_gate);
  auto dz_up_vec = as_vec_ptr(dz_up);
  auto dz_gate_vec = as_vec_ptr(dz_gate);

  for (int vec_idx = threadIdx.x; vec_idx < kVecCount;
       vec_idx += blockDim.x) {
    VecType<scalar_t> dy_frag = dy_vec[vec_idx];
    VecType<scalar_t> up_frag = up_vec[vec_idx];
    VecType<scalar_t> gate_frag = gate_vec[vec_idx];
    VecType<scalar_t> dz_up_frag;
    VecType<scalar_t> dz_gate_frag;

#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      float dy_val = to_float(dy_frag[i]);
      float up_val = to_float(up_frag[i]);
      float gate_val = to_float(gate_frag[i]);
      dz_up_frag[i] = from_float<scalar_t>(dy_val * Activation::fwd(gate_val));
      dz_gate_frag[i] = from_float<scalar_t>(dy_val * up_val * Activation::bwd(gate_val));
    }

    dz_up_vec[vec_idx] = dz_up_frag;
    dz_gate_vec[vec_idx] = dz_gate_frag;
  }
}

template <typename scalar_t, typename Activation>
__global__ void fused_gate_act_backward_dynamic_kernel(
    scalar_t const *dy, scalar_t const *z,
    scalar_t *dz, int64_t batch_size, int64_t seq_len,
    int64_t inter_size) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;

  int64_t row_idx = static_cast<int64_t>(blockIdx.x);
  int64_t row_count = batch_size * seq_len;
  if (row_idx >= row_count) return;

  int64_t z_offset = row_idx * 2 * inter_size;
  int64_t dy_offset = row_idx * inter_size;
  int64_t dz_offset = row_idx * 2 * inter_size;

  scalar_t const *dy_row = dy + dy_offset;
  scalar_t const *z_up = z + z_offset;
  scalar_t const *z_gate = z + z_offset + inter_size;
  scalar_t *dz_up = dz + dz_offset;
  scalar_t *dz_gate = dz + dz_offset + inter_size;

  int64_t vec_prefix_elems = (inter_size / kVecElems) * kVecElems;
  int64_t vec_count = vec_prefix_elems / kVecElems;
  bool row_aligned = vec_prefix_elems > 0 &&
                     is_aligned_ptr_device(dy_row) &&
                     is_aligned_ptr_device(z_up) &&
                     is_aligned_ptr_device(z_gate) &&
                     is_aligned_ptr_device(dz_up) &&
                     is_aligned_ptr_device(dz_gate);
  int64_t scalar_start = row_aligned ? vec_prefix_elems : 0;

  if (row_aligned) {
    auto dy_vec = as_vec_ptr(dy_row);
    auto up_vec = as_vec_ptr(z_up);
    auto gate_vec = as_vec_ptr(z_gate);
    auto dz_up_vec = as_vec_ptr(dz_up);
    auto dz_gate_vec = as_vec_ptr(dz_gate);

    for (int vec_idx = threadIdx.x; vec_idx < vec_count;
         vec_idx += blockDim.x) {
      VecType<scalar_t> dy_frag = dy_vec[vec_idx];
      VecType<scalar_t> up_frag = up_vec[vec_idx];
      VecType<scalar_t> gate_frag = gate_vec[vec_idx];
      VecType<scalar_t> dz_up_frag;
      VecType<scalar_t> dz_gate_frag;

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float dy_val = to_float(dy_frag[i]);
        float up_val = to_float(up_frag[i]);
        float gate_val = to_float(gate_frag[i]);
        dz_up_frag[i] = from_float<scalar_t>(dy_val * Activation::fwd(gate_val));
        dz_gate_frag[i] = from_float<scalar_t>(dy_val * up_val * Activation::bwd(gate_val));
      }

      dz_up_vec[vec_idx] = dz_up_frag;
      dz_gate_vec[vec_idx] = dz_gate_frag;
    }
  }

  for (int64_t i = scalar_start + threadIdx.x; i < inter_size;
       i += blockDim.x) {
    float dy_val = to_float(dy_row[i]);
    float up_val = to_float(z_up[i]);
    float gate_val = to_float(z_gate[i]);
    dz_up[i] = from_float<scalar_t>(dy_val * Activation::fwd(gate_val));
    dz_gate[i] = from_float<scalar_t>(dy_val * up_val * Activation::bwd(gate_val));
  }
}

// ---- Launch helpers ----

template <typename scalar_t, typename Activation, int InterSize>
void launch_aligned(scalar_t const *z, scalar_t *y,
                    int64_t batch_size, int64_t seq_len,
                    cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_gate_act_aligned_kernel<scalar_t, Activation, InterSize>
      <<<row_count, kGateActThreads, 0, stream>>>(
          z, y, batch_size, seq_len);
}

template <typename scalar_t, typename Activation>
void launch_dynamic(scalar_t const *z, scalar_t *y,
                    int64_t batch_size, int64_t seq_len, int64_t inter_size,
                    cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_gate_act_dynamic_kernel<scalar_t, Activation>
      <<<row_count, kGateActThreads, 0, stream>>>(
          z, y, batch_size, seq_len, inter_size);
}

template <typename scalar_t>
bool base_ptrs_support_aligned(scalar_t const *z, scalar_t *y,
                               int64_t inter_size) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  if (inter_size % kVecElems != 0) return false;
  uintptr_t z_ptr = reinterpret_cast<uintptr_t>(z);
  uintptr_t y_ptr = reinterpret_cast<uintptr_t>(y);
  constexpr size_t kAlign = alignof(VecType<scalar_t>);
  // Check z base, z gate half, and y base alignment
  if (z_ptr % kAlign != 0) return false;
  if ((z_ptr + inter_size * sizeof(scalar_t)) % kAlign != 0) return false;
  if (y_ptr % kAlign != 0) return false;
  return true;
}

template <typename scalar_t, typename Activation>
void dispatch_aligned_or_dynamic(scalar_t const *z, scalar_t *y,
                                 int64_t batch_size, int64_t seq_len,
                                 int64_t inter_size, cudaStream_t stream) {
  if (!base_ptrs_support_aligned(z, y, inter_size)) {
    launch_dynamic<scalar_t, Activation>(z, y, batch_size, seq_len, inter_size,
                                         stream);
    return;
  }

  switch (inter_size) {
  case 128:
    launch_aligned<scalar_t, Activation, 128>(z, y, batch_size, seq_len,
                                              stream);
    break;
  case 256:
    launch_aligned<scalar_t, Activation, 256>(z, y, batch_size, seq_len,
                                              stream);
    break;
  case 512:
    launch_aligned<scalar_t, Activation, 512>(z, y, batch_size, seq_len,
                                              stream);
    break;
  case 768:
    launch_aligned<scalar_t, Activation, 768>(z, y, batch_size, seq_len,
                                              stream);
    break;
  case 1024:
    launch_aligned<scalar_t, Activation, 1024>(z, y, batch_size, seq_len,
                                               stream);
    break;
  case 1152:
    launch_aligned<scalar_t, Activation, 1152>(z, y, batch_size, seq_len,
                                               stream);
    break;
  case 1536:
    launch_aligned<scalar_t, Activation, 1536>(z, y, batch_size, seq_len,
                                               stream);
    break;
  case 2048:
    launch_aligned<scalar_t, Activation, 2048>(z, y, batch_size, seq_len,
                                               stream);
    break;
  case 3072:
    launch_aligned<scalar_t, Activation, 3072>(z, y, batch_size, seq_len,
                                               stream);
    break;
  case 4096:
    launch_aligned<scalar_t, Activation, 4096>(z, y, batch_size, seq_len,
                                               stream);
    break;
  case 8192:
    launch_aligned<scalar_t, Activation, 8192>(z, y, batch_size, seq_len,
                                               stream);
    break;
  default:
    launch_dynamic<scalar_t, Activation>(z, y, batch_size, seq_len, inter_size,
                                         stream);
    break;
  }
}

template <typename scalar_t, typename Activation>
void launch_fused_gate_act(scalar_t const *z, scalar_t *y,
                           int64_t batch_size, int64_t seq_len,
                           int64_t inter_size, cudaStream_t stream) {
  dispatch_aligned_or_dynamic<scalar_t, Activation>(
      z, y, batch_size, seq_len, inter_size, stream);
}

// ---- Backward launch helpers ----

template <typename scalar_t, typename Activation, int InterSize>
void launch_backward_aligned(scalar_t const *dy, scalar_t const *z,
                             scalar_t *dz, int64_t batch_size,
                             int64_t seq_len, cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_gate_act_backward_aligned_kernel<scalar_t, Activation, InterSize>
      <<<row_count, kGateActThreads, 0, stream>>>(
          dy, z, dz, batch_size, seq_len);
}

template <typename scalar_t, typename Activation>
void launch_backward_dynamic(scalar_t const *dy, scalar_t const *z,
                             scalar_t *dz, int64_t batch_size,
                             int64_t seq_len, int64_t inter_size,
                             cudaStream_t stream) {
  int64_t row_count = batch_size * seq_len;
  fused_gate_act_backward_dynamic_kernel<scalar_t, Activation>
      <<<row_count, kGateActThreads, 0, stream>>>(
          dy, z, dz, batch_size, seq_len, inter_size);
}

template <typename scalar_t>
bool base_ptrs_support_backward_aligned(scalar_t const *dy, scalar_t const *z,
                                        scalar_t *dz, int64_t inter_size) {
  constexpr int kVecElems = VecWidth<scalar_t>::kElements;
  if (inter_size % kVecElems != 0) return false;
  constexpr size_t kAlign = alignof(VecType<scalar_t>);
  uintptr_t dy_ptr = reinterpret_cast<uintptr_t>(dy);
  uintptr_t dz_ptr = reinterpret_cast<uintptr_t>(dz);
  uintptr_t z_ptr = reinterpret_cast<uintptr_t>(z);
  if (dy_ptr % kAlign != 0) return false;
  if (dz_ptr % kAlign != 0) return false;
  if (z_ptr % kAlign != 0) return false;
  if ((z_ptr + inter_size * sizeof(scalar_t)) % kAlign != 0) return false;
  if ((dz_ptr + inter_size * sizeof(scalar_t)) % kAlign != 0) return false;
  return true;
}

template <typename scalar_t, typename Activation>
void dispatch_backward_aligned_or_dynamic(
    scalar_t const *dy, scalar_t const *z, scalar_t *dz,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  if (!base_ptrs_support_backward_aligned(dy, z, dz, inter_size)) {
    launch_backward_dynamic<scalar_t, Activation>(
        dy, z, dz, batch_size, seq_len, inter_size, stream);
    return;
  }

  switch (inter_size) {
  case 128:
    launch_backward_aligned<scalar_t, Activation, 128>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 256:
    launch_backward_aligned<scalar_t, Activation, 256>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 512:
    launch_backward_aligned<scalar_t, Activation, 512>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 768:
    launch_backward_aligned<scalar_t, Activation, 768>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 1024:
    launch_backward_aligned<scalar_t, Activation, 1024>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 1152:
    launch_backward_aligned<scalar_t, Activation, 1152>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 1536:
    launch_backward_aligned<scalar_t, Activation, 1536>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 2048:
    launch_backward_aligned<scalar_t, Activation, 2048>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 3072:
    launch_backward_aligned<scalar_t, Activation, 3072>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 4096:
    launch_backward_aligned<scalar_t, Activation, 4096>(
        dy, z, dz, batch_size, seq_len, stream); break;
  case 8192:
    launch_backward_aligned<scalar_t, Activation, 8192>(
        dy, z, dz, batch_size, seq_len, stream); break;
  default:
    launch_backward_dynamic<scalar_t, Activation>(
        dy, z, dz, batch_size, seq_len, inter_size, stream);
    break;
  }
}

template <typename scalar_t, typename Activation>
void launch_fused_gate_act_backward(
    scalar_t const *dy, scalar_t const *z, scalar_t *dz,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  dispatch_backward_aligned_or_dynamic<scalar_t, Activation>(
      dy, z, dz, batch_size, seq_len, inter_size, stream);
}

// ---- Validation ----

void check_gate_act_inputs(at::Tensor const &z) {
  NOVADIT_CHECK_INPUT(z);
  TORCH_CHECK(z.dim() == 3, "z must have shape [B, S, 2*I]");
  TORCH_CHECK(z.size(2) % 2 == 0,
              "z last dim must be even (2 * intermediate_size)");
}

void check_launch(cudaError_t error, char const *kernel_name) {
  TORCH_CHECK(error == cudaSuccess, kernel_name,
              " launch failed: ", cudaGetErrorString(error));
}

} // namespace

// ---- C-linkage forward entry points ----

void fused_geglu_fp16_forward(
    at::Half const *z, at::Half *y,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act<at::Half, GeluActivation>(
      z, y, batch_size, seq_len, inter_size, stream);
}

void fused_geglu_bf16_forward(
    at::BFloat16 const *z, at::BFloat16 *y,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act<at::BFloat16, GeluActivation>(
      z, y, batch_size, seq_len, inter_size, stream);
}

void fused_silu_gate_fp16_forward(
    at::Half const *z, at::Half *y,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act<at::Half, SiluActivation>(
      z, y, batch_size, seq_len, inter_size, stream);
}

void fused_silu_gate_bf16_forward(
    at::BFloat16 const *z, at::BFloat16 *y,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act<at::BFloat16, SiluActivation>(
      z, y, batch_size, seq_len, inter_size, stream);
}

// ---- C-linkage backward entry points ----

void fused_geglu_fp16_backward(
    at::Half const *dy, at::Half const *z, at::Half *dz,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act_backward<at::Half, GeluActivation>(
      dy, z, dz, batch_size, seq_len, inter_size, stream);
}

void fused_geglu_bf16_backward(
    at::BFloat16 const *dy, at::BFloat16 const *z, at::BFloat16 *dz,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act_backward<at::BFloat16, GeluActivation>(
      dy, z, dz, batch_size, seq_len, inter_size, stream);
}

void fused_silu_gate_fp16_backward(
    at::Half const *dy, at::Half const *z, at::Half *dz,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act_backward<at::Half, SiluActivation>(
      dy, z, dz, batch_size, seq_len, inter_size, stream);
}

void fused_silu_gate_bf16_backward(
    at::BFloat16 const *dy, at::BFloat16 const *z, at::BFloat16 *dz,
    int64_t batch_size, int64_t seq_len, int64_t inter_size,
    cudaStream_t stream) {
  launch_fused_gate_act_backward<at::BFloat16, SiluActivation>(
      dy, z, dz, batch_size, seq_len, inter_size, stream);
}

// ---- Torch entry point: fused_geglu_forward ----

at::Tensor fused_geglu_forward(const at::Tensor &z) {
  check_gate_act_inputs(z);
  TORCH_CHECK(z.scalar_type() == at::kFloat ||
                  z.scalar_type() == at::kHalf ||
                  z.scalar_type() == at::kBFloat16,
              "fused_geglu_forward only supports fp32/fp16/bf16 tensors");

  at::Tensor z_contig = z.contiguous();
  int64_t batch_size = z_contig.size(0);
  int64_t seq_len = z_contig.size(1);
  int64_t inter_size = z_contig.size(2) / 2;
  at::Tensor y = at::empty({batch_size, seq_len, inter_size},
                           z_contig.options());

  c10::cuda::CUDAGuard device_guard(z_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(z_contig.get_device()).stream();

  switch (z_contig.scalar_type()) {
  case at::kFloat:
    launch_fused_gate_act<float, GeluActivation>(
        z_contig.data_ptr<float>(), y.data_ptr<float>(),
        batch_size, seq_len, inter_size, stream);
    check_launch(cudaGetLastError(), "fused_geglu_forward<float>");
    break;
  case at::kHalf:
    fused_geglu_fp16_forward(
        z_contig.data_ptr<at::Half>(), y.data_ptr<at::Half>(),
        batch_size, seq_len, inter_size, stream);
    check_launch(cudaGetLastError(), "fused_geglu_forward<fp16>");
    break;
  case at::kBFloat16:
    fused_geglu_bf16_forward(
        z_contig.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>(),
        batch_size, seq_len, inter_size, stream);
    check_launch(cudaGetLastError(), "fused_geglu_forward<bf16>");
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused_geglu_forward");
  }

  return y;
}

// ---- Torch entry point: fused_gate_act_forward (act_id = 0 -> SiLU, 1 -> GELU) ----

at::Tensor fused_gate_act_forward(const at::Tensor &x, int64_t act_id) {
  check_gate_act_inputs(x);
  TORCH_CHECK(act_id == 0 || act_id == 1,
              "act_id must be 0 (SiLU) or 1 (GELU), got ", act_id);
  TORCH_CHECK(x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "fused_gate_act_forward only supports fp32/fp16/bf16 tensors");

  at::Tensor x_contig = x.contiguous();
  int64_t batch_size = x_contig.size(0);
  int64_t seq_len = x_contig.size(1);
  int64_t inter_size = x_contig.size(2) / 2;
  at::Tensor y = at::empty({batch_size, seq_len, inter_size},
                           x_contig.options());

  c10::cuda::CUDAGuard device_guard(x_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x_contig.get_device()).stream();

  if (act_id == 0) {
    // SiLU-gated
    switch (x_contig.scalar_type()) {
    case at::kFloat:
      launch_fused_gate_act<float, SiluActivation>(
          x_contig.data_ptr<float>(), y.data_ptr<float>(),
          batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_forward<silu,float>");
      break;
    case at::kHalf:
      fused_silu_gate_fp16_forward(
          x_contig.data_ptr<at::Half>(), y.data_ptr<at::Half>(),
          batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_forward<silu,fp16>");
      break;
    case at::kBFloat16:
      fused_silu_gate_bf16_forward(
          x_contig.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>(),
          batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_forward<silu,bf16>");
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype");
    }
  } else {
    // GELU-gated (act_id == 1)
    switch (x_contig.scalar_type()) {
    case at::kFloat:
      launch_fused_gate_act<float, GeluActivation>(
          x_contig.data_ptr<float>(), y.data_ptr<float>(),
          batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_forward<gelu,float>");
      break;
    case at::kHalf:
      fused_geglu_fp16_forward(
          x_contig.data_ptr<at::Half>(), y.data_ptr<at::Half>(),
          batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_forward<gelu,fp16>");
      break;
    case at::kBFloat16:
      fused_geglu_bf16_forward(
          x_contig.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>(),
          batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_forward<gelu,bf16>");
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype");
    }
  }

  return y;
}

// ---- Torch entry point: fused_geglu_backward ----

at::Tensor fused_geglu_backward(const at::Tensor &dy, const at::Tensor &z) {
  check_gate_act_inputs(z);
  TORCH_CHECK(dy.dim() == 3, "dy must have shape [B, S, I]");
  TORCH_CHECK(dy.size(0) == z.size(0) && dy.size(1) == z.size(1),
              "dy batch/seq dims must match z");
  TORCH_CHECK(dy.size(2) * 2 == z.size(2),
              "dy last dim must be half of z last dim");
  TORCH_CHECK(dy.scalar_type() == z.scalar_type(),
              "dy and z must have the same dtype");
  TORCH_CHECK(z.scalar_type() == at::kFloat ||
                  z.scalar_type() == at::kHalf ||
                  z.scalar_type() == at::kBFloat16,
              "fused_geglu_backward only supports fp32/fp16/bf16 tensors");

  at::Tensor z_contig = z.contiguous();
  at::Tensor dy_contig = dy.contiguous();
  int64_t batch_size = z_contig.size(0);
  int64_t seq_len = z_contig.size(1);
  int64_t inter_size = z_contig.size(2) / 2;
  at::Tensor dz = at::empty_like(z_contig);

  c10::cuda::CUDAGuard device_guard(z_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(z_contig.get_device()).stream();

  switch (z_contig.scalar_type()) {
  case at::kFloat:
    launch_fused_gate_act_backward<float, GeluActivation>(
        dy_contig.data_ptr<float>(), z_contig.data_ptr<float>(),
        dz.data_ptr<float>(), batch_size, seq_len, inter_size, stream);
    check_launch(cudaGetLastError(), "fused_geglu_backward<float>");
    break;
  case at::kHalf:
    fused_geglu_fp16_backward(
        dy_contig.data_ptr<at::Half>(), z_contig.data_ptr<at::Half>(),
        dz.data_ptr<at::Half>(), batch_size, seq_len, inter_size, stream);
    check_launch(cudaGetLastError(), "fused_geglu_backward<fp16>");
    break;
  case at::kBFloat16:
    fused_geglu_bf16_backward(
        dy_contig.data_ptr<at::BFloat16>(), z_contig.data_ptr<at::BFloat16>(),
        dz.data_ptr<at::BFloat16>(), batch_size, seq_len, inter_size, stream);
    check_launch(cudaGetLastError(), "fused_geglu_backward<bf16>");
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused_geglu_backward");
  }

  return dz;
}

// ---- Torch entry point: fused_gate_act_backward ----

at::Tensor fused_gate_act_backward(const at::Tensor &dy,
                                   const at::Tensor &x, int64_t act_id) {
  check_gate_act_inputs(x);
  TORCH_CHECK(act_id == 0 || act_id == 1,
              "act_id must be 0 (SiLU) or 1 (GELU), got ", act_id);
  TORCH_CHECK(dy.dim() == 3, "dy must have shape [B, S, I]");
  TORCH_CHECK(dy.size(0) == x.size(0) && dy.size(1) == x.size(1),
              "dy batch/seq dims must match x");
  TORCH_CHECK(dy.size(2) * 2 == x.size(2),
              "dy last dim must be half of x last dim");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type(),
              "dy and x must have the same dtype");
  TORCH_CHECK(x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "fused_gate_act_backward only supports fp32/fp16/bf16 tensors");

  at::Tensor x_contig = x.contiguous();
  at::Tensor dy_contig = dy.contiguous();
  int64_t batch_size = x_contig.size(0);
  int64_t seq_len = x_contig.size(1);
  int64_t inter_size = x_contig.size(2) / 2;
  at::Tensor dz = at::empty_like(x_contig);

  c10::cuda::CUDAGuard device_guard(x_contig.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x_contig.get_device()).stream();

  if (act_id == 0) {
    switch (x_contig.scalar_type()) {
    case at::kFloat:
      launch_fused_gate_act_backward<float, SiluActivation>(
          dy_contig.data_ptr<float>(), x_contig.data_ptr<float>(),
          dz.data_ptr<float>(), batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_backward<silu,float>");
      break;
    case at::kHalf:
      fused_silu_gate_fp16_backward(
          dy_contig.data_ptr<at::Half>(), x_contig.data_ptr<at::Half>(),
          dz.data_ptr<at::Half>(), batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_backward<silu,fp16>");
      break;
    case at::kBFloat16:
      fused_silu_gate_bf16_backward(
          dy_contig.data_ptr<at::BFloat16>(),
          x_contig.data_ptr<at::BFloat16>(),
          dz.data_ptr<at::BFloat16>(), batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_backward<silu,bf16>");
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype");
    }
  } else {
    switch (x_contig.scalar_type()) {
    case at::kFloat:
      launch_fused_gate_act_backward<float, GeluActivation>(
          dy_contig.data_ptr<float>(), x_contig.data_ptr<float>(),
          dz.data_ptr<float>(), batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_backward<gelu,float>");
      break;
    case at::kHalf:
      fused_geglu_fp16_backward(
          dy_contig.data_ptr<at::Half>(), x_contig.data_ptr<at::Half>(),
          dz.data_ptr<at::Half>(), batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_backward<gelu,fp16>");
      break;
    case at::kBFloat16:
      fused_geglu_bf16_backward(
          dy_contig.data_ptr<at::BFloat16>(),
          x_contig.data_ptr<at::BFloat16>(),
          dz.data_ptr<at::BFloat16>(), batch_size, seq_len, inter_size, stream);
      check_launch(cudaGetLastError(), "fused_gate_act_backward<gelu,bf16>");
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype");
    }
  }

  return dz;
}
