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
};

struct GeluActivation {
  __device__ __forceinline__ static float fwd(float t) {
    // tanh approximation: GELU(t) ≈ 0.5 * t * (1 + tanh(√(2/π) * (t + 0.044715*t³)))
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    // 0.044715 as a float
    float t_cube = t * t * t;
    float inner = kSqrt2OverPi * (t + 0.044715f * t_cube);
    return 0.5f * t * (1.0f + tanhf(inner));
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
