#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kActSilu = 0;
constexpr int64_t kActGelu = 1;

void check_fc1_bias_act_inputs(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    int64_t act_id) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.dim() == 3, "x must have shape [batch, seq, hidden]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [2 * intermediate, hidden]");
  TORCH_CHECK(weight.size(0) % 2 == 0, "weight.size(0) must be even");
  TORCH_CHECK(
      weight.size(1) == x.size(2),
      "weight hidden dimension must match x.size(2)");
  TORCH_CHECK(
      x.scalar_type() == weight.scalar_type(),
      "x and weight must have the same dtype");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "fused_fc1_bias_act_forward supports fp16 and bf16 inputs");
  TORCH_CHECK(
      act_id == kActSilu || act_id == kActGelu,
      "act_id must be 0 (SiLU) or 1 (GELU)");

  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dim() == 1, "bias must have shape [2 * intermediate]");
    TORCH_CHECK(
        bias->size(0) == weight.size(0),
        "bias.size(0) must match weight.size(0)");
    TORCH_CHECK(
        bias->scalar_type() == x.scalar_type(),
        "bias dtype must match x dtype");
  }
}

}  // namespace

at::Tensor fused_fc1_bias_act_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias,
    int64_t act_id) {
  check_fc1_bias_act_inputs(x, weight, bias, act_id);

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t intermediate_size = weight.size(0) / 2;
  auto y = at::empty({x.size(0), x.size(1), intermediate_size}, x.options());

  TORCH_CHECK(
      false,
      "fused_fc1_bias_act_forward is scaffolded for the CuTe DSL AOT kernel, "
      "but the generated launcher is not wired yet. See "
      "csrc/cute_dsl/fc1_bias_act/ for the DSL source and export entry.");

  return y;
}
