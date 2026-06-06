#include <torch/extension.h>

at::Tensor fused_out_residual_forward(
    const at::Tensor& x,
    const at::Tensor& residual,
    c10::optional<at::Tensor> bias) {
    TORCH_CHECK(false, "fused_out_residual_forward is scaffolded but not implemented yet");
}
