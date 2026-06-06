#include <torch/extension.h>

at::Tensor fused_gate_act_forward(
    const at::Tensor& x,
    int64_t act_id) {
    TORCH_CHECK(false, "fused_gate_act_forward is scaffolded but not implemented yet");
}
