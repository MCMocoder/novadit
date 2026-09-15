#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> fused_adaln_forward(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& shift,
    double eps);

at::Tensor fused_adaln_forward_inference(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& shift,
    double eps);

at::Tensor fused_adaln_backward_dx(
    const at::Tensor& dy,
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& mean,
    const at::Tensor& invstd);

std::vector<at::Tensor> fused_adaln_backward_dparam(
    const at::Tensor& dy,
    const at::Tensor& x,
    const at::Tensor& mean,
    const at::Tensor& invstd);

std::vector<at::Tensor> fused_qkv_pack_forward(
    const at::Tensor& qkv,
    int64_t num_heads,
    int64_t head_dim);

at::Tensor fused_out_residual_forward(
    const at::Tensor& x,
    const at::Tensor& residual,
    c10::optional<at::Tensor> bias);

at::Tensor fused_geglu_forward(
    const at::Tensor& z);

at::Tensor fused_geglu_backward(
    const at::Tensor& dy,
    const at::Tensor& z);

at::Tensor fused_gate_act_forward(
    const at::Tensor& x,
    int64_t act_id);

at::Tensor fused_gate_act_backward(
    const at::Tensor& dy,
    const at::Tensor& x,
    int64_t act_id);

std::vector<at::Tensor> fused_adarmsnorm_forward(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& shift,
    double eps);

at::Tensor fused_adarmsnorm_forward_inference(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& shift,
    double eps);

at::Tensor fused_adarmsnorm_backward_dx(
    const at::Tensor& dy,
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& inv_rms);

std::vector<at::Tensor> fused_adarmsnorm_backward_dparam(
    const at::Tensor& dy,
    const at::Tensor& x,
    const at::Tensor& inv_rms);

at::Tensor fused_fc1_bias_act_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias,
    int64_t act_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_adaln_forward", &fused_adaln_forward,
        "NovaDiT fused AdaLN forward");
  m.def("fused_adaln_forward_inference", &fused_adaln_forward_inference,
        "NovaDiT fused AdaLN forward (inference-only, returns y without stats)");
  m.def("fused_adaln_backward_dx", &fused_adaln_backward_dx,
        "NovaDiT fused AdaLN backward dx");
  m.def("fused_adaln_backward_dparam", &fused_adaln_backward_dparam,
        "NovaDiT fused AdaLN backward dscale/dshift");
  m.def("fused_qkv_pack_forward", &fused_qkv_pack_forward,
        "NovaDiT fused QKV pack forward");
  m.def("fused_out_residual_forward", &fused_out_residual_forward,
        "NovaDiT fused out residual forward");
  m.def("fused_geglu_forward", &fused_geglu_forward,
        "NovaDiT fused GeGLU forward (GELU-gated, tanh approx)");
  m.def("fused_geglu_backward", &fused_geglu_backward,
        "NovaDiT fused GeGLU backward");
  m.def("fused_gate_act_forward", &fused_gate_act_forward,
        "NovaDiT fused gate act forward (act_id: 0=SiLU, 1=GELU)");
  m.def("fused_gate_act_backward", &fused_gate_act_backward,
        "NovaDiT fused gate act backward (act_id: 0=SiLU, 1=GELU)");
  m.def("fused_adarmsnorm_forward", &fused_adarmsnorm_forward,
        "NovaDiT fused AdaRMSNorm forward");
  m.def("fused_adarmsnorm_forward_inference", &fused_adarmsnorm_forward_inference,
        "NovaDiT fused AdaRMSNorm forward (inference-only, returns y without inv_rms)");
  m.def("fused_adarmsnorm_backward_dx", &fused_adarmsnorm_backward_dx,
        "NovaDiT fused AdaRMSNorm backward dx");
  m.def("fused_adarmsnorm_backward_dparam", &fused_adarmsnorm_backward_dparam,
        "NovaDiT fused AdaRMSNorm backward dscale/dshift");
  m.def("fused_fc1_bias_act_forward", &fused_fc1_bias_act_forward,
        "NovaDiT fused FC1 + bias + gated activation forward (CUTLASS SM90)");
}
