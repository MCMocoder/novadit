#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> fused_qkv_pack_forward(
    const at::Tensor& qkv,
    int64_t num_heads,
    int64_t head_dim) {
    TORCH_CHECK(false, "fused_qkv_pack_forward is scaffolded but not implemented yet");
}
