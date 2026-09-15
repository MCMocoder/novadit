#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace novadit::cute_dsl::fc1_bias_act {

enum class Activation : int32_t {
  kSilu = 0,
  kGelu = 1,
};

// Fused gated FC1 + activation kernel: y = (x @ w_value^T) * act(x @ w_gate^T).
// The interleaved fc1 weight is split into a gate half and a value half before
// being passed in. There is no bias and no FC2 in this kernel (FC1+Act only).
struct ForwardParams {
  void const* x = nullptr;
  void const* fc1_gate_weight = nullptr;
  void const* fc1_value_weight = nullptr;
  void* y = nullptr;

  int32_t token_count = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  Activation activation = Activation::kSilu;
};

inline bool has_aot_forward() {
  return false;
}

inline cudaError_t launch_aot_forward(
    ForwardParams const&,
    cudaStream_t) {
  return cudaErrorNotSupported;
}

}  // namespace novadit::cute_dsl::fc1_bias_act
