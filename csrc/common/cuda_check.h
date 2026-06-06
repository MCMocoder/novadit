#pragma once

#include <torch/extension.h>

#define NOVADIT_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define NOVADIT_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define NOVADIT_CHECK_INPUT(x) \
    NOVADIT_CHECK_CUDA(x);     \
    NOVADIT_CHECK_CONTIGUOUS(x)
