#pragma once

#include <torch/extension.h>

#define NOVADIT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_FLOATING_TYPES_AND2(at::Half, at::BFloat16, TYPE, NAME, __VA_ARGS__)
