#pragma once

#include <cstdint>

namespace novadit {

enum class ActivationType : int64_t {
    kSilu = 0,
    kGelu = 1,
};

}  // namespace novadit
