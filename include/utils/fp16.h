#pragma once
#include <cstdint>

namespace fp16 {

uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

}