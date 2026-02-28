#include "utils/fp16.h"

namespace fp16 {

uint16_t float_to_half(float f) {
    uint32_t x = *((uint32_t*)&f);
    uint16_t h = (x >> 16) & 0x8000;
    uint32_t mantissa = x & 0x7fffff;
    uint32_t exp = (x >> 23) & 0xff;

    if (exp > 112) {
        exp -= 112;
        if (exp > 31) exp = 31;
        h |= (exp << 10) | (mantissa >> 13);
    }

    return h;
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = (h & 0x3ff) << 13;

    if (exp != 0)
        exp = exp + 112;

    uint32_t f = sign | (exp << 23) | mant;
    return *((float*)&f);
}

}