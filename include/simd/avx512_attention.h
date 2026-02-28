#pragma once
#include <immintrin.h>

inline float dot16_avx512(const float* a,
                          const float* b)
{
    __m512 va = _mm512_loadu_ps(a);
    __m512 vb = _mm512_loadu_ps(b);
    __m512 sum = _mm512_mul_ps(va,vb);
    return _mm512_reduce_add_ps(sum);
}