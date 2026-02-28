#pragma once
#include <immintrin.h>

inline float dot_avx512(const float* a,
                        const float* b,
                        int n)
{
    __m512 sum = _mm512_setzero_ps();

    for(int i=0;i<n;i+=16)
    {
        __m512 va = _mm512_loadu_ps(a+i);
        __m512 vb = _mm512_loadu_ps(b+i);
        sum = _mm512_fmadd_ps(va,vb,sum);
    }

    float result = _mm512_reduce_add_ps(sum);
    return result;
}