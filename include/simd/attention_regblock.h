#pragma once
#include <immintrin.h>

inline float dot_regblock(
    const float* q,
    const float* k,
    int Dh)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    for(int d=0; d<Dh; d+=32)
    {
        __m512 q0 = _mm512_loadu_ps(q+d);
        __m512 k0 = _mm512_loadu_ps(k+d);
        acc0 = _mm512_fmadd_ps(q0,k0,acc0);

        __m512 q1 = _mm512_loadu_ps(q+d+16);
        __m512 k1 = _mm512_loadu_ps(k+d+16);
        acc1 = _mm512_fmadd_ps(q1,k1,acc1);
    }

    return _mm512_reduce_add_ps(acc0)
         + _mm512_reduce_add_ps(acc1);
}