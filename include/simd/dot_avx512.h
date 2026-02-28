#pragma once
#include <immintrin.h>

inline float dot_avx512_f32(
    const float* a,
    const float* b,
    int n)
{
    __m512 acc = _mm512_setzero_ps();

    int i = 0;
    for(; i + 16 <= n; i += 16)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }

    float sum = _mm512_reduce_add_ps(acc);

    // 残り
    for(; i < n; ++i)
        sum += a[i] * b[i];

    return sum;
}