#pragma once
#include "tensor4d.h"

class KVCache {
public:
    int B;
    int H;
    int D;
    int maxT;
    int T;      // 現在のステップ数

    Tensor4D K;
    Tensor4D V;

    KVCache(int B_, int maxT_, int H_, int D_)
        : B(B_), H(H_), D(D_), maxT(maxT_), T(0),
          K(B_, maxT_, H_, D_),
          V(B_, maxT_, H_, D_) {}

    void append(const Tensor4D& k,
                const Tensor4D& v)
    {
        for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
        {
            K.at(b, T, h, d) = k.at(b, 0, h, d);
            V.at(b, T, h, d) = v.at(b, 0, h, d);
        }

        T++;
    }

    void reset()
    {
        T = 0;
        K.zero();
        V.zero();
    }
};