// softmax4d.h
#pragma once
#include "tensor4d.h"
#include <cmath>

inline Tensor4D softmax4d(const Tensor4D& x) {
    Tensor4D y(x.B,x.T,x.H,x.D);

    for(int b=0;b<x.B;b++)
    for(int t=0;t<x.T;t++)
    for(int h=0;h<x.H;h++) {
        float maxv = -1e9f;
        for(int d=0;d<x.D;d++)
            maxv = std::max(maxv, x.at(b,t,h,d));

        float sum = 0.0f;
        for(int d=0;d<x.D;d++) {
            float e = std::exp(x.at(b,t,h,d) - maxv);
            y.at(b,t,h,d) = e;
            sum += e;
        }
        for(int d=0;d<x.D;d++)
            y.at(b,t,h,d) /= sum;
    }
    return y;
}