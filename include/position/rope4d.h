#pragma once
#include "tensor4d.h"

class RoPE4D {
public:
    // 4D (B,T,H,D) 前提
    void apply(Tensor4D& x);
};