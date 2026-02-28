#pragma once
#include "attention4d.h"

struct Transformer4D {
    Attention4D attn;
    Transformer4D(int h,int d):attn(h,d){}
    Tensor4D forward(const Tensor4D& x, World4D& world);
};