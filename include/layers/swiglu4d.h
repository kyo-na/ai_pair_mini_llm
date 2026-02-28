#pragma once
#include "tensor4d.h"
#include <vector>

class SwiGLU4D {
public:
    SwiGLU4D(int heads, int head_dim, int hidden);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int H_;
    int D_;
    int hidden_;

    Tensor4D W1_;  // (1,1,H*D,hidden)
    Tensor4D W2_;  // (1,1,H*D,hidden)
    Tensor4D W3_;  // (1,1,hidden,H*D)

    Tensor4D last_x_;
    Tensor4D last_u_;
    Tensor4D last_v_;
    Tensor4D last_gelu_;
};