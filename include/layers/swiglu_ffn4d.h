#pragma once
#include "tensor4d.h"
#include <vector>

class SwiGLUFFN4D {
public:
    SwiGLUFFN4D(int d_model, int hidden_dim);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad_out);

    std::vector<Tensor4D*> parameters();

private:
    int d_model_;
    int hidden_;

    Tensor4D W1_, b1_;
    Tensor4D W2_, b2_;
    Tensor4D W3_, b3_;

    Tensor4D last_x_;
    Tensor4D last_x1_;
    Tensor4D last_x2_;
};