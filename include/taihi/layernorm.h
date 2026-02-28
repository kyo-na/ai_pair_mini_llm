#pragma once
#include "tensor.h"

struct LayerNorm {
    Tensor gamma, beta, x_cache;
    float mean, var;
    int dim;

    LayerNorm(int dim);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};