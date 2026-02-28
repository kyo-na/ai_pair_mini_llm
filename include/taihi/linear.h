#pragma once
#include "tensor.h"

struct Linear {
    Tensor W, b;
    Tensor x_cache;

    Linear(int in, int out);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};