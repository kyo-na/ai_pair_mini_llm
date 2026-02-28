#pragma once
#include "layers/linear.h"

struct Attention {
    Linear q, k, v;

    Attention(int dim);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};