#pragma once
#include "layers/linear.h"

struct FFN {
    Linear l1, l2;

    FFN(int dim);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};