#pragma once
#include "tensor4d.h"

class GELU4D {
public:
    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad_out);

private:
    Tensor4D last_x_;
};