#pragma once
#include "tensor4d.h"

class Softmax4D {
public:
    Tensor4D forward(const Tensor4D& x);        // (B,T,1,V)
    Tensor4D backward(const Tensor4D& grad);    // (B,T,1,V)

private:
    Tensor4D last_y_;
};