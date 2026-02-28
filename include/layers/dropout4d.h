#pragma once
#include "tensor4d.h"
#include <random>

class Dropout4D
{
public:
    Dropout4D(float p);

    Tensor4D forward(const Tensor4D& x, bool train_mode);
    Tensor4D backward(const Tensor4D& grad);

private:
    float p_;
    Tensor4D mask_;
};