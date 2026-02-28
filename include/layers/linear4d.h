#pragma once
#include "tensor4d.h"
#include <vector>

class Linear4D {
public:
    Linear4D(int in_dim, int out_dim);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int in_;
    int out_;

    Tensor4D W_;
    Tensor4D b_;

    Tensor4D last_x_;
};