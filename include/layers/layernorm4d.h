#pragma once
#include <vector>
#include "tensor4d.h"

class LayerNorm4D {
public:
    LayerNorm4D(int dim);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int D_;

    Tensor4D gamma_;
    Tensor4D beta_;

    Tensor4D last_input_;
    Tensor4D last_mean_;
    Tensor4D last_var_;
};