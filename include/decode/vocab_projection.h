#pragma once
#include "tensor4d.h"
#include <vector>

class VocabProjection {
public:
    // tied constructor
    VocabProjection(Tensor4D* tied_embedding);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad_out);

    std::vector<Tensor4D*> parameters();

private:
    Tensor4D* tied_weight_;   // embedding weight
    Tensor4D last_x_;
};