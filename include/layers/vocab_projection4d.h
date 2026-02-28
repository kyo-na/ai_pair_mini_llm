#pragma once
#include "tensor4d.h"
#include <vector>

class VocabProjection4D {
public:
    VocabProjection4D(int d_model, int vocab_size);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad_out);

    std::vector<Tensor4D*> parameters();

private:
    int d_model_;
    int vocab_;

    Tensor4D W_;  // (1,1,d_model,vocab)
    Tensor4D b_;  // (1,1,1,vocab)

    Tensor4D last_x_;
};