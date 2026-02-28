#pragma once
#include "tensor4d.h"
#include <vector>

class LinearVocab4D {
public:
    LinearVocab4D(int heads, int head_dim, int vocab_size);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int H_;
    int D_;
    int vocab_;

    Tensor4D weight_;   // (1,1,H*D,vocab)

    Tensor4D last_input_;
};