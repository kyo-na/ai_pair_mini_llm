#pragma once
#include "tensor.h"

struct Embedding {
    int vocab, dim;
    Tensor W;
    std::vector<int> last_ids;

    Embedding(int v, int d);

    Tensor forward(const std::vector<int>& ids);
    void backward(const Tensor& dy);
};