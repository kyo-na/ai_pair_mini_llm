#pragma once
#include "tensor.h"
#include <vector>

struct Embedding {
    Tensor2D W; // [vocab, hidden]

    Embedding(int vocab,int hidden):W(vocab,hidden){}
    std::vector<float> forward(int token);
};