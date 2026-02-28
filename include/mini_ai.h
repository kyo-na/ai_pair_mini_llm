#pragma once
#include <vector>
#include <cstdint>
#include "tensor.h"

struct MiniAI {
    int vocab, dim;
    Tensor emb;    // vocab x dim
    Tensor proj;   // dim x vocab

    MiniAI(int v, int d);

    uint32_t forward_token(uint32_t token);
};