#pragma once
#include <vector>
#include <cstdint>
#include "embedding.h"
#include "linear.h"

struct MiniAI {
    Embedding emb;
    Linear proj;

    MiniAI(int vocab, int dim);
    uint32_t forward_token(uint32_t t);
};