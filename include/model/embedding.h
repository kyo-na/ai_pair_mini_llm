#pragma once
#include <vector>
#include <cstdint>

struct Embedding {
    int vocab, dim;
    std::vector<float> W;

    Embedding(int v, int d);
    std::vector<float> forward(uint32_t token);
};