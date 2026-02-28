#include "../../include/model/embedding.h"

Embedding::Embedding(int v, int d) : vocab(v), dim(d), W(v*d) {}

std::vector<float> Embedding::forward(uint32_t t) {
    std::vector<float> out(dim);
    if (t >= (uint32_t)vocab) t = 0;
    for (int i = 0; i < dim; ++i)
        out[i] = W[t*dim + i];
    return out;
}