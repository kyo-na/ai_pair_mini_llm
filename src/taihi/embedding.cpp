#include "layers/embedding.h"

Embedding::Embedding(int v, int d)
    : vocab(v), dim(d), W(v, d) {}

Tensor Embedding::forward(const std::vector<int>& ids) {
    last_ids = ids;
    Tensor out(ids.size(), dim);

    for (int i = 0; i < ids.size(); i++) {
        int id = ids[i];
        for (int j = 0; j < dim; j++)
            out(i,j) = W(id,j);
    }
    return out;
}

void Embedding::backward(const Tensor& dy) {
    for (int i = 0; i < last_ids.size(); i++) {
        int id = last_ids[i];
        for (int j = 0; j < dim; j++)
            W.grad[id * dim + j] += dy(i,j);
    }
}