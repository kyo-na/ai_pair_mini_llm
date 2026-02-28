#include "layers/attention.h"

Attention::Attention(int dim)
    : q(dim, dim), k(dim, dim), v(dim, dim) {}

Tensor Attention::forward(const Tensor& x) {
    return v.forward(x); // 最小実装（Self-Attention簡略）
}

Tensor Attention::backward(const Tensor& dout) {
    return v.backward(dout);
}