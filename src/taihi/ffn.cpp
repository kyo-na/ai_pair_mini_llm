#include "layers/ffn.h"

FFN::FFN(int dim)
    : l1(dim, dim * 4), l2(dim * 4, dim) {}

Tensor FFN::forward(const Tensor& x) {
    return l2.forward(l1.forward(x));
}

Tensor FFN::backward(const Tensor& dout) {
    return l1.backward(l2.backward(dout));
}