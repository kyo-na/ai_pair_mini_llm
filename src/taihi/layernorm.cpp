#include "layers/layernorm.h"
#include <cmath>

LayerNorm::LayerNorm(int d)
    : gamma(d), beta(d), dim(d) {
    for (int i = 0; i < d; i++) {
        gamma.data[i] = 1.0f;
        beta.data[i] = 0.0f;
    }
}

Tensor LayerNorm::forward(const Tensor& x) {
    x_cache = x;
    mean = 0.0f;
    for (float v : x.data) mean += v;
    mean /= dim;

    var = 0.0f;
    for (float v : x.data) var += (v - mean) * (v - mean);
    var /= dim;

    Tensor y(dim);
    for (int i = 0; i < dim; i++)
        y.data[i] = gamma.data[i] * (x.data[i] - mean) / std::sqrt(var + 1e-5f) + beta.data[i];
    return y;
}

Tensor LayerNorm::backward(const Tensor& dout) {
    Tensor dx(dim);
    for (int i = 0; i < dim; i++) {
        gamma.grad[i] += dout.grad[i] * (x_cache.data[i] - mean);
        beta.grad[i] += dout.grad[i];
        dx.grad[i] += gamma.data[i] * dout.grad[i];
    }
    return dx;
}