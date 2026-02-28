#include "layers/linear.h"
#include <random>

static float randf() {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> d(-0.1f, 0.1f);
    return d(rng);
}

Linear::Linear(int in, int out)
    : W(in * out), b(out) {
    for (auto& v : W.data) v = randf();
    for (auto& v : b.data) v = 0.0f;
}

Tensor Linear::forward(const Tensor& x) {
    x_cache = x;
    int out_dim = b.n;
    int in_dim = x.n;
    Tensor y(out_dim);

    for (int o = 0; o < out_dim; o++) {
        y.data[o] = b.data[o];
        for (int i = 0; i < in_dim; i++)
            y.data[o] += x.data[i] * W.data[i * out_dim + o];
    }
    return y;
}

Tensor Linear::backward(const Tensor& dout) {
    int out_dim = b.n;
    int in_dim = x_cache.n;
    Tensor dx(in_dim);

    for (int i = 0; i < in_dim; i++)
        for (int o = 0; o < out_dim; o++)
            W.grad[i * out_dim + o] += x_cache.data[i] * dout.grad[o];

    for (int o = 0; o < out_dim; o++)
        b.grad[o] += dout.grad[o];

    for (int i = 0; i < in_dim; i++)
        for (int o = 0; o < out_dim; o++)
            dx.grad[i] += W.data[i * out_dim + o] * dout.grad[o];

    return dx;
}