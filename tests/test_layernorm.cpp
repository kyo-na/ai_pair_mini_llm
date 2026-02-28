#include <iostream>
#include <cmath>
#include "layers/layernorm.h"

float loss_sum(const Tensor& y) {
    float s = 0.f;
    for (float v : y.data) s += v;
    return s;
}

int main() {
    LayerNorm ln(4);

    Tensor x(3, 4);
    init_uniform(x);

    Tensor y = ln.forward(x);

    Tensor dy(3, 4);
    for (auto& v : dy.data) v = 1.f;

    ln.backward(dy);

    // numeric grad check (x[0,0])
    float eps = 1e-3f;
    float orig = x(0,0);

    x(0,0) = orig + eps;
    float l1 = loss_sum(ln.forward(x));

    x(0,0) = orig - eps;
    float l2 = loss_sum(ln.forward(x));

    x(0,0) = orig;

    float num_grad = (l1 - l2) / (2 * eps);
    float back_grad = ln.backward(dy)(0,0);

    std::cout << "numeric grad = " << num_grad << "\n";
    std::cout << "backward grad = " << back_grad << "\n";
    std::cout << "diff = " << std::abs(num_grad - back_grad) << "\n";
}