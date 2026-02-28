#include <iostream>
#include <cmath>
#include "layers/attention.h"

float loss_sum(const Tensor& y) {
    float s = 0.f;
    for (float v : y.data) s += v;
    return s;
}

int main() {
    Attention attn(4);

    Tensor x(3,4);
    init_uniform(x);

    Tensor y = attn.forward(x);

    Tensor dy(3,4);
    for (auto& v : dy.data) v = 1.f;

    attn.backward(dy);

    // numeric grad (x[0,0])
    float eps = 1e-3f;
    float orig = x(0,0);

    x(0,0) = orig + eps;
    float l1 = loss_sum(attn.forward(x));

    x(0,0) = orig - eps;
    float l2 = loss_sum(attn.forward(x));

    x(0,0) = orig;

    float num_grad = (l1 - l2)/(2*eps);
    float back_grad = attn.backward(dy)(0,0);

    std::cout << "numeric=" << num_grad << "\n";
    std::cout << "backward=" << back_grad << "\n";
    std::cout << "diff=" << std::abs(num_grad-back_grad) << "\n";
}