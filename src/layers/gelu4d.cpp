#include "layers/gelu4d.h"
#include <cmath>

static constexpr float PI_F = 3.14159265358979323846f;

static float gelu(float x)
{
    const float k = std::sqrt(2.0f / PI_F);
    return 0.5f * x * (1.0f + std::tanh(
        k * (x + 0.044715f * x * x * x)
    ));
}

static float gelu_grad(float x)
{
    const float k = std::sqrt(2.0f / PI_F);

    float inner = k * (x + 0.044715f * x * x * x);
    float tanh_term = std::tanh(inner);
    float sech2 = 1.0f - tanh_term * tanh_term;

    float term1 = 0.5f * (1.0f + tanh_term);
    float term2 = 0.5f * x * sech2 * k * (1.0f + 3.0f * 0.044715f * x * x);

    return term1 + term2;
}

Tensor4D GELU4D::forward(const Tensor4D& x)
{
    last_x_ = x;
    Tensor4D y = x;

    for (auto& v : y.data)
        v = gelu(v);

    return y;
}

Tensor4D GELU4D::backward(const Tensor4D& grad_out)
{
    Tensor4D dx = grad_out;

    for (size_t i = 0; i < dx.data.size(); ++i)
        dx.data[i] *= gelu_grad(last_x_.data[i]);

    return dx;
}