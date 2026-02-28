#include "layers/dropout4d.h"

Dropout4D::Dropout4D(float p) : p_(p) {}

Tensor4D Dropout4D::forward(const Tensor4D& x, bool train_mode)
{
    if (!train_mode || p_ <= 0.0f)
        return x;

    mask_ = Tensor4D(x.B, x.H, x.T, x.D);

    std::mt19937 gen(42);
    std::bernoulli_distribution dist(1.0 - p_);

    Tensor4D out = x;

    for (size_t i = 0; i < x.data.size(); ++i)
    {
        float m = dist(gen) ? 1.0f : 0.0f;
        mask_.data[i] = m;
        out.data[i] *= m / (1.0f - p_);
    }

    return out;
}

Tensor4D Dropout4D::backward(const Tensor4D& grad)
{
    Tensor4D out = grad;

    for (size_t i = 0; i < grad.data.size(); ++i)
    {
        out.data[i] *= mask_.data[i];
    }

    return out;
}