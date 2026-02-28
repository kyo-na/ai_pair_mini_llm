#include "loss/cross_entropy.h"
#include <cmath>

float CrossEntropy::forward(const Tensor4D& logits,
                            const std::vector<int>& target)
{
    target_ = target;

    int B = logits.B;
    int T = logits.T;
    int D = logits.D;

    probs_ = Tensor4D(B, T, 1, D);

    float loss = 0.0f;

    for(int t = 0; t < T; ++t)
    {
        float maxv = -1e9f;
        for(int d = 0; d < D; ++d)
            maxv = std::max(maxv, logits.at(0,t,0,d));

        float sum = 0.0f;
        for(int d = 0; d < D; ++d)
        {
            float e = std::exp(logits.at(0,t,0,d) - maxv);
            probs_.at(0,t,0,d) = e;
            sum += e;
        }

        for(int d = 0; d < D; ++d)
            probs_.at(0,t,0,d) /= sum;

        loss -= std::log(probs_.at(0,t,0,target[t]) + 1e-9f);
    }

    return loss / T;
}

Tensor4D CrossEntropy::backward()
{
    Tensor4D grad = probs_;

    int T = grad.T;
    int D = grad.D;

    for(int t=0; t<T; ++t)
        grad.at(0,t,0,target_[t]) -= 1.0f;

    return grad;
}