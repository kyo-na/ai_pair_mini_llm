#include "loss/cross_entropy4d.h"
#include <cmath>
#include <algorithm>

float CrossEntropy4D::forward(
    const Tensor4D& logits,
    const Tensor4D& target)
{
    last_logits_ = logits;
    last_target_ = target;

    int B = logits.B;
    int T = logits.T;
    int H = logits.H;
    int D = logits.D;

    float loss = 0.0f;

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H;++h)
    {
        float maxv = -1e9f;
        for(int d=0; d<D; ++d)
            maxv = std::max(maxv, logits.at(b,t,h,d));

        float denom = 0.0f;
        for(int d=0; d<D; ++d)
            denom += std::exp(logits.at(b,t,h,d) - maxv);

        for(int d=0; d<D; ++d)
        {
            float p = std::exp(logits.at(b,t,h,d) - maxv)
                      / (denom + 1e-9f);

            float y = target.at(b,t,h,d);

            if(y > 0.0f)
                loss -= y * std::log(p + 1e-9f);
        }
    }

    return loss / (B*T*H);
}

Tensor4D CrossEntropy4D::backward()
{
    int B = last_logits_.B;
    int T = last_logits_.T;
    int H = last_logits_.H;
    int D = last_logits_.D;

    Tensor4D grad(B,T,H,D);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H;++h)
    {
        float maxv = -1e9f;
        for(int d=0; d<D; ++d)
            maxv = std::max(maxv,
                           last_logits_.at(b,t,h,d));

        float denom = 0.0f;
        for(int d=0; d<D; ++d)
            denom += std::exp(
                last_logits_.at(b,t,h,d) - maxv);

        for(int d=0; d<D; ++d)
        {
            float p = std::exp(
                last_logits_.at(b,t,h,d) - maxv)
                / (denom + 1e-9f);

            float y = last_target_.at(b,t,h,d);

            grad.at(b,t,h,d) =
                (p - y) / (B*T*H);
        }
    }

    return grad;
}