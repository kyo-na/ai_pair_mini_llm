#include "layers/rmsnorm4d.h"
#include <cmath>

RMSNorm4D::RMSNorm4D(int dim, float eps)
    : D_(dim),
      eps_(eps),
      gamma_(1,1,1,dim)
{
    for(auto& v : gamma_.data)
        v = 1.0f;
}

Tensor4D RMSNorm4D::forward(const Tensor4D& x)
{
    last_x_ = x;
    last_r_.resize(x.B * x.T * x.H);

    Tensor4D out(x.B, x.T, x.H, x.D);

    int idx = 0;

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int h=0;h<x.H;++h)
    {
        float sum=0.0f;
        for(int d=0;d<D_;++d)
        {
            float v = x.at(b,t,h,d);
            sum += v*v;
        }

        float r = 1.0f / std::sqrt(sum / D_ + eps_);
        last_r_[idx++] = r;

        for(int d=0;d<D_;++d)
        {
            out.at(b,t,h,d) =
                x.at(b,t,h,d) * r * gamma_.at(0,0,0,d);
        }
    }

    return out;
}

Tensor4D RMSNorm4D::backward(const Tensor4D& grad)
{
    Tensor4D dx(last_x_.B, last_x_.T, last_x_.H, last_x_.D);
    gamma_.grad.assign(gamma_.grad.size(), 0.0f);

    int idx=0;

    for(int b=0;b<last_x_.B;++b)
    for(int t=0;t<last_x_.T;++t)
    for(int h=0;h<last_x_.H;++h)
    {
        float r = last_r_[idx++];

        float dot=0.0f;
        for(int d=0;d<D_;++d)
        {
            float g = grad.at(b,t,h,d);
            float x = last_x_.at(b,t,h,d);
            float gamma = gamma_.at(0,0,0,d);
            dot += g * gamma * x;
        }

        for(int d=0;d<D_;++d)
        {
            float g = grad.at(b,t,h,d);
            float x = last_x_.at(b,t,h,d);
            float gamma = gamma_.at(0,0,0,d);

            dx.at(b,t,h,d) =
                r * gamma * g
                - (r*r*r / D_) * x * dot;

            gamma_.grad[d] += g * x * r;
        }
    }

    return dx;
}

std::vector<Tensor4D*> RMSNorm4D::parameters()
{
    return { &gamma_ };
}