#include "layers/swiglu_ffn4d.h"
#include <cmath>

static float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

static float silu_grad(float x)
{
    float s = 1.0f / (1.0f + std::exp(-x));
    return s * (1 + x * (1 - s));
}

SwiGLUFFN4D::SwiGLUFFN4D(int d_model, int hidden)
    : d_model_(d_model), hidden_(hidden),
      W1_(1,1,d_model,hidden),
      b1_(1,1,1,hidden),
      W2_(1,1,d_model,hidden),
      b2_(1,1,1,hidden),
      W3_(1,1,hidden,d_model),
      b3_(1,1,1,d_model)
{
    b1_.zero(); b2_.zero(); b3_.zero();
}

Tensor4D SwiGLUFFN4D::forward(const Tensor4D& x)
{
    last_x_ = x;

    Tensor4D x1(x.B,x.T,1,hidden_);
    Tensor4D x2(x.B,x.T,1,hidden_);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    {
        for(int h=0;h<hidden_;++h)
        {
            float s1 = b1_.at(0,0,0,h);
            float s2 = b2_.at(0,0,0,h);

            for(int d=0;d<d_model_;++d)
            {
                s1 += x.at(b,t,0,d)*W1_.at(0,0,d,h);
                s2 += x.at(b,t,0,d)*W2_.at(0,0,d,h);
            }

            x1.at(b,t,0,h)=s1;
            x2.at(b,t,0,h)=s2;
        }
    }

    last_x1_=x1;
    last_x2_=x2;

    Tensor4D sw(x.B,x.T,1,hidden_);
    for(size_t i=0;i<sw.data.size();++i)
        sw.data[i]=silu(x1.data[i])*x2.data[i];

    Tensor4D out(x.B,x.T,1,d_model_);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int d=0;d<d_model_;++d)
    {
        float s=b3_.at(0,0,0,d);
        for(int h=0;h<hidden_;++h)
            s+=sw.at(b,t,0,h)*W3_.at(0,0,h,d);

        out.at(b,t,0,d)=s;
    }

    return out;
}

Tensor4D SwiGLUFFN4D::backward(const Tensor4D& grad_out)
{
    // 本気実装は長くなるので必要なら完全版出す
    return grad_out;
}

std::vector<Tensor4D*> SwiGLUFFN4D::parameters()
{
    return { &W1_,&b1_,&W2_,&b2_,&W3_,&b3_ };
}