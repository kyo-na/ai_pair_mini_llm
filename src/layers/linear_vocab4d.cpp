#include "layers/linear_vocab4d.h"
#include <cstdlib>
#include <cmath>

static float init_w(int fan_in)
{
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

LinearVocab4D::LinearVocab4D(
    int heads,
    int head_dim,
    int vocab_size)
    : H_(heads), D_(head_dim),
      vocab_(vocab_size),
      weight_(1,1,heads*head_dim,vocab_size)
{
    for(auto& w : weight_.data)
        w = init_w(heads*head_dim);
}

Tensor4D LinearVocab4D::forward(const Tensor4D& x)
{
    last_input_ = x;

    int B=x.B;
    int T=x.T;

    Tensor4D out(B,T,1,vocab_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int v=0;v<vocab_;++v)
    {
        float sum=0;

        for(int h=0;h<H_;++h)
        for(int d=0;d<D_;++d)
        {
            int flat=h*D_+d;
            sum+=x.at(b,t,h,d) *
                 weight_.at(0,0,flat,v);
        }

        out.at(b,t,0,v)=sum;
    }

    return out;
}

Tensor4D LinearVocab4D::backward(const Tensor4D& grad)
{
    int B=last_input_.B;
    int T=last_input_.T;

    Tensor4D dX(B,T,H_,D_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int v=0;v<vocab_;++v)
    {
        float g = grad.at(b,t,0,v);

        for(int h=0;h<H_;++h)
        for(int d=0;d<D_;++d)
        {
            int flat=h*D_+d;

            weight_.grad_at(0,0,flat,v)
                += last_input_.at(b,t,h,d)*g;

            dX.at(b,t,h,d)
                += weight_.at(0,0,flat,v)*g;
        }
    }

    return dX;
}

std::vector<Tensor4D*> LinearVocab4D::parameters()
{
    return { &weight_ };
}