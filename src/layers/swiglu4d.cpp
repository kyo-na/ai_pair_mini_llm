#include "layers/swiglu4d.h"
#include <cmath>
#include <cstdlib>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float init_w(int fan_in)
{
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

static float gelu(float x)
{
    return 0.5f*x*(1.0f+std::tanh(
        std::sqrt(2.0f/M_PI)*(x+0.044715f*x*x*x)));
}

static float gelu_deriv(float x)
{
    float tanh_out = std::tanh(
        std::sqrt(2.0f/M_PI)*(x+0.044715f*x*x*x));

    float sech2 = 1 - tanh_out*tanh_out;

    float term = std::sqrt(2.0f/M_PI)*
                 (1+3*0.044715f*x*x);

    return 0.5f*(1+tanh_out) +
           0.5f*x*sech2*term;
}

SwiGLU4D::SwiGLU4D(int heads,int head_dim,int hidden)
: H_(heads), D_(head_dim), hidden_(hidden),
  W1_(1,1,heads*head_dim,hidden),
  W2_(1,1,heads*head_dim,hidden),
  W3_(1,1,hidden,heads*head_dim)
{
    for(auto& w:W1_.data) w=init_w(heads*head_dim);
    for(auto& w:W2_.data) w=init_w(heads*head_dim);
    for(auto& w:W3_.data) w=init_w(hidden);
}

Tensor4D SwiGLU4D::forward(const Tensor4D& x)
{
    last_x_ = x;

    int B=x.B;
    int T=x.T;

    Tensor4D u(B,T,1,hidden_);
    Tensor4D v(B,T,1,hidden_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<hidden_;++h)
    {
        float sum1=0,sum2=0;

        for(int hh=0;hh<H_;++hh)
        for(int d=0;d<D_;++d)
        {
            int flat=hh*D_+d;
            sum1+=x.at(b,t,hh,d)*W1_.at(0,0,flat,h);
            sum2+=x.at(b,t,hh,d)*W2_.at(0,0,flat,h);
        }

        u.at(b,t,0,h)=sum1;
        v.at(b,t,0,h)=sum2;
    }

    last_u_=u;
    last_v_=v;

    Tensor4D out(B,T,H_,D_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int hh=0;hh<H_;++hh)
    for(int d=0;d<D_;++d)
    {
        int flat=hh*D_+d;
        float sum=0;

        for(int h=0;h<hidden_;++h)
        {
            float g = gelu(u.at(b,t,0,h));
            sum+=g * v.at(b,t,0,h)
                 * W3_.at(0,0,h,flat);
        }

        out.at(b,t,hh,d)=sum;
    }

    return out;
}

Tensor4D SwiGLU4D::backward(const Tensor4D& grad)
{
    int B = last_x_.B;
    int T = last_x_.T;

    Tensor4D dX(B,T,H_,D_);

    // ===== dW3 & dh =====
    Tensor4D dh(B,T,1,hidden_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<hidden_;++h)
    {
        float sum = 0.0f;

        for(int hh=0;hh<H_;++hh)
        for(int d=0;d<D_;++d)
        {
            int flat = hh*D_+d;

            float g_out = grad.at(b,t,hh,d);

            // dW3
            W3_.grad_at(0,0,h,flat)
                += last_v_.at(b,t,0,h)
                   * gelu(last_u_.at(b,t,0,h))
                   * g_out;

            sum += g_out * W3_.at(0,0,h,flat);
        }

        dh.at(b,t,0,h) = sum;
    }

    // ===== dg, dv =====
    Tensor4D du(B,T,1,hidden_);
    Tensor4D dv(B,T,1,hidden_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<hidden_;++h)
    {
        float g = gelu(last_u_.at(b,t,0,h));

        dv.at(b,t,0,h) = dh.at(b,t,0,h) * g;

        float g_deriv = gelu_deriv(last_u_.at(b,t,0,h));
        du.at(b,t,0,h) = dh.at(b,t,0,h)
                         * last_v_.at(b,t,0,h)
                         * g_deriv;
    }

    // ===== dW1, dW2 & dX =====
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int hh=0;hh<H_;++hh)
    for(int d=0;d<D_;++d)
    {
        int flat = hh*D_+d;
        float gx = 0.0f;

        for(int h=0;h<hidden_;++h)
        {
            float x_val = last_x_.at(b,t,hh,d);

            // dW1
            W1_.grad_at(0,0,flat,h)
                += x_val * du.at(b,t,0,h);

            // dW2
            W2_.grad_at(0,0,flat,h)
                += x_val * dv.at(b,t,0,h);

            // dX
            gx += du.at(b,t,0,h)
                  * W1_.at(0,0,flat,h);

            gx += dv.at(b,t,0,h)
                  * W2_.at(0,0,flat,h);
        }

        dX.at(b,t,hh,d) = gx;
    }

    return dX;
}

std::vector<Tensor4D*> SwiGLU4D::parameters()
{
    return {&W1_,&W2_,&W3_};
}