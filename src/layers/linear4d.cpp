#include "layers/linear4d.h"
#include <cmath>
#include <cstdlib>

static float init_w(int fan_in){
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

Linear4D::Linear4D(int in_dim,int out_dim)
: in_(in_dim), out_(out_dim),
  W_(1,1,in_dim,out_dim),
  b_(1,1,1,out_dim)
{
    for(auto& w:W_.data) w=init_w(in_dim);
    for(auto& v:b_.data) v=0.0f;
}

Tensor4D Linear4D::forward(const Tensor4D& x)
{
    last_x_=x;

    int B=x.B;
    int T=x.T;
    int H=x.H;

    Tensor4D out(B,T,H,out_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H;++h)
    for(int o=0;o<out_;++o){
        float s=b_.at(0,0,0,o);
        for(int i=0;i<in_;++i)
            s+=x.at(b,t,h,i)*W_.at(0,0,i,o);
        out.at(b,t,h,o)=s;
    }

    return out;
}

Tensor4D Linear4D::backward(const Tensor4D& grad)
{
    int B=grad.B;
    int T=grad.T;
    int H=grad.H;

    W_.zero_grad();
    b_.zero_grad();

    Tensor4D dx(B,T,H,in_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H;++h)
    for(int o=0;o<out_;++o){
        float go=grad.at(b,t,h,o);

        b_.grad_at(0,0,0,o)+=go;

        for(int i=0;i<in_;++i){
            W_.grad_at(0,0,i,o)+=
                last_x_.at(b,t,h,i)*go;

            dx.at(b,t,h,i)+=
                W_.at(0,0,i,o)*go;
        }
    }

    return dx;
}

std::vector<Tensor4D*> Linear4D::parameters()
{
    return {&W_,&b_};
}