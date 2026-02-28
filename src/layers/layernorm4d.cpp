#include "layers/layernorm4d.h"
#include <cmath>

LayerNorm4D::LayerNorm4D(int dim)
    : D_(dim),
      gamma_(1,1,1,dim),
      beta_(1,1,1,dim)
{
    for(int d=0; d<dim; ++d)
    {
        gamma_.at(0,0,0,d)=1.0f;
        beta_.at(0,0,0,d)=0.0f;
    }
}

Tensor4D LayerNorm4D::forward(const Tensor4D& x)
{
    last_input_ = x;

    Tensor4D out(x.B,x.T,x.H,x.D);
    last_mean_ = Tensor4D(x.B,x.T,x.H,1);
    last_var_  = Tensor4D(x.B,x.T,x.H,1);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int h=0;h<x.H;++h)
    {
        float mean=0;
        for(int d=0;d<x.D;++d)
            mean+=x.at(b,t,h,d);
        mean/=x.D;

        float var=0;
        for(int d=0;d<x.D;++d)
        {
            float diff=x.at(b,t,h,d)-mean;
            var+=diff*diff;
        }
        var/=x.D;

        last_mean_.at(b,t,h,0)=mean;
        last_var_.at(b,t,h,0)=var;

        float inv=1.0f/std::sqrt(var+1e-5f);

        for(int d=0;d<x.D;++d)
        {
            float norm=(x.at(b,t,h,d)-mean)*inv;
            out.at(b,t,h,d)=norm*gamma_.at(0,0,0,d)
                             +beta_.at(0,0,0,d);
        }
    }

    return out;
}

Tensor4D LayerNorm4D::backward(const Tensor4D& grad)
{
    int B = last_input_.B;
    int T = last_input_.T;
    int H = last_input_.H;
    int D = last_input_.D;

    Tensor4D dX(B, T, H, D);

    const float eps = 1e-5f;

    for(int b=0; b<B; ++b)
    for(int t=0; t<T; ++t)
    for(int h=0; h<H; ++h)
    {
        float mean = last_mean_.at(b,t,h,0);
        float var  = last_var_.at(b,t,h,0);
        float inv_std = 1.0f / std::sqrt(var + eps);

        // ----- dBeta / dGamma -----
        for(int d=0; d<D; ++d)
        {
            float x = last_input_.at(b,t,h,d);
            float norm = (x - mean) * inv_std;

            beta_.grad_at(0,0,0,d)  += grad.at(b,t,h,d);
            gamma_.grad_at(0,0,0,d) += grad.at(b,t,h,d) * norm;
        }

        // ----- dX -----

        // ① dx_hat = grad * gamma
        std::vector<float> dx_hat(D);
        for(int d=0; d<D; ++d)
        {
            dx_hat[d] = grad.at(b,t,h,d) *
                        gamma_.at(0,0,0,d);
        }

        // ② sum(dx_hat)
        float sum_dxhat = 0.0f;
        float sum_dxhat_xmu = 0.0f;

        for(int d=0; d<D; ++d)
        {
            float xmu = last_input_.at(b,t,h,d) - mean;
            sum_dxhat += dx_hat[d];
            sum_dxhat_xmu += dx_hat[d] * xmu;
        }

        // ③ dX
        for(int d=0; d<D; ++d)
        {
            float xmu = last_input_.at(b,t,h,d) - mean;

            float term1 = dx_hat[d] * inv_std;
            float term2 = sum_dxhat / D * inv_std;
            float term3 = xmu * inv_std * inv_std *
                          sum_dxhat_xmu / D;

            dX.at(b,t,h,d) = term1
                           - term2
                           - term3;
        }
    }

    return dX;
}

std::vector<Tensor4D*> LayerNorm4D::parameters()
{
    return {&gamma_, &beta_};
}