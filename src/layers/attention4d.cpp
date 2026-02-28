#include "layers/attention4d.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

static float init_w(int fan_in){
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

Attention4D::Attention4D(int H,int D)
: H_(H), D_(D),
  Wq_(1,1,D,D),
  Wk_(1,1,D,D),
  Wv_(1,1,D,D),
  Wo_(1,1,D,D)
{
    for(auto& w:Wq_.data) w=init_w(D);
    for(auto& w:Wk_.data) w=init_w(D);
    for(auto& w:Wv_.data) w=init_w(D);
    for(auto& w:Wo_.data) w=init_w(D);
}

Tensor4D Attention4D::forward(
    const Tensor4D& x,
    KVCache4D* cache,
    bool use_cache)
{
    last_x_ = x;

    int B = x.B;
    int T = x.T;

    Tensor4D Q(B,T,H_,D_);
    Tensor4D K(B,T,H_,D_);
    Tensor4D V(B,T,H_,D_);

    // ===== projection =====
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int d=0;d<D_;++d)
    {
        float q=0,k=0,v=0;
        for(int i=0;i<D_;++i){
            float xi=x.at(b,t,h,i);
            q+=xi*Wq_.at(0,0,i,d);
            k+=xi*Wk_.at(0,0,i,d);
            v+=xi*Wv_.at(0,0,i,d);
        }
        Q.at(b,t,h,d)=q;
        K.at(b,t,h,d)=k;
        V.at(b,t,h,d)=v;
    }

    last_Q_=Q;
    last_K_=K;
    last_V_=V;

    Tensor4D attn(B,T,H_,T);
    Tensor4D context(B,T,H_,D_);

    float scale = 1.0f/std::sqrt((float)D_);

    for(int b=0;b<B;++b)
    for(int h=0;h<H_;++h)
    for(int t=0;t<T;++t)
    {
        float max_s=-1e9f;

        for(int tk=0;tk<=t;++tk){
            float s=0;
            for(int d=0;d<D_;++d)
                s+=Q.at(b,t,h,d)*K.at(b,tk,h,d);
            s*=scale;
            if(s>max_s) max_s=s;
        }

        float denom=0.0f;
        for(int tk=0;tk<=t;++tk){
            float s=0;
            for(int d=0;d<D_;++d)
                s+=Q.at(b,t,h,d)*K.at(b,tk,h,d);
            s*=scale;
            float e=std::exp(s-max_s);
            attn.at(b,t,h,tk)=e;
            denom+=e;
        }

        for(int tk=0;tk<=t;++tk)
            attn.at(b,t,h,tk) /= (denom+1e-9f);

        for(int d=0;d<D_;++d){
            float sum=0;
            for(int tk=0;tk<=t;++tk)
                sum+=attn.at(b,t,h,tk)*V.at(b,tk,h,d);
            context.at(b,t,h,d)=sum;
        }
    }

    last_attn_=attn;
    last_context_=context;

    Tensor4D out(B,T,H_,D_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int d=0;d<D_;++d){
        float s=0;
        for(int i=0;i<D_;++i)
            s+=context.at(b,t,h,i)*Wo_.at(0,0,i,d);
        out.at(b,t,h,d)=s;
    }

    return out;
}

Tensor4D Attention4D::backward(const Tensor4D& dOut)
{
    int B = last_x_.B;
    int T = last_x_.T;

    Tensor4D dX(B,T,H_,D_);
    Tensor4D dQ(B,T,H_,D_);
    Tensor4D dK(B,T,H_,D_);
    Tensor4D dV(B,T,H_,D_);
    Tensor4D dContext(B,T,H_,D_);

    // ===== Wo backward =====
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int i=0;i<D_;++i)
    {
        float sum=0;
        for(int d=0;d<D_;++d){
            sum+=dOut.at(b,t,h,d)*Wo_.at(0,0,i,d);
            Wo_.grad_at(0,0,i,d)+=
                last_context_.at(b,t,h,i)*dOut.at(b,t,h,d);
        }
        dContext.at(b,t,h,i)=sum;
    }

    float scale = 1.0f/std::sqrt((float)D_);

    for(int b=0;b<B;++b)
    for(int h=0;h<H_;++h)
    for(int t=0;t<T;++t)
    {
        std::vector<float> dA(t+1,0.0f);

        for(int tk=0;tk<=t;++tk){
            float s=0;
            for(int d=0;d<D_;++d)
                s+=dContext.at(b,t,h,d)*last_V_.at(b,tk,h,d);
            dA[tk]=s;
        }

        for(int tk=0;tk<=t;++tk){
            float a=last_attn_.at(b,t,h,tk);
            for(int d=0;d<D_;++d)
                dV.at(b,tk,h,d)+=
                    a*dContext.at(b,t,h,d);
        }

        float dot=0;
        for(int tk=0;tk<=t;++tk)
            dot+=dA[tk]*last_attn_.at(b,t,h,tk);

        for(int tk=0;tk<=t;++tk){
            float a=last_attn_.at(b,t,h,tk);
            float dS=(dA[tk]-dot)*a;

            for(int d=0;d<D_;++d){
                dQ.at(b,t,h,d)+=
                    dS*last_K_.at(b,tk,h,d)*scale;
                dK.at(b,tk,h,d)+=
                    dS*last_Q_.at(b,t,h,d)*scale;
            }
        }
    }

    // ===== project backward =====
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int i=0;i<D_;++i)
    {
        float gx=0;
        for(int d=0;d<D_;++d){
            float xi=last_x_.at(b,t,h,i);

            Wq_.grad_at(0,0,i,d)+=
                xi*dQ.at(b,t,h,d);
            Wk_.grad_at(0,0,i,d)+=
                xi*dK.at(b,t,h,d);
            Wv_.grad_at(0,0,i,d)+=
                xi*dV.at(b,t,h,d);

            gx+=dQ.at(b,t,h,d)*Wq_.at(0,0,i,d);
            gx+=dK.at(b,t,h,d)*Wk_.at(0,0,i,d);
            gx+=dV.at(b,t,h,d)*Wv_.at(0,0,i,d);
        }
        dX.at(b,t,h,i)=gx;
    }

    return dX;
}

std::vector<Tensor4D*> Attention4D::parameters()
{
    return {&Wq_,&Wk_,&Wv_,&Wo_};
}