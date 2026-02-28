#include "layers/attention4d.h"
#include <cmath>
#include <algorithm>

Tensor4D Attention4D::forward_flash_cpu(
    const Tensor4D& x,
    KVCache4D* cache)
{
    int B=1;
    int H=H_;
    int Dh=D_;

    Tensor4D Q(B,1,H,Dh);
    Tensor4D K_new(B,1,H,Dh);
    Tensor4D V_new(B,1,H,Dh);

    // projection
    for(int h=0;h<H;++h)
    for(int d=0;d<Dh;++d)
    {
        float q=0,k=0,v=0;
        for(int i=0;i<Dh;++i)
        {
            float xi=x.at(0,0,h,i);
            q+=xi*Wq_.at(0,0,i,d);
            k+=xi*Wk_.at(0,0,i,d);
            v+=xi*Wv_.at(0,0,i,d);
        }
        Q.at(0,0,h,d)=q;
        K_new.at(0,0,h,d)=k;
        V_new.at(0,0,h,d)=v;
    }

    cache->append(K_new,V_new);

    Tensor4D& K_all=cache->K();
    Tensor4D& V_all=cache->V();

    int L=cache->length();

    Tensor4D context(B,1,H,Dh);

    for(int h=0;h<H;++h)
    {
        float max_score=-1e9f;

        for(int t=0;t<L;++t)
        {
            float s=0;
            for(int d=0;d<Dh;++d)
                s+=Q.at(0,0,h,d)
                   *K_all.at(0,t,h,d);

            s/=std::sqrt((float)Dh);

            if(s>max_score) max_score=s;
        }

        float denom=0;

        for(int t=0;t<L;++t)
        {
            float s=0;
            for(int d=0;d<Dh;++d)
                s+=Q.at(0,0,h,d)
                   *K_all.at(0,t,h,d);

            s/=std::sqrt((float)Dh);
            denom+=std::exp(s-max_score);
        }

        for(int d=0;d<Dh;++d)
        {
            float sum=0;
            for(int t=0;t<L;++t)
            {
                float s=0;
                for(int i=0;i<Dh;++i)
                    s+=Q.at(0,0,h,i)
                       *K_all.at(0,t,h,i);

                s/=std::sqrt((float)Dh);

                float attn=
                    std::exp(s-max_score)/denom;

                sum+=attn
                     *V_all.at(0,t,h,d);
            }

            context.at(0,0,h,d)=sum;
        }
    }

    return context;
}