#include "layers/moe_ffn4d.h"
#include <algorithm>
#include <cmath>

MoEFFN4D::MoEFFN4D(int d_model, int hidden, int experts)
: experts_(experts),
  router_(d_model, experts)
{
    expert_.reserve((size_t)experts);
    for(int i=0;i<experts;++i){
        expert_.emplace_back(d_model, hidden);
    }
}

Tensor4D MoEFFN4D::forward(const Tensor4D& x)
{
    // gate logits: (B,T,H,experts)
    Tensor4D gate = router_.forward(x);

    Tensor4D y(x.B, x.T, x.H, x.D);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int h=0;h<x.H;++h)
    {
        // Top2
        int e1=0,e2=0;
        float v1=-1e30f, v2=-1e30f;

        for(int e=0;e<experts_;++e){
            float v = gate.at(b,t,h,e);
            if(v > v1){ v2=v1; e2=e1; v1=v; e1=e; }
            else if(v > v2){ v2=v; e2=e; }
        }

        // softmax over top2 only（安定）
        float m = std::max(v1,v2);
        float a1 = std::exp(v1-m);
        float a2 = std::exp(v2-m);
        float inv = 1.0f / std::max(a1+a2, 1e-20f);
        a1 *= inv; a2 *= inv;

        // x の (b,t,h,:) を 1トークンTensorに切る（簡易）
        Tensor4D x1(1,1,1,x.D);
        for(int d=0; d<x.D; ++d) x1.at(0,0,0,d) = x.at(b,t,h,d);

        Tensor4D o1 = expert_[e1].forward(x1);
        Tensor4D o2 = expert_[e2].forward(x1);

        for(int d=0; d<x.D; ++d){
            y.at(b,t,h,d) = a1*o1.at(0,0,0,d) + a2*o2.at(0,0,0,d);
        }
    }

    return y;
}

std::vector<Tensor4D*> MoEFFN4D::parameters()
{
    std::vector<Tensor4D*> p;
    auto r = router_.parameters();
    p.insert(p.end(), r.begin(), r.end());
    for(auto& ex : expert_){
        auto ep = ex.parameters();
        p.insert(p.end(), ep.begin(), ep.end());
    }
    return p;
}