#include "optimizer/adamw.h"

AdamW::AdamW(float lr):lr_(lr){}

void AdamW::step(std::vector<Tensor4D*>& params)
{
    for(auto* p:params)
    {
        for(size_t i=0;i<p->data.size();++i)
        {
            p->data[i]-=lr_*p->grad[i];
            p->grad[i]=0.0f;
        }
    }
}