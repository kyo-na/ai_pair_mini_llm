#include "loss/mse4d.h"

float mse_loss(const Tensor4D& y, const Tensor4D& t){
    float s=0;
    for(size_t i=0;i<y.data.size();i++){
        float d=y.data[i]-t.data[i];
        s+=d*d;
    }
    return s / y.data.size();
}

Tensor4D mse_grad(const Tensor4D& y, const Tensor4D& t){
    Tensor4D g=y;
    for(size_t i=0;i<g.data.size();i++){
        g.data[i]=2.f*(y.data[i]-t.data[i])/g.data.size();
    }
    return g;
}