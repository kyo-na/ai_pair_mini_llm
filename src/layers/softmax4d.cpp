#include "layers/softmax4d.h"
#include <cmath>
#include <algorithm>

Tensor4D Softmax4D::forward(const Tensor4D& x){
    Tensor4D y=x;

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t){
        float m=-1e9f;
        for(int v=0;v<x.D;++v)
            m=std::max(m,x.at(b,t,0,v));

        float sum=0;
        for(int v=0;v<x.D;++v){
            float e=std::exp(x.at(b,t,0,v)-m);
            y.at(b,t,0,v)=e;
            sum+=e;
        }
        for(int v=0;v<x.D;++v)
            y.at(b,t,0,v)/=sum;
    }
    last_y_=y;
    return y;
}

Tensor4D Softmax4D::backward(const Tensor4D& grad){
    Tensor4D dx=grad;
    for(int b=0;b<dx.B;++b)
    for(int t=0;t<dx.T;++t)
    for(int i=0;i<dx.D;++i){
        float s=0;
        for(int j=0;j<dx.D;++j){
            float term=(i==j)
              ? last_y_.at(b,t,0,i)*(1-last_y_.at(b,t,0,j))
              : -last_y_.at(b,t,0,i)*last_y_.at(b,t,0,j);
            s+=grad.at(b,t,0,j)*term;
        }
        dx.at(b,t,0,i)=s;
    }
    return dx;
}