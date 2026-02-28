#include "layers/ffn4d.h"

FFN4D::FFN4D(int heads, int dim, float dropout_p)
    : linear1_(heads, dim),
      linear2_(heads, dim),
      dropout_(dropout_p)
{
}

Tensor4D FFN4D::forward(const Tensor4D& x, bool train_mode)
{
    hidden_ = linear1_.forward(x);

    // ReLU
    for(int b=0;b<hidden_.B;++b)
    for(int t=0;t<hidden_.T;++t)
    for(int h=0;h<hidden_.H;++h)
    for(int d=0;d<hidden_.D;++d)
        if(hidden_.at(b,t,h,d) < 0)
            hidden_.at(b,t,h,d) = 0;

    hidden_ = dropout_.forward(hidden_, train_mode);

    return linear2_.forward(hidden_);
}

Tensor4D FFN4D::backward(const Tensor4D& grad)
{
    // Linear2 backward
    Tensor4D dHidden = linear2_.backward(grad);

    // Dropout backward
    dHidden = dropout_.backward(dHidden);

    // ReLU backward
    for(int b=0;b<dHidden.B;++b)
    for(int t=0;t<dHidden.T;++t)
    for(int h=0;h<dHidden.H;++h)
    for(int d=0;d<dHidden.D;++d)
        if(hidden_.at(b,t,h,d) <= 0)
            dHidden.at(b,t,h,d) = 0;

    return linear1_.backward(dHidden);
}

std::vector<Tensor4D*> FFN4D::parameters()
{
    std::vector<Tensor4D*> p;

    auto p1 = linear1_.parameters();
    auto p2 = linear2_.parameters();

    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());

    return p;
}