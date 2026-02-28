#include "decode/vocab_projection.h"

VocabProjection::VocabProjection(Tensor4D* tied_embedding)
: tied_weight_(tied_embedding)
{}

Tensor4D VocabProjection::forward(const Tensor4D& x)
{
    last_x_ = x;

    int B=x.B;
    int T=x.T;
    int H=x.H;
    int D=x.D;

    int vocab = tied_weight_->B; // assume (vocab,1,1,D)

    Tensor4D logits(B,T,H,vocab);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H;++h)
    for(int v=0;v<vocab;++v)
    {
        float sum=0;
        for(int d=0;d<D;++d)
            sum += x.at(b,t,h,d) *
                   tied_weight_->at(v,0,0,d);

        logits.at(b,t,h,v)=sum;
    }

    return logits;
}

Tensor4D VocabProjection::backward(const Tensor4D& grad_out)
{
    int B=last_x_.B;
    int T=last_x_.T;
    int H=last_x_.H;
    int D=last_x_.D;
    int vocab=tied_weight_->B;

    tied_weight_->zero_grad();

    Tensor4D dx(B,T,H,D);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H;++h)
    for(int v=0;v<vocab;++v)
    {
        float g=grad_out.at(b,t,h,v);

        for(int d=0;d<D;++d)
        {
            tied_weight_->grad_at(v,0,0,d)+=
                last_x_.at(b,t,h,d)*g;

            dx.at(b,t,h,d)+=
                tied_weight_->at(v,0,0,d)*g;
        }
    }

    return dx;
}

std::vector<Tensor4D*> VocabProjection::parameters()
{
    return { tied_weight_ };
}