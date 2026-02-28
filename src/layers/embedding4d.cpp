#include "layers/embedding4d.h"
#include <cstdlib>
#include <cmath>

static float init_w(int fan_in)
{
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

Embedding4D::Embedding4D(int heads, int head_dim, int vocab_size)
    : H_(heads), D_(head_dim), vocab_(vocab_size),
      weight_(1,1,vocab_size, heads*head_dim)
{
    for(auto& w : weight_.data)
        w = init_w(heads*head_dim);
}

Tensor4D Embedding4D::forward(const std::vector<int>& ids)
{
    last_ids_ = ids;

    int B = 1;
    int T = (int)ids.size();

    Tensor4D out(B, T, H_, D_);

    for(int t=0; t<T; ++t)
    {
        int id = ids[t];

        for(int h=0; h<H_; ++h)
        for(int d=0; d<D_; ++d)
        {
            int flat = h*D_ + d;
            out.at(0,t,h,d) =
                weight_.at(0,0,id,flat);
        }
    }

    return out;
}

Tensor4D Embedding4D::backward(const Tensor4D& grad)
{
    int T = (int)last_ids_.size();

    for(int t=0; t<T; ++t)
    {
        int id = last_ids_[t];

        for(int h=0; h<H_; ++h)
        for(int d=0; d<D_; ++d)
        {
            int flat = h*D_ + d;
            weight_.grad_at(0,0,id,flat) +=
                grad.at(0,t,h,d);
        }
    }

    return Tensor4D(); // embeddingは下流に勾配流さない
}

std::vector<Tensor4D*> Embedding4D::parameters()
{
    return { &weight_ };
}