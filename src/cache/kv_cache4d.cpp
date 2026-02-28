#include "cache/kv_cache4d.h"

KVCache4D::KVCache4D(
    int heads,
    int head_dim,
    int max_len)
: H_(heads),
  Dh_(head_dim),
  max_len_(max_len),
  cur_len_(0),
  K_cache_(1,max_len,heads,head_dim),
  V_cache_(1,max_len,heads,head_dim)
{}

void KVCache4D::reset()
{
    cur_len_=0;
}

void KVCache4D::append(
    const Tensor4D& K_new,
    const Tensor4D& V_new)
{
    int index = cur_len_ % max_len_;

    for(int h=0;h<H_;++h)
    for(int d=0;d<Dh_;++d)
    {
        K_cache_.at(0,index,h,d)
            =K_new.at(0,0,h,d);

        V_cache_.at(0,index,h,d)
            =V_new.at(0,0,h,d);
    }

    cur_len_++;
}

Tensor4D& KVCache4D::K(){ return K_cache_; }
Tensor4D& KVCache4D::V(){ return V_cache_; }

int KVCache4D::length() const
{
    return std::min(cur_len_, max_len_);
}