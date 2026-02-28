#pragma once
#include "tensor4d.h"
#include <algorithm>

// 形状: K,V = (B, maxT, H, D)
class KVCacheRing4D {
public:
    KVCacheRing4D(int B, int maxT, int H, int D)
        : B_(B), maxT_(maxT), H_(H), D_(D),
          K_(B, maxT, H, D), V_(B, maxT, H, D),
          len_(0), head_(0) {}

    void reset() { len_ = 0; head_ = 0; }

    // 1ステップ分の K,V (B,1,H,D) を追加
    void append_step(const Tensor4D& K1, const Tensor4D& V1) {
        const int t = head_;
        for(int b=0;b<B_;++b)
        for(int h=0;h<H_;++h)
        for(int d=0;d<D_;++d){
            K_.at(b,t,h,d) = K1.at(b,0,h,d);
            V_.at(b,t,h,d) = V1.at(b,0,h,d);
        }
        head_ = (head_ + 1) % maxT_;
        len_ = std::min(len_ + 1, maxT_);
    }

    // “論理順”でアクセスする: 0..len_-1 が過去→現在
    inline float K_at(int b,int logical_t,int h,int d) const {
        int phys = (head_ - len_ + logical_t);
        if(phys < 0) phys += maxT_;
        return K_.at(b, phys, h, d);
    }

    inline float V_at(int b,int logical_t,int h,int d) const {
        int phys = (head_ - len_ + logical_t);
        if(phys < 0) phys += maxT_;
        return V_.at(b, phys, h, d);
    }

    int length() const { return len_; }
    int maxT() const { return maxT_; }
    int B() const { return B_; }
    int H() const { return H_; }
    int D() const { return D_; }

private:
    int B_, maxT_, H_, D_;
    Tensor4D K_, V_;
    int len_;
    int head_;
};