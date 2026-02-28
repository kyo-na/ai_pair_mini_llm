#pragma once
#include <vector>
#include <algorithm>
#include <cassert>

class Tensor4D {
public:
    int B,T,H,D;

    std::vector<float> data;
    std::vector<float> grad;

    Tensor4D()
        : B(0),T(0),H(0),D(0)
    {}

    Tensor4D(int b,int t,int h,int d)
        : B(b),T(t),H(h),D(d),
          data((size_t)b*t*h*d, 0.0f),
          grad((size_t)b*t*h*d, 0.0f)
    {}

    inline int index(int b,int t,int h,int d) const
    {
        return ((b*T + t)*H + h)*D + d;
    }

    inline float& at(int b,int t,int h,int d)
    {
        return data[(size_t)index(b,t,h,d)];
    }

    inline float at(int b,int t,int h,int d) const
    {
        return data[(size_t)index(b,t,h,d)];
    }

    inline float& grad_at(int b,int t,int h,int d)
    {
        return grad[(size_t)index(b,t,h,d)];
    }

    void zero()
    {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    void zero_grad()
    {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }

    // =========================
    // DL拡張（既存を壊さない）
    // =========================

    // shape一致の要素加算
    Tensor4D add(const Tensor4D& other) const
    {
        assert(B==other.B && T==other.T && H==other.H && D==other.D);
        Tensor4D out(B,T,H,D);
        for (size_t i=0;i<data.size();++i) out.data[i] = data[i] + other.data[i];
        return out;
    }

    void add_inplace(const Tensor4D& other)
    {
        assert(B==other.B && T==other.T && H==other.H && D==other.D);
        for (size_t i=0;i<data.size();++i) data[i] += other.data[i];
    }

    // スカラー倍
    Tensor4D mul_scalar(float s) const
    {
        Tensor4D out(B,T,H,D);
        for (size_t i=0;i<data.size();++i) out.data[i] = data[i] * s;
        return out;
    }

    // (B,T,H,D) -> (B,T,D,H)
    Tensor4D transpose_last2() const
    {
        Tensor4D out(B,T,D,H);
        for(int b=0;b<B;++b)
        for(int t=0;t<T;++t)
        for(int h=0;h<H;++h)
        for(int d=0;d<D;++d)
            out.at(b,t,d,h) = at(b,t,h,d);
        return out;
    }

    // (B,T,H,D) x (B,T,D,K) -> (B,T,H,K)
    // ※ other は (B,T,D,K) を想定 → other.H==D
    Tensor4D matmul_lastdim(const Tensor4D& other) const
    {
        assert(B==other.B && T==other.T);
        assert(D==other.H);
        int K = other.D;

        Tensor4D out(B,T,H,K);

        for(int b=0;b<B;++b)
        for(int t=0;t<T;++t)
        for(int h=0;h<H;++h)
        for(int k=0;k<K;++k)
        {
            float sum = 0.0f;
            for(int d=0; d<D; ++d)
                sum += at(b,t,h,d) * other.at(b,t,d,k);
            out.at(b,t,h,k) = sum;
        }
        return out;
    }

    // operatorはラッパ（既存互換）
    Tensor4D operator+(const Tensor4D& other) const { return add(other); }
    Tensor4D& operator+=(const Tensor4D& other) { add_inplace(other); return *this; }
    Tensor4D operator*(float s) const { return mul_scalar(s); }
};