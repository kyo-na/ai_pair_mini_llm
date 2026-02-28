#pragma once
#include "tensor4d.h"
#include "cache/kv_cache4d.h"
#include "cache/kv_cache_ring4d.h"
#include "runtime/infer_context.h"
#include <vector>

class Attention4D {
public:
    Attention4D(int H,int D);

    // 学習・通常forward（既存互換）
    Tensor4D forward(
        const Tensor4D& x,
        KVCache4D* cache=nullptr,
        bool use_cache=false);

    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

    // 推論専用 fused
    Tensor4D forward_infer_fused(
        const Tensor4D& x_step,
        KVCacheRing4D* ring,
        InferContext& ctx);

private:
    int H_;
    int D_;

    Tensor4D Wq_;
    Tensor4D Wk_;
    Tensor4D Wv_;
    Tensor4D Wo_;

    // backward用保存
    Tensor4D last_x_;
    Tensor4D last_Q_;
    Tensor4D last_K_;
    Tensor4D last_V_;
    Tensor4D last_context_;
    Tensor4D last_attn_; // (B,T,H,T) ただし未来側は0
};