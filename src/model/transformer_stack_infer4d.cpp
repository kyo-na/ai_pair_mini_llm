#include "model/transformer_stack_infer4d.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

static float init_w(int fan_in){
    return std::sqrt(2.0f/fan_in) * (((float)rand()/RAND_MAX)-0.5f);
}

TransformerStackInfer4D::TransformerStackInfer4D(int layers,int vocab,int H,int D,int maxT)
: L_(layers), vocab_(vocab), H_(H), D_(D),
  emb_(vocab, H, D),
  W_vocab_(1,1,D,vocab)
{
    attn_.reserve((size_t)L_);
    norm1_.reserve((size_t)L_);
    norm2_.reserve((size_t)L_);
    ffn_.reserve((size_t)L_);
    kv_.reserve((size_t)L_);

    for(int i=0;i<L_;++i){
        attn_.emplace_back(H, D);
        norm1_.emplace_back(D);
        norm2_.emplace_back(D);
        ffn_.emplace_back(D, D*4);   // hidden=4x
        kv_.emplace_back(1, maxT, H, D);
    }

    for(auto& w: W_vocab_.data) w = init_w(D);
}

void TransformerStackInfer4D::reset()
{
    for(auto& k : kv_) k.reset();
}

std::vector<float> TransformerStackInfer4D::forward_step(int32_t token_id, InferContext& ctx)
{
    ctx.reset_step();

    // x: (1,1,H,D)
    std::vector<int32_t> ids = { token_id };
    Tensor4D x = emb_.forward(ids);

    for(int i=0;i<L_;++i){
        // norm + attn (fused)
        Tensor4D xn = norm1_[i].forward(x);

        Tensor4D a = attn_[i].forward_infer_fused(xn, &kv_[i], ctx);

        // residual
        for(int d=0; d<x.D; ++d)
            x.at(0,0,0,d) += a.at(0,0,0,d);

        // norm + ffn
        Tensor4D x2 = norm2_[i].forward(x);
        Tensor4D f = ffn_[i].forward(x2);

        for(int d=0; d<x.D; ++d)
            x.at(0,0,0,d) += f.at(0,0,0,d);
    }

    // vocab head: logits[v] = dot(x, W_vocab[:,v])
    std::vector<float> logits((size_t)vocab_, 0.0f);
    for(int v=0; v<vocab_; ++v){
        float s=0;
        for(int d=0; d<D_; ++d){
            s += x.at(0,0,0,d) * W_vocab_.at(0,0,d,v);
        }
        logits[(size_t)v] = s;
    }
    return logits;
}