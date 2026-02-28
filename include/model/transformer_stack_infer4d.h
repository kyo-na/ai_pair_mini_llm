#pragma once
#include "tensor4d.h"
#include "runtime/infer_context.h"
#include "cache/kv_cache_ring4d.h"
#include "layers/embedding4d.h"
#include "layers/attention4d.h"
#include "layers/rmsnorm4d.h"
#include "layers/swiglu_ffn4d.h"
#include <vector>

class TransformerStackInfer4D {
public:
    TransformerStackInfer4D(int layers, int vocab, int H, int D, int maxT);

    // ids: length=1 の 1トークン入力（B=1前提でもOK）
    // return logits: std::vector<float>(vocab)
    std::vector<float> forward_step(int32_t token_id, InferContext& ctx);

    void reset();

private:
    int L_;
    int vocab_;
    int H_;
    int D_;

    Embedding4D emb_;
    std::vector<Attention4D> attn_;
    std::vector<RMSNorm4D> norm1_;
    std::vector<RMSNorm4D> norm2_;
    std::vector<SwiGLUFFN4D> ffn_;

    // 層別RingKV
    std::vector<KVCacheRing4D> kv_;

    // vocab head（いまは linear_vocab4d を使うより安全に “簡易 head”）
    Tensor4D W_vocab_; // (1,1,D,vocab)
};