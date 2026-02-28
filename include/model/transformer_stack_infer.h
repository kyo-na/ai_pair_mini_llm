#pragma once
#include <vector>
#include "layers/attention4d.h"
#include "layers/rmsnorm4d.h"
#include "layers/swiglu4d.h"
#include "cache/kv_cache4d.h"

class TransformerStackInfer {
public:
    TransformerStackInfer(
        int layers,
        int heads,
        int head_dim,
        int ffn_hidden,
        int max_seq_len);

    void reset();

    // single token step
    Tensor4D forward_step(const Tensor4D& x);

private:
    int layers_;
    int H_;
    int Dh_;
    int Dmodel_;

    std::vector<Attention4D> attn_;
    std::vector<RMSNorm4D> norm1_;
    std::vector<RMSNorm4D> norm2_;
    std::vector<SwiGLU4D> ffn_;

    std::vector<KVCache4D> caches_;
};