#pragma once
#include "layers/attention4d.h"
#include "layers/rmsnorm4d.h"
#include "layers/swiglu4d.h"
#include <vector>

class TransformerBlock4D {
public:
    TransformerBlock4D(
        int heads,
        int head_dim,
        int ffn_hidden);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int H_;
    int D_;

    Attention4D attn_;
    RMSNorm4D norm1_;
    RMSNorm4D norm2_;
    SwiGLU4D ffn_;

    // backward用保存
    Tensor4D last_x_;
    Tensor4D last_n1_;
    Tensor4D last_attn_out_;
    Tensor4D last_res1_;
    Tensor4D last_n2_;
    Tensor4D last_ffn_out_;
};