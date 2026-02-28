#pragma once
#include "layers/layernorm.h"
#include "layers/attention.h"
#include "layers/ffn.h"

struct TransformerBlock {
    LayerNorm ln1;
    Attention attn;
    LayerNorm ln2;
    FFN ffn;

    TransformerBlock(int dim);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};