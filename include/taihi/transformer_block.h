#pragma once
#include "layers/attention.h"
#include "layers/layernorm.h"
#include "layers/ffn.h"

struct TransformerBlock {
    Attention attn;
    LayerNorm ln1;
    FFN ffn;
    LayerNorm ln2;

    Tensor x1, x2;

    TransformerBlock(int dim);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dy);
};