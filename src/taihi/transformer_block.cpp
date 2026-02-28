#pragma once
#include "layers/attention.h"
#include "layers/ffn.h"
#include "layers/layernorm.h"

struct TransformerBlock {
    Attention attn;
    FFN ffn;
    LayerNorm ln1,ln2;

    TransformerBlock(int d)
        :attn(d),ffn(d,d*4),ln1(d),ln2(d){}
};