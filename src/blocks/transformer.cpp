#include "blocks/transformer_block.h"

TransformerBlock::TransformerBlock(int dim)
    : ln1(dim), attn(dim), ln2(dim), ffn(dim) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    Tensor h1 = ln1.forward(x);
    Tensor a = attn.forward(h1);
    Tensor r1 = a + x;
    Tensor h2 = ln2.forward(r1);
    Tensor f = ffn.forward(h2);
    return f + r1;
}

Tensor TransformerBlock::backward(const Tensor& dout) {
    Tensor df = ffn.backward(dout);
    Tensor dr = df + dout;
    Tensor dh = ln2.backward(dr);
    Tensor da = attn.backward(dh);
    return ln1.backward(da) + dr;
}