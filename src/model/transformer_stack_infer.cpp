#include "model/transformer_stack_infer.h"

TransformerStackInfer::TransformerStackInfer(
    int layers,
    int heads,
    int head_dim,
    int ffn_hidden,
    int max_seq_len)
: layers_(layers),
  H_(heads),
  Dh_(head_dim),
  Dmodel_(heads*head_dim)
{
    for(int i=0;i<layers_;++i)
    {
        attn_.emplace_back(heads, head_dim);
        norm1_.emplace_back(heads, Dmodel_);
        norm2_.emplace_back(heads, Dmodel_);
        ffn_.emplace_back(heads, Dmodel_, ffn_hidden);

        caches_.emplace_back(heads, head_dim, max_seq_len);
    }
}

void TransformerStackInfer::reset()
{
    for(auto& c : caches_)
        c.reset();
}

Tensor4D TransformerStackInfer::forward_step(const Tensor4D& input)
{
    // input: (1,1,1,Dmodel_)
    Tensor4D x = input;

    for(int l=0;l<layers_;++l)
    {
        // ---- RMSNorm1 ----
        Tensor4D normed1 = norm1_[l].forward(x);

        // ---- Flash Attention ----
        Tensor4D attn_out =
            attn_[l].forward_flash(
                normed1,
                caches_[l]);

        // ---- Residual ----
        for(int d=0; d<Dmodel_; ++d)
            x.at(0,0,0,d) +=
                attn_out.at(0,0,0,d);

        // ---- RMSNorm2 ----
        Tensor4D normed2 = norm2_[l].forward(x);

        // ---- FFN ----
        Tensor4D ffn_out =
            ffn_[l].forward(normed2);

        // ---- Residual ----
        for(int d=0; d<Dmodel_; ++d)
            x.at(0,0,0,d) +=
                ffn_out.at(0,0,0,d);
    }

    return x;
}