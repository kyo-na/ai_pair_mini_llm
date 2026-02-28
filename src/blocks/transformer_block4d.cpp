#include "blocks/transformer_block4d.h"

TransformerBlock4D::TransformerBlock4D(
    int heads,
    int head_dim,
    int ffn_hidden)
: H_(heads),
  D_(head_dim),
  attn_(heads, head_dim),
  norm1_(heads * head_dim),
  norm2_(heads * head_dim),
  ffn_(heads, head_dim, ffn_hidden)
{}

Tensor4D TransformerBlock4D::forward(const Tensor4D& x)
{
    last_x_ = x;

    // Norm1
    Tensor4D n1 = norm1_.forward(x);
    last_n1_ = n1;

    // Attention（シグネチャ一致：x, cache, use_cache）
    Tensor4D attn_out = attn_.forward(n1, nullptr, false);
    last_attn_out_ = attn_out;

    // Residual1 : res1 = x + attn_out
    Tensor4D res1 = x + attn_out;
    last_res1_ = res1;

    // Norm2
    Tensor4D n2 = norm2_.forward(res1);
    last_n2_ = n2;

    // FFN
    Tensor4D ffn_out = ffn_.forward(n2);
    last_ffn_out_ = ffn_out;

    // Residual2 : out = res1 + ffn_out
    Tensor4D out = res1 + ffn_out;
    return out;
}

Tensor4D TransformerBlock4D::backward(const Tensor4D& grad)
{
    // out = res1 + ffn_out
    Tensor4D d_res1 = grad;   // skip branch
    Tensor4D d_ffn  = grad;   // ffn branch

    // FFN backward -> d_n2
    Tensor4D d_n2 = ffn_.backward(d_ffn);

    // Norm2 backward -> adds to res1
    Tensor4D d_res1_from_norm2 = norm2_.backward(d_n2);
    d_res1 += d_res1_from_norm2;

    // res1 = x + attn_out
    Tensor4D d_x_skip = d_res1;
    Tensor4D d_attn_out = d_res1;

    // Attention backward -> d_n1
    Tensor4D d_n1 = attn_.backward(d_attn_out);

    // Norm1 backward -> d_x_from_norm1
    Tensor4D d_x_from_norm1 = norm1_.backward(d_n1);

    // total
    Tensor4D d_x = d_x_skip + d_x_from_norm1;
    return d_x;
}

std::vector<Tensor4D*> TransformerBlock4D::parameters()
{
    std::vector<Tensor4D*> ps;

    auto a = attn_.parameters();
    auto n1 = norm1_.parameters();
    auto n2 = norm2_.parameters();
    auto f = ffn_.parameters();

    ps.insert(ps.end(), a.begin(), a.end());
    ps.insert(ps.end(), n1.begin(), n1.end());
    ps.insert(ps.end(), n2.begin(), n2.end());
    ps.insert(ps.end(), f.begin(), f.end());

    return ps;
}