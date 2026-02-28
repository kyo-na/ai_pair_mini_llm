#include "model/transformer_stack4d.h"

TransformerStack4D::TransformerStack4D(
    int layers,
    int heads,
    int head_dim,
    int ffn_hidden)
    : num_layers_(layers)
{
    for(int i = 0; i < layers; ++i)
    {
        blocks_.emplace_back(heads, head_dim, ffn_hidden);
    }
}

Tensor4D TransformerStack4D::forward(
    const Tensor4D& x,
    bool use_cache)
{
    Tensor4D out = x;

    for(auto& block : blocks_)
    {
        out = block.forward(out);  // ← use_cache渡さない
    }

    return out;
}

Tensor4D TransformerStack4D::backward(const Tensor4D& grad)
{
    Tensor4D g = grad;

    for(int i = (int)blocks_.size() - 1; i >= 0; --i)
    {
        g = blocks_[i].backward(g);
    }

    return g;
}

std::vector<Tensor4D*> TransformerStack4D::parameters()
{
    std::vector<Tensor4D*> ps;

    for(auto& b : blocks_)
    {
        auto p = b.parameters();
        ps.insert(ps.end(), p.begin(), p.end());
    }

    return ps;
}