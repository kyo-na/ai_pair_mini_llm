#pragma once
#include <vector>
#include "blocks/transformer_block4d.h"
#include "tensor4d.h"

class TransformerStack4D {
public:
    TransformerStack4D(
        int layers,
        int heads,
        int head_dim,
        int ffn_hidden);

    Tensor4D forward(
        const Tensor4D& x,
        bool use_cache = false);

    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int num_layers_;

    // ★ これが無いからエラー出てた
    std::vector<TransformerBlock4D> blocks_;
};