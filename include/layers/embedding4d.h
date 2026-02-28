#pragma once
#include "tensor4d.h"
#include <vector>

class Embedding4D {
public:
    Embedding4D(int heads, int head_dim, int vocab_size);

    Tensor4D forward(const std::vector<int>& ids);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int H_;
    int D_;
    int vocab_;

    Tensor4D weight_;   // (1,1,vocab, H*D)

    std::vector<int> last_ids_;
};