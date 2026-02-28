#pragma once
#include "tensor4d.h"
#include "layers/linear4d.h"
#include "layers/swiglu_ffn4d.h"
#include <vector>

class MoEFFN4D {
public:
    MoEFFN4D(int d_model, int hidden, int experts);

    // 推論forward（学習は次ステップで追加）
    Tensor4D forward(const Tensor4D& x);

    std::vector<Tensor4D*> parameters();

private:
    int experts_;
    Linear4D router_;                  // d_model -> experts
    std::vector<SwiGLUFFN4D> expert_;  // experts
};