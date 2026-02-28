#pragma once
#include "tensor4d.h"

// World consistency loss
// L = || pred_{t+1} - actual_{t+1} ||^2
struct WorldConsistencyLoss {

    // loss 値
    float loss = 0.0f;

    // backward 用勾配
    Tensor4D grad;

    WorldConsistencyLoss() = default;

    // pred : Ŝ_{t+1}
// actual: S_{t+1}
    float forward(const Tensor4D& pred,
                  const Tensor4D& actual);

    const Tensor4D& backward() const;
};