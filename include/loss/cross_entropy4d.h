#pragma once
#include "tensor4d.h"

class CrossEntropy4D {
public:
    float forward(const Tensor4D& logits,
                  const Tensor4D& target);

    Tensor4D backward();

private:
    Tensor4D last_logits_;
    Tensor4D last_target_;
};