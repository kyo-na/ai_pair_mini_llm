#pragma once
#include "tensor4d.h"
#include <vector>

class CrossEntropy {
public:
    float forward(const Tensor4D& logits,
                  const std::vector<int>& target);

    Tensor4D backward();

private:
    Tensor4D probs_;
    std::vector<int> target_;
};