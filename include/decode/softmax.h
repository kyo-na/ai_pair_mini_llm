#pragma once
#include <vector>

class Softmax {
public:
    static void apply(const std::vector<float>& logits,
                      std::vector<float>& probs);
};