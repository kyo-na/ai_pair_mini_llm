#pragma once
#include <vector>

void softmax_vocab(
    const std::vector<float>& logits,
    std::vector<float>& probs,
    int B,int T,int V
);