#pragma once
#include <vector>

void apply_causal_mask(
    std::vector<float>& logits,
    int B,int T,int V
);