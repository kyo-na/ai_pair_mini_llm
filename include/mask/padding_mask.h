#pragma once
#include <vector>

void apply_padding_mask(
    std::vector<float>& logits,
    const std::vector<int>& valid_len,
    int B,int T,int V
);