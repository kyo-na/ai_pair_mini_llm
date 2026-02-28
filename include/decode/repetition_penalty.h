#pragma once
#include <vector>

void apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int>& history,
    float penalty
);