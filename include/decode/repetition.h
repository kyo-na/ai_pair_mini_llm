#pragma once
#include <vector>

// logits を直接いじる（softmax 前）
void apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<unsigned>& history,
    float penalty
);