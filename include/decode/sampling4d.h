#pragma once
#include <vector>

int sample_next_token(
    std::vector<float>& logits,
    float temperature,
    int top_k,
    float top_p);