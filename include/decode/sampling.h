#pragma once
#include <vector>

int sample_token(
    std::vector<float> probs,
    float temperature,
    int top_k,
    float top_p
);