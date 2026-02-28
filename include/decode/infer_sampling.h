#pragma once
#include <vector>
#include <cstdint>

struct InferSamplingConfig {
    float temperature = 1.0f;
    int top_k = 0;          // 0=無効
    float top_p = 0.0f;     // 0=無効
    float repetition_penalty = 1.0f;
    uint32_t rng_seed = 1234;
};

int sample_next_token(
    const std::vector<float>& logits,
    const std::vector<int32_t>& recent_tokens, // 直近履歴（repetition用）
    const InferSamplingConfig& cfg);