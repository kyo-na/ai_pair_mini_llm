#include "rlfj.h"
#include <cmath>

float RLFJ::evaluate(const std::vector<int>& tokens) {
    // ★ オリジナル要素
    // 例: 長さペナルティ + 繰り返し検出
    float reward = 0.0f;

    reward += tokens.size() * 0.01f;

    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i-1])
            reward -= 0.05f;
    }

    return reward;
}

float RLFJ::policy_loss(float reward, float logprob) {
    // シンプルな REINFORCE
    return -reward * logprob;
}