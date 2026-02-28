#pragma once
#include <vector>

class RLFJ {
public:
    // 生成されたトークン列を評価
    float evaluate(const std::vector<int>& tokens);

    // policy loss へ変換
    float policy_loss(float reward, float logprob);
};