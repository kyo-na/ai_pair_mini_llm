#pragma once
#include <vector>
#include <cstddef>

struct World4D;

// token-level RLFJ：logprob を「その token の確信度」として報酬に混ぜる
class RLFJJudge {
public:
    // prev/curr: 直前と現在の world（4D）
    // logprob: 選んだ token の log P(token)
    float evaluate(const World4D& prev, const World4D& curr, float logprob) const;

private:
    float energy_score(const World4D& w) const;
    float entropy_proxy(const World4D& w) const; // ざっくり（分散ベース）
};