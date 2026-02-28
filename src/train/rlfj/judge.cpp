#include "train/rlfj/judge.h"
#include "world4d.h"
#include <cmath>
#include <algorithm>

float RLFJJudge::energy_score(const World4D& w) const {
    // L2 エネルギー（大きすぎる表現を罰する）
    double e = 0.0;
    for (float v : w.data) e += (double)v * (double)v;
    return (float)e;
}

float RLFJJudge::entropy_proxy(const World4D& w) const {
    // “ばらつき”が小さすぎる（全ゼロ/固定）も罰するため、分散から proxy を作る
    if (w.data.empty()) return 0.0f;
    double mean = 0.0;
    for (float v : w.data) mean += v;
    mean /= (double)w.data.size();

    double var = 0.0;
    for (float v : w.data) {
        double d = (double)v - mean;
        var += d * d;
    }
    var /= (double)w.data.size();
    return (float)var;
}

float RLFJJudge::evaluate(const World4D& prev, const World4D& curr, float logprob) const {
    (void)prev;

    const float energy  = energy_score(curr);
    const float var     = entropy_proxy(curr);

    // 目標：エネルギーは抑えたい、でも完全固定も嫌、logprob は高い方が良い
    // ※重みは後でチューニング
    float reward = 0.0f;
    reward += (+0.10f) * logprob;                 // token の確信
    reward += (-0.0001f) * energy;                // 発散抑制
    reward += (+0.01f) * std::sqrt(std::max(0.0f, var)); // 固定化抑制（弱）

    return reward;
}