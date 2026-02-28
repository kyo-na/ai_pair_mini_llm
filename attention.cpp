#include "attention.h"
#include <cmath>

namespace mini_llm {

Attention::Attention(int dim) : dim_(dim) {}

std::vector<float> Attention::forward(
    const std::vector<std::vector<float>>& context)
{
    // 超最小：平均 Attention（合法・安定）
    std::vector<float> out(dim_, 0.0f);
    if (context.empty()) return out;

    for (auto& v : context) {
        for (int i = 0; i < dim_; ++i)
            out[i] += v[i];
    }
    for (int i = 0; i < dim_; ++i)
        out[i] /= context.size();

    return out;
}

}