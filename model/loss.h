#pragma once
#include <vector>

namespace mini_llm {
float mse_loss(const std::vector<float>& y,
               const std::vector<float>& t);
std::vector<float> mse_grad(const std::vector<float>& y,
                            const std::vector<float>& t);
}