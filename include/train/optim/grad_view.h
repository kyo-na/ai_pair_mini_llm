#pragma once
#include <vector>

namespace mini_llm {
namespace optim {

// 任意の Tensor メモリを 1D view として追加
struct GradView {
    float* data;
    size_t size;
};

void collect_grad_view(
    std::vector<GradView>& views,
    float* data,
    size_t size
);

} // namespace optim
} // namespace mini_llm