#include "train/optim/grad_view.h"

namespace mini_llm {
namespace optim {

void collect_grad_view(
    std::vector<GradView>& views,
    float* data,
    size_t size
) {
    if (!data || size == 0) return;
    views.push_back({ data, size });
}

} // namespace optim
} // namespace mini_llm