#include "train/optim/grad_clip.h"
#include <cmath>

namespace train {
namespace optim {

void clip_grad_norm(std::vector<Tensor4D*>& params, float max_norm)
{
    float total = 0.0f;

    for (auto* p : params)
        for (float g : p->grad)
            total += g * g;

    total = std::sqrt(total);
    if (total <= max_norm) return;

    float scale = max_norm / (total + 1e-6f);

    for (auto* p : params)
        for (auto& g : p->grad)
            g *= scale;
}

}} // namespace train::optim