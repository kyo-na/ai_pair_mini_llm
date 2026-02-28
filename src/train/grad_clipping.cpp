#include "train/grad_clipping.h"
#include <cmath>

void clip_grad_norm(Tensor4D& t, float max_norm) {
    float sum_sq = 0.0f;
    for (float g : t.grad) sum_sq += g * g;

    float norm = std::sqrt(sum_sq);
    if (norm <= max_norm) return;

    float scale = max_norm / (norm + 1e-6f);
    for (float& g : t.grad) g *= scale;
}