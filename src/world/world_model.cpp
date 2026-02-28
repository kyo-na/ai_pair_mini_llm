#include "world_model.h"
#include <cmath>

WorldModel::WorldModel(size_t B, size_t T, size_t D, size_t C)
    : world(B, T, D, C) {}

void WorldModel::step_infer(size_t t) {
    // sample で動いていた 4D 更新ロジック
    for (size_t b = 0; b < world.B; ++b)
    for (size_t d = 0; d < world.D; ++d)
    for (size_t c = 0; c < world.C; ++c) {
        float prev = (t > 0) ? world.at(b, t-1, d, c) : 0.0f;
        world.at(b, t, d, c) = std::tanh(prev);
    }
}