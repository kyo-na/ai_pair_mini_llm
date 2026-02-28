#pragma once
#include "world4d.h"

// World4D 専用・推論用インターフェース
struct WorldModel {
    World4D world;

    WorldModel(size_t B, size_t T, size_t D, size_t C);

    void step_infer(size_t t);
};