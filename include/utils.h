#pragma once
#include <random>

inline float init_uniform() {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(-0.1f,0.1f);
    return dist(rng);
}