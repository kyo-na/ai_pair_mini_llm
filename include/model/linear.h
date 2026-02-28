#pragma once
#include <vector>

struct Linear {
    int in, out;
    std::vector<float> W, b;

    Linear(int i, int o);
    std::vector<float> forward(const std::vector<float>& x);
};