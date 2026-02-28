#pragma once
#include <cmath>

inline float perplexity(float loss)
{
    return std::exp(loss);
}