#include "decode/sampling4d.h"
#include <algorithm>
#include <random>
#include <cmath>

int sample_next_token(
    std::vector<float>& logits,
    float temperature,
    int top_k,
    float top_p)
{
    // Temperature
    for (auto& v : logits)
        v /= temperature;

    // softmax
    float maxv = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (auto& v : logits)
    {
        v = std::exp(v - maxv);
        sum += v;
    }

    for (auto& v : logits)
        v /= sum;

    // Top-K
    if (top_k > 0)
    {
        std::vector<float> sorted = logits;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        float threshold = sorted[top_k-1];

        for (auto& v : logits)
            if (v < threshold)
                v = 0.0f;
    }

    // Top-P
    if (top_p < 1.0f)
    {
        std::vector<float> sorted = logits;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());

        float cumulative = 0.0f;
        float cutoff = 0.0f;

        for (float v : sorted)
        {
            cumulative += v;
            if (cumulative >= top_p)
            {
                cutoff = v;
                break;
            }
        }

        for (auto& v : logits)
            if (v < cutoff)
                v = 0.0f;
    }

    // renormalize
    sum = 0.0f;
    for (auto v : logits) sum += v;
    for (auto& v : logits) v /= (sum + 1e-9f);

    std::discrete_distribution<int> dist(logits.begin(), logits.end());
    std::mt19937 gen(std::random_device{}());

    return dist(gen);
}