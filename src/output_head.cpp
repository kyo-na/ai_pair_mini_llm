#include "output_head.h"
#include <cmath>
#include <random>
#include <algorithm>

static float randf() {
    static std::mt19937 rng(1234);
    static std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    return dist(rng);
}

OutputHead::OutputHead(size_t h, size_t v)
    : hidden(h), vocab(v)
{
    W.resize(hidden * vocab);
    b.resize(vocab);
    gradW.resize(hidden * vocab);
    gradb.resize(vocab);

    for (auto& x : W) x = randf();
    for (auto& x : b) x = 0.0f;
}

void OutputHead::forward(
    const std::vector<float>& h_t,
    std::vector<float>& logits)
{
    logits.assign(vocab, 0.0f);

    for (size_t j = 0; j < vocab; ++j) {
        float sum = b[j];
        for (size_t i = 0; i < hidden; ++i)
            sum += h_t[i] * W[i * vocab + j];
        logits[j] = sum;
    }
}

float OutputHead::backward(
    const std::vector<float>& h_t,
    const std::vector<float>& logits,
    int target)
{
    std::vector<float> p(vocab);
    float maxv = *std::max_element(
        logits.begin(), logits.end());

    float sum = 0.f;
    for (size_t i = 0; i < vocab; ++i) {
        p[i] = std::exp(logits[i] - maxv);
        sum += p[i];
    }
    for (auto& x : p) x /= sum;

    float loss = -std::log(p[target] + 1e-9f);

    for (size_t j = 0; j < vocab; ++j) {
        float g = p[j] - (j == (size_t)target ? 1.f : 0.f);
        gradb[j] += g;
        for (size_t i = 0; i < hidden; ++i)
            gradW[i * vocab + j] += g * h_t[i];
    }
    return loss;
}

void OutputHead::step(float lr)
{
    for (size_t i = 0; i < W.size(); ++i) {
        W[i] -= lr * gradW[i];
        gradW[i] = 0.0f;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] -= lr * gradb[i];
        gradb[i] = 0.0f;
    }
}