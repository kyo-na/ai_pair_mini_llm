#pragma once
#include <vector>
#include <cstddef>

struct OutputHead {
    size_t hidden;
    size_t vocab;

    std::vector<float> W;
    std::vector<float> b;
    std::vector<float> gradW;
    std::vector<float> gradb;

    OutputHead(size_t h, size_t v);

    void forward(
        const std::vector<float>& h_t,
        std::vector<float>& logits);

    float backward(
        const std::vector<float>& h_t,
        const std::vector<float>& logits,
        int target);

    void step(float lr);
};