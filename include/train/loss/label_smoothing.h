#pragma once
#include <vector>

class LabelSmoothing {
public:
    explicit LabelSmoothing(float eps = 0.1f) : eps_(eps) {}
    void apply(std::vector<float>& probs, unsigned target) const;

private:
    float eps_;
};