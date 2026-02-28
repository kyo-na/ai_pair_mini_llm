#pragma once
#include <vector>
#include <unordered_map>
#include "tensor4d.h"

class Adam {
public:
    Adam(float lr = 1e-3f);

    void step(std::vector<Tensor4D*>& params);

private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;

    int t_;

    // parameter pointer ごとに保持
    std::unordered_map<Tensor4D*, std::vector<float>> m_;
    std::unordered_map<Tensor4D*, std::vector<float>> v_;
};