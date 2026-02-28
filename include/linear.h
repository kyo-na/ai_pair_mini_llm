#pragma once
#include "tensor.h"
#include <vector>

struct Linear {
    Tensor2D W;
    std::vector<float> b;

    Linear(int in,int out):W(in,out),b(out,0.f){}
    std::vector<float> forward(const std::vector<float>& x);
};