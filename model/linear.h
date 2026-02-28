#pragma once
#include <vector>

namespace mini_llm {

class Linear {
public:
    Linear(int in, int out);
    const std::vector<float>& forward(const std::vector<float>& x);
    std::vector<float> backward(const std::vector<float>& grad);
    void step(float lr);

private:
    int in_, out_;
    std::vector<float> w_, b_, gw_, gb_, tmp_;
};

}