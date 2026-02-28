#include "linear.h"
#include <random>

namespace mini_llm {

Linear::Linear(int in,int out)
: in_(in), out_(out),
  w_(in*out), b_(out,0.0f),
  gw_(in*out,0.0f), gb_(out,0.0f),
  tmp_(out)
{
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> d(-0.01f,0.01f);
    for (auto& x : w_) x = d(rng);
}

const std::vector<float>& Linear::forward(const std::vector<float>& x) {
    for (int o=0;o<out_;++o) {
        float s = b_[o];
        for (int i=0;i<in_;++i)
            s += w_[o*in_ + i] * x[i];
        tmp_[o] = s;
    }
    return tmp_;
}

std::vector<float> Linear::backward(const std::vector<float>& grad) {
    std::vector<float> dx(in_,0.0f);
    for (int o=0;o<out_;++o) {
        gb_[o] += grad[o];
        for (int i=0;i<in_;++i) {
            gw_[o*in_ + i] += grad[o];
            dx[i] += w_[o*in_ + i] * grad[o];
        }
    }
    return dx;
}

void Linear::step(float lr) {
    for (size_t i=0;i<w_.size();++i) {
        w_[i] -= lr * gw_[i];
        gw_[i] = 0.0f;
    }
    for (size_t i=0;i<b_.size();++i) {
        b_[i] -= lr * gb_[i];
        gb_[i] = 0.0f;
    }
}

}