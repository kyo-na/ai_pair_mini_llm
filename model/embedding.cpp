#include "embedding.h"
#include <random>

namespace mini_llm {

Embedding::Embedding(int vocab, int dim)
: vocab_(vocab), dim_(dim),
  w_(vocab*dim), g_(vocab*dim,0.0f), tmp_(dim)
{
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-0.01f,0.01f);
    for (auto& x : w_) x = d(rng);
}

const std::vector<float>& Embedding::forward(int id) {
    for (int i=0;i<dim_;++i)
        tmp_[i] = w_[id*dim_ + i];
    return tmp_;
}

void Embedding::backward(int id, const std::vector<float>& grad) {
    for (int i=0;i<dim_;++i)
        g_[id*dim_ + i] += grad[i];
}

void Embedding::step(float lr) {
    for (size_t i=0;i<w_.size();++i) {
        w_[i] -= lr * g_[i];
        g_[i] = 0.0f;
    }
}

}