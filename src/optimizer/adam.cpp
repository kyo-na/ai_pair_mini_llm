#include "optimizer/adam.h"
#include <cmath>

Adam::Adam(float lr)
    : lr_(lr),
      beta1_(0.9f),
      beta2_(0.999f),
      eps_(1e-8f),
      t_(0)
{
}

void Adam::step(std::vector<Tensor4D*>& params)
{
    t_++;

    for (auto* p : params)
    {
        if (m_.find(p) == m_.end())
        {
            m_[p] = std::vector<float>(p->data.size(), 0.0f);
            v_[p] = std::vector<float>(p->data.size(), 0.0f);
        }

        for (size_t i = 0; i < p->data.size(); ++i)
        {
            float g = p->grad[i];

            m_[p][i] = beta1_ * m_[p][i]
                     + (1 - beta1_) * g;

            v_[p][i] = beta2_ * v_[p][i]
                     + (1 - beta2_) * g * g;

            float m_hat = m_[p][i] /
                (1 - std::pow(beta1_, t_));

            float v_hat = v_[p][i] /
                (1 - std::pow(beta2_, t_));

            p->data[i] -= lr_ *
                m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}