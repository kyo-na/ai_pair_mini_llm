#include "loss.h"

namespace mini_llm {

float mse_loss(const std::vector<float>& y,
               const std::vector<float>& t) {
    float s=0.0f;
    for (size_t i=0;i<y.size();++i) {
        float d = y[i]-t[i];
        s += d*d;
    }
    return s / y.size();
}

std::vector<float> mse_grad(const std::vector<float>& y,
                            const std::vector<float>& t) {
    std::vector<float> g(y.size());
    for (size_t i=0;i<y.size();++i)
        g[i] = 2.0f*(y[i]-t[i]) / y.size();
    return g;
}

}