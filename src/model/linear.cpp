#include "../../include/model/linear.h"

Linear::Linear(int i, int o)
: in(i), out(o), W(i*o), b(o) {}

std::vector<float> Linear::forward(const std::vector<float>& x) {
    std::vector<float> y(out);
    for (int o = 0; o < out; ++o) {
        float s = b[o];
        for (int i = 0; i < in; ++i)
            s += W[o*in + i] * x[i];
        y[o] = s;
    }
    return y;
}