#include "train/loss/label_smoothing.h"

void LabelSmoothing::apply(std::vector<float>& p, unsigned t) const {
    float n = float(p.size());
    for (float& x : p) x *= (1.0f - eps_);
    if (t < p.size()) p[t] += eps_ / n;
}