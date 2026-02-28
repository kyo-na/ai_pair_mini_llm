#pragma once
#include <cmath>
#include <algorithm>

class LRScheduler {
public:
    LRScheduler(
        float base_lr,
        int warmup_steps,
        int total_steps,
        float min_lr = 1e-5f)
        : base_lr_(base_lr),
          warmup_steps_(warmup_steps),
          total_steps_(total_steps),
          min_lr_(min_lr)
    {}

    float get_lr(int step) const
    {
        if (step < warmup_steps_)
        {
            return base_lr_ *
                   (float(step) / float(warmup_steps_));
        }

        float progress =
            float(step - warmup_steps_) /
            float(total_steps_ - warmup_steps_);

        progress = std::clamp(progress, 0.0f, 1.0f);

        const float PI = 3.14159265358979323846f;

        float cosine =
            0.5f * (1.0f + std::cos(PI * progress));

        return min_lr_ +
               (base_lr_ - min_lr_) * cosine;
    }

private:
    float base_lr_;
    int warmup_steps_;
    int total_steps_;
    float min_lr_;
};