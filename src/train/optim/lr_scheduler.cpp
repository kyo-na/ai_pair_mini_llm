// engine/mini_llm/src/train/optim/lr_scheduler.cpp
#include "train/optim/lr_scheduler.h"
#include <cmath>

static constexpr float PI = 3.14159265358979323846f;

float lr_schedule(
    int step,
    int warmup,
    int total,
    float base_lr
){
    if(step < warmup){
        return base_lr * float(step) / float(warmup);
    }
    if(step >= total){
        return 0.0f;
    }
    float p = float(step - warmup) / float(total - warmup);
    return base_lr * 0.5f * (1.0f + std::cos(PI * p));
}