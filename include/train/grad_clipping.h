#pragma once
#include "../tensor4d.h"

void clip_grad_norm(
    Tensor4D& t,
    float max_norm
);