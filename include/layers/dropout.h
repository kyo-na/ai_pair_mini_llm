#pragma once
#include "tensor4d.h"

void apply_dropout(
    Tensor4D& x,
    float p,
    bool train
);