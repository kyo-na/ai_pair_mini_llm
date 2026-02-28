#pragma once
#include "tensor4d.h"
#include <vector>

void apply_repetition_penalty(
    Tensor4D& logits,
    const std::vector<int>& history,
    float penalty);