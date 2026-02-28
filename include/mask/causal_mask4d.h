#pragma once
#include "tensor4d.h"

class CausalMask4D {
public:
    static void apply(Tensor4D& scores);
};