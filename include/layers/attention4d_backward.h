#pragma once
#include "tensor4d.h"

class AttentionBackward4D {
public:
    Tensor4D backward(const Tensor4D& grad,
                      const Tensor4D& Q,
                      const Tensor4D& K,
                      const Tensor4D& V);
};