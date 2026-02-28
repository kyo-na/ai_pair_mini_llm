#pragma once
#include "tensor4d.h"

float mse_loss(const Tensor4D& y, const Tensor4D& t);
Tensor4D mse_grad(const Tensor4D& y, const Tensor4D& t);