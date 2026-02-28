#pragma once
#include <vector>
#include "tensor4d.h"

namespace train {
namespace optim {

void clip_grad_norm(std::vector<Tensor4D*>& params, float max_norm);

}} // namespace train::optim