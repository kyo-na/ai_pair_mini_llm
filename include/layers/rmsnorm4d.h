#pragma once
#include "tensor4d.h"
#include <vector>

class RMSNorm4D {
public:
    explicit RMSNorm4D(int dim, float eps = 1e-6f);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    int D_;
    float eps_;

    Tensor4D gamma_;

    // backward用保存
    Tensor4D last_x_;
    std::vector<float> last_r_;
};