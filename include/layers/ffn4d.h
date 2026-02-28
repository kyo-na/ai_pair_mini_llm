#pragma once
#include <vector>
#include "tensor4d.h"
#include "layers/linear4d.h"
#include "layers/dropout4d.h"

class FFN4D {
public:
    FFN4D(int heads, int dim, float dropout_p);

    Tensor4D forward(const Tensor4D& x, bool train_mode);
    Tensor4D backward(const Tensor4D& grad);

    std::vector<Tensor4D*> parameters();

private:
    Linear4D linear1_;
    Linear4D linear2_;
    Dropout4D dropout_;

    Tensor4D hidden_;
};