#pragma once
#include <vector>
#include "quant/int8_quant.h"

class LinearINT8 {
public:
    LinearINT8(int in, int out);

    // float重みを渡して量子化（out-major: w[o*in+i]）
    void quantize_from_float(const std::vector<float>& w);

    // x: [in] -> y: [out]
    void forward_vec(const float* x, float* y) const;

    int in() const { return in_; }
    int out() const { return out_; }

private:
    int in_, out_;
    Int8PerChannel q_;
};