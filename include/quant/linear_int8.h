#pragma once
#include <cstdint>
#include <cmath>

class LinearINT8 {
public:
    LinearINT8(int in,int out);

    void quantize(const float* w);
    void forward(const float* x,
                 float* y);

private:
    int in_,out_;
    int8_t* w_q_;
    float* scale_;
};