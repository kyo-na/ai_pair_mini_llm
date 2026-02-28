#pragma once
#include <vector>
#include "tensor4d.h"

class AdamW
{
public:
    AdamW(float lr);
    void step(std::vector<Tensor4D*>& params);

private:
    float lr_;
};