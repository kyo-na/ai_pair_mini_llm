#pragma once
#include "optimizer/adam.h"

struct OptimizerContext {
    float lr=1e-3f;
    int step=0;

    void update(Tensor4D& t){
        step++;
        adam_update(t,lr,step);
    }

    void update_vec(
        std::vector<float>& w,
        const std::vector<float>& g,
        std::vector<float>& m,
        std::vector<float>& v
    ){
        step++;
        adam_update_vec(w,g,m,v,lr,step);
    }
};