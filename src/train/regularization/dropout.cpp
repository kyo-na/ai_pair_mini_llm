#include "train/dropout.h"
#include <random>

void apply_dropout(
    Tensor4D& x,
    float p,
    bool train
){
    if(!train || p<=0.0f) return;

    static std::mt19937 rng(1);
    std::bernoulli_distribution keep(1.0-p);

    for(size_t i=0;i<x.data.size();++i){
        if(!keep(rng)) x.data[i]=0.0f;
        else x.data[i]/=(1.0f-p);
    }
}