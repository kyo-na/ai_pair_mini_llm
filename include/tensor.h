#pragma once
#include <vector>
#include <cstdint>

struct Tensor {
    int rows, cols;
    std::vector<float> data;

    Tensor(int r=0,int c=0):rows(r),cols(c),data(r*c){}

    float& operator()(int r,int c){
        return data[r*cols+c];
    }
};