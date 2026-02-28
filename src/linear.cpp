#include "../include/linear.h"

std::vector<float> Linear::forward(const std::vector<float>& x){
    std::vector<float> y(b);
    for(int o=0;o<W.cols;o++)
        for(int i=0;i<W.rows;i++)
            y[o]+=x[i]*W.at(i,o);
    return y;
}