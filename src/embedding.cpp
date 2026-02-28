#include "../include/embedding.h"

std::vector<float> Embedding::forward(int token){
    std::vector<float> v(W.cols);
    for(int i=0;i<W.cols;i++)
        v[i]=W.at(token,i);
    return v;
}