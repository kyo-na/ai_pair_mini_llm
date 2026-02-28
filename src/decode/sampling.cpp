#include "decode/sampling.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

int sample_token(
    std::vector<float> probs,
    float temperature,
    int top_k,
    float top_p
){
    static std::mt19937 rng(1234);

    if(temperature!=1.0f){
        for(float& p:probs)
            p=std::pow(p,1.0f/temperature);
    }

    if(top_k>0){
        std::vector<int> idx(probs.size());
        std::iota(idx.begin(),idx.end(),0);

        std::partial_sort(
            idx.begin(), idx.begin()+top_k, idx.end(),
            [&](int a,int b){ return probs[a]>probs[b]; }
        );

        std::vector<float> newp(probs.size(),0.0f);
        for(int i=0;i<top_k;++i)
            newp[idx[i]]=probs[idx[i]];
        probs.swap(newp);
    }

    if(top_p<1.0f){
        std::vector<int> idx(probs.size());
        std::iota(idx.begin(),idx.end(),0);

        std::sort(
            idx.begin(), idx.end(),
            [&](int a,int b){ return probs[a]>probs[b]; }
        );

        float cum=0.0f;
        for(size_t i=0;i<idx.size();++i){
            cum+=probs[idx[i]];
            if(cum>top_p){
                for(size_t j=i+1;j<idx.size();++j)
                    probs[idx[j]]=0.0f;
                break;
            }
        }
    }

    std::discrete_distribution<int> dist(probs.begin(),probs.end());
    return dist(rng);
}