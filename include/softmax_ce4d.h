#pragma once
#include <vector>
#include <cmath>

inline float softmax_ce(
    const std::vector<float>& logits,
    int target,
    std::vector<float>& dlogits
){
    int V = (int)logits.size();
    dlogits.resize(V);

    float maxv = logits[0];
    for(int i = 1; i < V; i++)
        if(logits[i] > maxv) maxv = logits[i];

    float sum = 0.0f;
    for(int i = 0; i < V; i++){
        dlogits[i] = std::exp(logits[i] - maxv);
        sum += dlogits[i];
    }

    float loss = -std::log(dlogits[target] / sum);

    for(int i = 0; i < V; i++){
        dlogits[i] /= sum;
        if(i == target) dlogits[i] -= 1.0f;
    }

    return loss;
}