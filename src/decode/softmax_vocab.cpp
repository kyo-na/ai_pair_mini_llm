#include "decode/softmax_vocab.h"
#include <cmath>
#include <algorithm>

void softmax_vocab(
    const std::vector<float>& logits,
    std::vector<float>& probs,
    int B,int T,int V
){
    probs.resize(B*T*V);

    for(int i=0;i<B*T;++i){
        float maxv = -1e9f;

        for(int v=0;v<V;++v)
            maxv = std::max(maxv, logits[i*V+v]);

        float sum=0.0f;

        for(int v=0;v<V;++v){
            float e = std::exp(logits[i*V+v]-maxv);
            probs[i*V+v]=e;
            sum+=e;
        }

        for(int v=0;v<V;++v)
            probs[i*V+v]/=sum;
    }
}