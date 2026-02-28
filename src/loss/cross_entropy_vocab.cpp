#include "loss/cross_entropy_vocab.h"
#include <cmath>

CrossEntropyVocab::CrossEntropyVocab(int V_) : V(V_) {}

float CrossEntropyVocab::forward(
    const std::vector<float>& logits,
    const std::vector<int>& target,
    int B,int T
){
    targets = target;
    probs.resize(B*T*V);

    float loss = 0.0f;

    for(int i=0;i<B*T;++i){
        // softmax
        float maxv = -1e9f;
        for(int v=0;v<V;++v)
            maxv = std::max(maxv, logits[i*V+v]);

        float sum = 0.0f;
        for(int v=0;v<V;++v){
            probs[i*V+v] = std::exp(logits[i*V+v]-maxv);
            sum += probs[i*V+v];
        }
        for(int v=0;v<V;++v)
            probs[i*V+v] /= sum;

        int y = target[i];
        loss -= std::log(probs[i*V + y] + 1e-9f);
    }
    return loss / (B*T);
}

void CrossEntropyVocab::backward(
    std::vector<float>& dlogits
){
    int N = targets.size();
    dlogits = probs;

    for(int i=0;i<N;++i){
        dlogits[i*V + targets[i]] -= 1.0f;
    }
}