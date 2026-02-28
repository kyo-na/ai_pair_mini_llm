#pragma once
#include <vector>

struct CrossEntropyVocab {
    int V;
    std::vector<float> probs;   // softmax結果保存
    std::vector<int> targets;

    CrossEntropyVocab(int V);

    float forward(
        const std::vector<float>& logits, // (B*T*V)
        const std::vector<int>& target,   // (B*T)
        int B,int T
    );

    void backward(
        std::vector<float>& dlogits // (B*T*V)
    );
};