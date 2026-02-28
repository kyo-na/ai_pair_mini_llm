#include "decode/softmax.h"
#include <cmath>
#include <algorithm>

void Softmax::apply(const std::vector<float>& logits,
                    std::vector<float>& probs) {
    probs.resize(logits.size());

    float maxv = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;

    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxv);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;
}