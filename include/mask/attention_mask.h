#pragma once
#include <vector>

inline void apply_attention_mask(
    std::vector<float>& scores,
    size_t t,
    const std::vector<unsigned>& tokens,
    unsigned pad_id
) {
    for (size_t j = 0; j < scores.size(); ++j) {
        // causal
        if (j > t) {
            scores[j] = -1e9f;
            continue;
        }
        // padding
        if (j < tokens.size() && tokens[j] == pad_id) {
            scores[j] = -1e9f;
        }
    }
}