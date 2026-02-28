#pragma once
#include <vector>
#include "model/transformer_stack_infer.h"
#include "layers/embedding4d.h"
#include "decode/vocab_projection.h"

class InferenceEngine {
public:
    InferenceEngine(
        int vocab,
        int layers,
        int heads,
        int head_dim,
        int ffn_hidden,
        int max_seq_len,
        float temperature = 1.0f,
        int top_k = 0,
        float top_p = 0.0f);

    void reset();
    int step(int token_id);

private:
    float temperature_;
    int top_k_;
    float top_p_;

    Embedding4D embedding_;
    TransformerStackInfer stack_;
    VocabProjection projection_;

    int sample(std::vector<float>& probs);
};