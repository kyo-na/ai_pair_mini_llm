#include "decode/decode_step.h"
#include "decode/softmax.h"
#include "decode/sampling.h"

uint32_t decode_next_token(
    std::vector<float> logits,
    const std::vector<uint32_t>& history,
    const DecodeConfig& cfg
){
    // 1) repetition penalty（logitsへ）
    apply_repetition_penalty_logits(logits, history, cfg.rep);

    // 2) softmax
    std::vector<float> probs;
    Softmax sm;
    sm.apply(logits, probs);

    // 3) sampling
    int tok = sample_from_probs(probs, cfg.temperature, cfg.top_k, cfg.top_p);
    return (uint32_t)tok;
}