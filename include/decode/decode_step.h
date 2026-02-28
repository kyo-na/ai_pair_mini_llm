#pragma once
#include <vector>
#include <cstdint>
#include "decode/repetition_penalty.h"

struct DecodeConfig {
    // sampling側
    float temperature = 1.0f;
    int   top_k = 0;
    float top_p = 1.0f;

    // repetition penalty側
    RepetitionPenaltyConfig rep;
};

// logits（softmax前）から次tokenを返す
uint32_t decode_next_token(
    std::vector<float> logits,
    const std::vector<uint32_t>& history,
    const DecodeConfig& cfg
);