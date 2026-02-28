#include "../../include/model/mini_AI.h"
#include <fstream>
#include <cmath>

static uint32_t argmax(const std::vector<float>& v) {
    uint32_t m = 0;
    for (uint32_t i = 1; i < v.size(); ++i)
        if (v[i] > v[m]) m = i;
    return m;
}

MiniAI::MiniAI(int vocab, int dim)
: emb(vocab, dim), proj(dim, vocab) {

    std::ifstream f1("engine/mini_llm/train/weights_emb.bin", std::ios::binary);
    f1.read((char*)emb.W.data(), emb.W.size()*sizeof(float));

    std::ifstream f2("engine/mini_llm/train/weights_proj.bin", std::ios::binary);
    f2.read((char*)proj.W.data(), proj.W.size()*sizeof(float));
}

uint32_t MiniAI::forward_token(uint32_t t) {
    auto h = emb.forward(t);
    auto logits = proj.forward(h);
    return argmax(logits);
}