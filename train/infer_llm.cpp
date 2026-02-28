#include <iostream>
#include <vector>
#include <string>

#include "../include/tokenizer/tokenizer.h"
#include "../include/tokenizer/vocab.h"
#include "../include/model/embedding.h"
#include "../include/model/linear.h"

using namespace mini_llm;

int argmax(const std::vector<float>& v) {
    int idx = 0;
    float best = v[0];
    for (int i = 1; i < (int)v.size(); i++) {
        if (v[i] > best) {
            best = v[i];
            idx = i;
        }
    }
    return idx;
}

int main() {
    std::cout << "mini_llm chat start\n";

    Tokenizer tokenizer;
    Vocab vocab;

    constexpr int DIM = 64;
    Embedding emb(4096, DIM);
    Linear proj(DIM, DIM);

    // ★ 重みロード（必須）
    emb.weight.load("weights_emb.bin");
    proj.weight.load("weights_proj.bin");

    std::string input;
    std::cout << ">> ";
    std::getline(std::cin, input);

    auto cps = tokenizer.encode(input);
    std::vector<int> ids;
    for (auto cp : cps)
        ids.push_back(vocab.token_to_id(cp));

    int cur = ids.back();

    for (int step = 0; step < 50; step++) {
        auto x = emb.forward(cur);
        auto y = proj.forward(x);

        int next = argmax(y);
        cur = next;

        char32_t cp = vocab.id_to_token(cur);
        std::string out = utf8_encode(cp);
        std::cout << out;
    }

    std::cout << "\n";
    return 0;
}