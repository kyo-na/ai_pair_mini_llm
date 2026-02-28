#include <iostream>
#include <string>
#include <vector>

#include "../tokenizer/tokenizer.h"
#include "../tokenizer/vocab.h"
#include "../model/embedding.h"
#include "../model/linear.h"

using namespace mini_llm;

int main() {
    std::cout << "infer_llm start\n";

    Tokenizer tokenizer;
    Vocab vocab;

    constexpr int DIM = 64;
    Embedding emb(4096, DIM);
    Linear proj(DIM, DIM);

    std::string input;
    std::cout << ">> ";
    std::getline(std::cin, input);

    auto cps = tokenizer.encode(input);

    std::vector<int> ids;
    for (auto cp : cps)
        ids.push_back(vocab.token_to_id(cp));

    std::cout << "output: ";

    for (int step = 0; step < 32 && !ids.empty(); ++step) {
        int last_id = ids.back();

        // ---- forward ----
        auto x = emb.forward(last_id);
        auto y = proj.forward(x);

        // ---- argmax ----
        int best = 0;
        float best_val = y.data[0];
        for (int i = 1; i < (int)y.data.size(); ++i) {
            if (y.data[i] > best_val) {
                best_val = y.data[i];
                best = i;
            }
        }

        ids.push_back(best);

        // decode 1 token
        std::string out = vocab.id_to_token(best);
        std::cout << out;
    }

    std::cout << "\n";
    return 0;
}