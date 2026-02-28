#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "tokenizer/tokenizer.h"
#include "tokenizer/vocab.h"
#include "model/embedding.h"
#include "model/linear.h"
#include "loss/mse4d.h"
#include "optimizer/adam.h"

using namespace mini_llm;

int main() {
    std::cout << "train_llm start\n";

    Tokenizer tokenizer;
    Vocab vocab;

    constexpr int DIM = 64;
    Embedding emb(4096, DIM);
    Linear proj(DIM, DIM);

    std::ifstream fin("../data/shard_000.txt");
    if (!fin) {
        std::cerr << "failed to open ../data/shard_000.txt\n";
        return 1;
    }

    std::string line;
    float total_loss = 0.0f;
    int steps = 0;

    while (std::getline(fin, line)) {
        auto cps = tokenizer.encode(line);

        std::vector<int> ids;
        for (auto cp : cps)
            ids.push_back(vocab.token_to_id(cp));

        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            auto x = emb.forward(ids[i]);
            auto y = proj.forward(x);
            auto t = emb.forward(ids[i + 1]);

            float loss = mse_loss(y, t);
            total_loss += loss;
            steps++;

            auto dy = mse_grad(y, t);
            auto dx = proj.backward(dy);
            emb.backward(ids[i], dx);

            emb.step(0.05f);
            proj.step(0.05f);
        }
    }

    std::cout << "avg_loss=" << (total_loss / steps) << "\n";
    return 0;
}