#include "../include/tensor4d.h"
#include "../include/adam.h"
#include "../include/softmax_ce4d.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib>

// ================================
// byte-level tokenizer (UTF-8安全)
// ================================
static std::vector<uint32_t> bytes_to_tokens(const std::string& s)
{
    std::vector<uint32_t> out;
    for (unsigned char c : s)
        out.push_back((uint32_t)c);
    return out;
}

static std::string tokens_to_string(const std::vector<uint32_t>& tokens)
{
    std::string out;
    for (auto t : tokens)
        out.push_back((char)t);
    return out;
}

// ================================
// main
// ================================
int main()
{
    std::cout << "loading corpus...\n";

    // ---- load corpus.txt ----
    std::ifstream file("../dataset/corpus.txt", std::ios::binary);
    if (!file)
    {
        std::cerr << "ERROR: dataset/corpus.txt not found\n";
        return 1;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    std::string text = ss.str();

    auto corpus = bytes_to_tokens(text);

    // ---- vocab build ----
    std::unordered_map<uint32_t,int> stoi;
    std::vector<uint32_t> itos;

    for (auto c : corpus)
    {
        if (stoi.find(c) == stoi.end())
        {
            stoi[c] = (int)itos.size();
            itos.push_back(c);
        }
    }

    int V = (int)itos.size();
    int D = 32; // 少し大きめに

    std::cout << "vocab size = " << V << "\n";

    std::vector<int> tokens;
    tokens.reserve(corpus.size());
    for (auto c : corpus)
        tokens.push_back(stoi[c]);

    // ---- model ----
    Tensor4D emb(1,1,V,D);
    Tensor4D proj(1,1,D,V);

    for (auto& x : emb.data)
        x = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;

    for (auto& x : proj.data)
        x = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;

    Adam optE((int)emb.data.size());
    Adam optP((int)proj.data.size());

    std::cout << "training...\n";

    // ---- training ----
    for (int epoch = 0; epoch < 3000; epoch++)
    {
        float loss_sum = 0.0f;

        for (int i = 0; i + 1 < (int)tokens.size(); i++)
        {
            int x = tokens[i];
            int y = tokens[i + 1];

            emb.zero_grad();
            proj.zero_grad();

            std::vector<float> h(D, 0.0f);
            std::vector<float> logits(V, 0.0f);

            // forward
            for (int d = 0; d < D; d++)
                h[d] = emb.at(0,0,x,d);

            for (int v = 0; v < V; v++)
                for (int d = 0; d < D; d++)
                    logits[v] += h[d] * proj.at(0,0,d,v);

            std::vector<float> dlogits;
            float loss = softmax_ce(logits, y, dlogits);
            loss_sum += loss;

            // backward proj
            for (int d = 0; d < D; d++)
                for (int v = 0; v < V; v++)
                    proj.grad[proj.idx(0,0,d,v)] += h[d] * dlogits[v];

            // backward emb
            for (int d = 0; d < D; d++)
            {
                float g = 0.0f;
                for (int v = 0; v < V; v++)
                    g += proj.at(0,0,d,v) * dlogits[v];
                emb.grad[emb.idx(0,0,x,d)] += g;
            }

            optP.update(proj);
            optE.update(emb);
        }

        if (epoch % 300 == 0)
        {
            std::cout << "epoch " << epoch
                      << " loss " << loss_sum << "\n";
        }
    }

    std::cout << "\n--- generate ---\n";

    // ---- generation ----
    while (true)
    {
        std::cout << "> ";
        std::string prompt;
        std::getline(std::cin, prompt);

        if (prompt.empty())
            continue;

        auto prompt_tokens = bytes_to_tokens(prompt);
        std::vector<uint32_t> output = prompt_tokens;

        for (int step = 0; step < 300; step++)
        {
            uint32_t last = prompt_tokens.back();
            if (stoi.find(last) == stoi.end())
                break;

            int x = stoi[last];

            std::vector<float> h(D, 0.0f);
            std::vector<float> logits(V, 0.0f);

            for (int d = 0; d < D; d++)
                h[d] = emb.at(0,0,x,d);

            for (int v = 0; v < V; v++)
                for (int d = 0; d < D; d++)
                    logits[v] += h[d] * proj.at(0,0,d,v);

            int best = 0;
            for (int v = 1; v < V; v++)
                if (logits[v] > logits[best])
                    best = v;

            uint32_t next = itos[best];

            output.push_back(next);
            prompt_tokens.push_back(next);

            if (next == '\n')
                break;
        }

        std::cout << tokens_to_string(output) << "\n";
    }

    return 0;
}