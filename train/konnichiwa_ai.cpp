#include "../include/tensor4d.h"
#include "../include/adam.h"
#include "../include/softmax_ce4d.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib>

// ================================
// byte-level tokenizer (安全版)
// ================================

static std::vector<uint32_t> utf8_bytes_to_tokens(const char* s)
{
    std::vector<uint32_t> out;
    const unsigned char* p = (const unsigned char*)s;
    while (*p)
    {
        out.push_back((uint32_t)(*p));
        ++p;
    }
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
// メイン
// ================================

int main()
{
    std::cout << "starting...\n";

    // ---- dataset (u8必須) ----
    const char* corpus_utf8 =
        u8"こんにちは\nこんにちは！\n"
        u8"元気？\n元気です\n";

    auto corpus = utf8_bytes_to_tokens(corpus_utf8);

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
    int D = 16;

    std::vector<int> tokens;
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
    for (int epoch = 0; epoch < 2000; epoch++)
    {
        float loss_sum = 0.0f;

        for (int i = 0; i + 1 < tokens.size(); i++)
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

        if (epoch % 200 == 0)
            std::cout << "epoch " << epoch
                      << " loss " << loss_sum << "\n";
    }

    std::cout << "\n--- chat ---\n";

    // ---- chat ----
    while (true)
    {
        std::cout << "> ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty())
            continue;

        auto input_tokens = utf8_bytes_to_tokens(input.c_str());
        std::vector<uint32_t> output_tokens = input_tokens;

        for (int step = 0; step < 32; step++)
        {
            uint32_t last = input_tokens.back();

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

            output_tokens.push_back(next);
            input_tokens.push_back(next);

            if (next == '\n')
                break;
        }

        std::cout << tokens_to_string(output_tokens) << "\n";
    }

    return 0;
}