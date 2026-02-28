#include "../include/tensor4d.h"
#include "../include/adam.h"
#include "../include/softmax_ce4d.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib>

// ================================
// byte-level tokenizer
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

int main()
{
    std::cout << "starting T=16 chat model...\n";

    // ------------------------------
    // training corpus
    // ------------------------------
    const char* corpus_text =
        "<USER>こんにちは\n"
        "<ANS>こんにちは。今日はどうしましたか？\n\n"
        "<USER>元気？\n"
        "<ANS>まあ、ぼちぼちです。\n\n"
        "<USER>AIとは？\n"
        "<ANS>確率モデルで、言葉の続きとして返事を生成します。\n";

    std::string corpus_str = corpus_text;
    auto corpus = bytes_to_tokens(corpus_str);

    // ------------------------------
    // vocab
    // ------------------------------
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
    const int T = 16;   // 文脈長
    const int D = 16;   // 埋め込み次元

    std::cout << "vocab = " << V << "\n";

    std::vector<int> tokens;
    for (auto c : corpus)
        tokens.push_back(stoi[c]);

    // ------------------------------
    // model
    // ------------------------------
    Tensor4D emb(1,1,V,D);           // [V,D]
    Tensor4D proj(1,1,D*T,V);        // [T*D, V]

    for (auto& x : emb.data)
        x = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;

    for (auto& x : proj.data)
        x = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;

    Adam optE((int)emb.data.size());
    Adam optP((int)proj.data.size());

    std::cout << "training...\n";

    // ------------------------------
    // training
    // ------------------------------
    for (int epoch = 0; epoch < 300; epoch++)
    {
        float loss_sum = 0.0f;

        for (int i = 0; i + T < (int)tokens.size(); i++)
        {
            if (i % 50 == 0)
            {
                std::cout << "epoch "
                          << epoch
                          << " step "
                          << i
                          << "\r"
                          << std::flush;
            }

            int context[T];
            for (int t = 0; t < T; t++)
                context[t] = tokens[i + t];

            int y = tokens[i + T];

            emb.zero_grad();
            proj.zero_grad();

            std::vector<float> h(D*T, 0.0f);
            std::vector<float> logits(V, 0.0f);

            // forward
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    h[t*D + d] = emb.at(0,0,context[t],d);

            for (int v = 0; v < V; v++)
                for (int d = 0; d < D*T; d++)
                    logits[v] += h[d] * proj.at(0,0,d,v);

            std::vector<float> dlogits;
            float loss = softmax_ce(logits, y, dlogits);
            loss_sum += loss;

            // backward proj
            for (int d = 0; d < D*T; d++)
                for (int v = 0; v < V; v++)
                    proj.grad[proj.idx(0,0,d,v)] += h[d] * dlogits[v];

            // backward emb
            for (int t = 0; t < T; t++)
            {
                int tok = context[t];
                for (int d = 0; d < D; d++)
                {
                    float g = 0.0f;
                    for (int v = 0; v < V; v++)
                        g += proj.at(0,0,t*D + d,v) * dlogits[v];

                    emb.grad[emb.idx(0,0,tok,d)] += g;
                }
            }

            optP.update(proj);
            optE.update(emb);
        }

        std::cout << "\nepoch "
                  << epoch
                  << " loss "
                  << loss_sum
                  << "\n";
    }

    // ------------------------------
    // save weights
    // ------------------------------
    emb.save("weights_emb.bin");
    proj.save("weights_proj.bin");
    std::cout << "weights saved\n";

    // ------------------------------
    // chat
    // ------------------------------
    std::cout << "\n--- chat ---\n";

    while (true)
    {
        std::cout << "> ";
        std::string input;
        std::getline(std::cin, input);
        if (input.empty()) continue;

        std::string prompt =
            "<USER>" + input + "\n<ANS>";

        auto ptoks = bytes_to_tokens(prompt);
        std::vector<uint32_t> output = ptoks;

        while ((int)ptoks.size() < T)
            ptoks.insert(ptoks.begin(), (uint32_t)' ');

        for (int step = 0; step < 200; step++)
        {
            int context[T];
            for (int t = 0; t < T; t++)
                context[t] = stoi[ptoks[ptoks.size() - T + t]];

            std::vector<float> h(D*T, 0.0f);
            std::vector<float> logits(V, 0.0f);

            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    h[t*D + d] = emb.at(0,0,context[t],d);

            for (int v = 0; v < V; v++)
                for (int d = 0; d < D*T; d++)
                    logits[v] += h[d] * proj.at(0,0,d,v);

            int best = 0;
            for (int v = 1; v < V; v++)
                if (logits[v] > logits[best])
                    best = v;

            uint32_t next = itos[best];
            output.push_back(next);
            ptoks.push_back(next);

            if (next == '\n') break;
        }

        std::cout << tokens_to_string(output) << "\n";
    }

    return 0;
}