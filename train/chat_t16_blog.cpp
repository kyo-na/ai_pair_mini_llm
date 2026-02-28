#include "../include/tensor4d.h"
#include "../include/adam.h"
#include "../include/softmax_ce4d.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>

// ---------------- config ----------------
static const int T = 16;
static const int D = 16;
static const int EPOCHS = 100;

// generation params
static const float TEMPERATURE = 0.8f;
static const int TOP_K = 8;
static const int MAX_GEN = 200;

// ---------------- tokenizer (byte-level) ----------------
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

// ---------------- util ----------------
static std::string load_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cout << "ERROR: " << path << " not found\n";
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------- sampling ----------------
static int sample_from_logits(
    std::vector<float>& logits,
    const std::vector<int>& recent_tokens)
{
    int V = (int)logits.size();

    // repetition penalty
    for (int v : recent_tokens)
        logits[v] *= 0.7f;

    // temperature
    for (int v = 0; v < V; v++)
        logits[v] /= TEMPERATURE;

    // top-k
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(), idx.begin() + TOP_K, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });

    // softmax
    float maxlog = logits[idx[0]];
    float sum = 0.f;
    std::vector<float> prob(TOP_K);

    for (int i = 0; i < TOP_K; i++) {
        prob[i] = std::exp(logits[idx[i]] - maxlog);
        sum += prob[i];
    }

    float r = ((float)rand() / RAND_MAX) * sum;
    float acc = 0.f;

    for (int i = 0; i < TOP_K; i++) {
        acc += prob[i];
        if (r <= acc)
            return idx[i];
    }
    return idx[0];
}

// ---------------- main ----------------
int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::cout << "loading blog...\n";

    // ---- load blog (LIMITED) ----
    std::string blog = load_file("../../../dataset/blog.txt");
    if (blog.size() > 2000)
        blog = blog.substr(0, 2000);

    std::string corpus =
        "<USER>こんにちは\n"
        "<ANS>こんにちは。今日はどうしましたか？\n\n"
        "<USER>元気？\n"
        "<ANS>まあ、ぼちぼちです。\n\n"
        "<BLOG>\n" + blog + "\n</BLOG>\n";

    auto corpus_bytes = bytes_to_tokens(corpus);

    std::unordered_map<uint32_t,int> stoi;
    std::vector<uint32_t> itos;

    for (auto b : corpus_bytes)
        if (!stoi.count(b)) {
            stoi[b] = (int)itos.size();
            itos.push_back(b);
        }

    int V = (int)itos.size();
    std::cout << "vocab = " << V << "\n";

    std::vector<int> tokens;
    for (auto b : corpus_bytes)
        tokens.push_back(stoi[b]);

    // ---- model ----
    Tensor4D emb(1,1,V,D);
    Tensor4D proj(1,1,D*T,V);

    for (auto& x : emb.data)
        x = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;

    for (auto& x : proj.data)
        x = ((float)rand()/RAND_MAX - 0.5f) * 0.002f;

    Adam optE((int)emb.data.size());
    Adam optP((int)proj.data.size());

    std::cout << "training...\n";

    // ---------------- TRAIN ----------------
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float loss_sum = 0.f;
        int steps = 0;

        for (int i = 0; i + T < (int)tokens.size(); i++)
        {
            int ctx[T];
            for (int t = 0; t < T; t++)
                ctx[t] = tokens[i + t];

            int target = tokens[i + T];

            emb.zero_grad();
            proj.zero_grad();

            std::vector<float> h(D*T, 0.f);
            std::vector<float> logits(V, 0.f);

            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    h[t*D + d] = emb.at(0,0,ctx[t],d);

            for (int v = 0; v < V; v++)
                for (int d = 0; d < D*T; d++)
                    logits[v] += h[d] * proj.at(0,0,d,v);

            for (int v = 0; v < V; v++) {
                if (logits[v] > 20.f) logits[v] = 20.f;
                if (logits[v] < -20.f) logits[v] = -20.f;
            }

            std::vector<float> dlogits;
            float loss = softmax_ce(logits, target, dlogits);
            loss_sum += loss;
            steps++;

            for (int d = 0; d < D*T; d++)
                for (int v = 0; v < V; v++)
                    proj.grad[proj.idx(0,0,d,v)] += h[d] * dlogits[v];

            for (int t = 0; t < T; t++)
            {
                int tok = ctx[t];
                for (int d = 0; d < D; d++)
                {
                    float g = 0.f;
                    for (int v = 0; v < V; v++)
                        g += proj.at(0,0,t*D + d,v) * dlogits[v];
                    emb.grad[emb.idx(0,0,tok,d)] += g;
                }
            }

            optP.update(proj);
            optE.update(emb);
        }

        std::cout << "epoch " << epoch
                  << " loss " << (loss_sum / steps) << "\n";
    }

    emb.save("weights_emb.bin");
    proj.save("weights_proj.bin");
    std::cout << "weights saved\n";

    // ---------------- CHAT ----------------
    std::cout << "\n--- chat ---\n";

    while (true)
    {
        std::cout << "> ";
        std::string input;
        if (!std::getline(std::cin, input)) break;

        std::string prompt = "<USER>" + input + "\n<ANS>";
        auto pbytes = bytes_to_tokens(prompt);

        std::vector<uint32_t> ctx;
        for (auto b : pbytes) ctx.push_back(b);

        while ((int)ctx.size() < T)
            ctx.insert(ctx.begin(), (uint32_t)' ');

        std::vector<uint32_t> out;
        std::vector<int> recent;

        for (int step = 0; step < MAX_GEN; step++)
        {
            int c[T];
            for (int t = 0; t < T; t++)
                c[t] = stoi[ctx[ctx.size() - T + t]];

            std::vector<float> h(D*T, 0.f);
            std::vector<float> logits(V, 0.f);

            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    h[t*D + d] = emb.at(0,0,c[t],d);

            for (int v = 0; v < V; v++)
                for (int d = 0; d < D*T; d++)
                    logits[v] += h[d] * proj.at(0,0,d,v);

            int next = sample_from_logits(logits, recent);
            recent.push_back(next);
            if (recent.size() > 8) recent.erase(recent.begin());

            uint32_t ch = itos[next];
            out.push_back(ch);
            ctx.push_back(ch);

            if (ch == '\n') break;
        }

        std::cout << tokens_to_string(out) << "\n";
    }

    return 0;
}