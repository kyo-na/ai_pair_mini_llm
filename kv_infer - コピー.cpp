#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// =====================
// hyper params
// =====================
constexpr int VOCAB  = 10;
constexpr int DIM    = 32;
constexpr int BRANCH = 5;

constexpr float GAMMA = 0.95f;
constexpr float LR    = 0.01f;

// =====================
// util
// =====================
float frand() {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}

float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

// =====================
// simple tensor
// =====================
struct Tensor {
    std::vector<float> data, grad;
    Tensor(int n=0) : data(n), grad(n) {}
};

// =====================
// World GRU weights
// =====================
Tensor Wz(DIM), Uz(DIM);
Tensor Wr(DIM), Ur(DIM);
Tensor Wh(DIM), Uh(DIM);

// =====================
// Critic NN weights (DIM → DIM → 1)
// =====================
Tensor Wc1(DIM * DIM);   // hidden
Tensor Wc2(DIM);         // output

// =====================
// reward shaping（強化）
// =====================
float reward(int token, int prev) {
    if (token == 3)     return +5.0f;   // 目標
    if (token == prev) return -0.2f;   // 停滞ペナルティ
    return -1.0f;                      // それ以外
}

// =====================
// Critic forward
// =====================
float critic_value(const std::vector<float>& world,
                   std::vector<float>* hidden = nullptr) {
    std::vector<float> h(DIM, 0.f);

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++)
            h[i] += Wc1.data[j*DIM + i] * world[j];
        h[i] = std::tanh(h[i]);
    }

    if (hidden) *hidden = h;

    float v = 0.f;
    for (int i = 0; i < DIM; i++)
        v += Wc2.data[i] * h[i];

    return v;   // ★ tanh しない（実数値）
}

// =====================
// Critic backward (TD + 正しい連鎖律)
// =====================
void critic_backward(const std::vector<float>& world, float td) {
    std::vector<float> h;
    float v = critic_value(world, &h);

    // dL/dv = td
    std::vector<float> dh(DIM, 0.f);

    for (int i = 0; i < DIM; i++) {
        Wc2.grad[i] += td * h[i];
        dh[i] = td * Wc2.data[i];
    }

    for (int i = 0; i < DIM; i++) {
        float da = dh[i] * (1.f - h[i] * h[i]);
        for (int j = 0; j < DIM; j++)
            Wc1.grad[j*DIM + i] += da * world[j];
    }
}

// =====================
// World GRU step
// =====================
void world_gru_step(
    const std::vector<float>& world,
    int token,
    std::vector<float>& world_next
) {
    for (int i = 0; i < DIM; i++) {
        float z = sigmoid(Wz.data[i] * world[i] + Uz.data[i] * token);
        float r = sigmoid(Wr.data[i] * world[i] + Ur.data[i] * token);
        float h = std::tanh(Wh.data[i] * (r * world[i]) + Uh.data[i] * token);
        world_next[i] = (1 - z) * world[i] + z * h;
    }
}

// =====================
// main
// =====================
int main() {
    // init weights
    for (auto* t : {&Wz,&Uz,&Wr,&Ur,&Wh,&Uh,&Wc1,&Wc2}) {
        for (auto& v : t->data)
            v = (frand() - 0.5f) * 0.1f;
    }

    std::vector<float> world(DIM, 0.f);
    int token = 0;
    int prev  = 0;

    std::cout << "generate:\n";

    for (int step = 0; step < 40; step++) {
        std::vector<int>   cand;
        std::vector<float> adv;

        float v_now = critic_value(world);

        // ===== branch & advantage =====
        for (int b = 0; b < BRANCH; b++) {
            int t = (frand() < 0.5f)
                ? (token + (frand() < 0.5f ? -1 : 1) + VOCAB) % VOCAB
                : rand() % VOCAB;

            std::vector<float> w2(DIM);
            world_gru_step(world, t, w2);

            float r = reward(t, token);
            float v_next = critic_value(w2);

            float A = r + GAMMA * v_next - v_now;

            cand.push_back(t);
            adv.push_back(A);
        }

        // ===== softmax select =====
        float m = *std::max_element(adv.begin(), adv.end());
        float sum = 0.f;
        for (float& a : adv) {
            a = std::exp(a - m);
            sum += a;
        }

        float p = frand() * sum;
        int sel = 0;
        for (; sel < BRANCH; sel++) {
            p -= adv[sel];
            if (p <= 0) break;
        }
        if (sel >= BRANCH) sel = BRANCH - 1;

        int next_token = cand[sel];

        // ===== world update =====
        std::vector<float> world_next(DIM);
        world_gru_step(world, next_token, world_next);

        // ===== TD learning =====
        float r = reward(next_token, token);
        float v_next = critic_value(world_next);
        float td = r + GAMMA * v_next - v_now;

        critic_backward(world, td);

        // SGD update
        for (auto* t : {&Wc1,&Wc2}) {
            for (int i = 0; i < (int)t->data.size(); i++) {
                t->data[i] += LR * t->grad[i];
                t->grad[i] = 0.f;
            }
        }

        prev  = token;
        token = next_token;
        world = world_next;

        std::cout
            << "step " << step
            << " token=" << token
            << " V=" << v_next
            << " world[0]=" << world[0]
            << "\n";
    }
}