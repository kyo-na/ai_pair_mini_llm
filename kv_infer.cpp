#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// =====================
// hyper params
// =====================
constexpr int VOCAB     = 10;
constexpr int DIM       = 32;
constexpr int WORLD_DIM = 4;
constexpr int BRANCH    = 5;

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
// World GRU weights (WORLD_DIM only)
// =====================
Tensor Wz(WORLD_DIM), Uz(WORLD_DIM);
Tensor Wr(WORLD_DIM), Ur(WORLD_DIM);
Tensor Wh(WORLD_DIM), Uh(WORLD_DIM);

// =====================
// Critic NN weights (WORLD_DIM → DIM → 1)
// =====================
Tensor Wc1(WORLD_DIM * DIM);
Tensor Wc2(DIM);

// =====================
// reward shaping
// =====================
float reward(int token, int prev) {
    if (token == 3)     return +5.0f;
    if (token == prev) return -0.2f;
    return -1.0f;
}

// =====================
// Critic forward
// =====================
float critic_value(const std::vector<float>& world,
                   std::vector<float>* hidden = nullptr) {
    std::vector<float> h(DIM, 0.f);

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < WORLD_DIM; j++)
            h[i] += Wc1.data[j*DIM + i] * world[j];
        h[i] = std::tanh(h[i]);
    }

    if (hidden) *hidden = h;

    float v = 0.f;
    for (int i = 0; i < DIM; i++)
        v += Wc2.data[i] * h[i];

    return v;
}

// =====================
// Critic backward
// =====================
void critic_backward(const std::vector<float>& world, float td) {
    std::vector<float> h;
    critic_value(world, &h);

    std::vector<float> dh(DIM, 0.f);

    for (int i = 0; i < DIM; i++) {
        Wc2.grad[i] += td * h[i];
        dh[i] = td * Wc2.data[i];
    }

    for (int i = 0; i < DIM; i++) {
        float da = dh[i] * (1.f - h[i] * h[i]);
        for (int j = 0; j < WORLD_DIM; j++)
            Wc1.grad[j*DIM + i] += da * world[j];
    }
}

// =====================
// World GRU step (WORLD_DIM)
// =====================
void world_gru_step(
    const std::vector<float>& world,
    int token,
    std::vector<float>& world_next
) {
    float x = (float)(token - 3);

    for (int i = 0; i < WORLD_DIM; i++) {
        float z = sigmoid(Wz.data[i] * world[i] + Uz.data[i] * x);
        float r = sigmoid(Wr.data[i] * world[i] + Ur.data[i] * x);
        float h = std::tanh(Wh.data[i] * (r * world[i]) + Uh.data[i] * x);
        world_next[i] = (1 - z) * world[i] + z * h;
    }
}

// =====================
// main
// =====================
int main() {
    for (auto* t : {&Wz,&Uz,&Wr,&Ur,&Wh,&Uh,&Wc1,&Wc2}) {
        for (auto& v : t->data)
            v = (frand() - 0.5f) * 0.1f;
    }

    std::vector<float> world(WORLD_DIM, 0.f);
    int token = 0;

    std::cout << "generate:\n";

    for (int step = 0; step < 40; step++) {
        std::vector<int>   cand;
        std::vector<float> adv;

        float v_now = critic_value(world);

        for (int b = 0; b < BRANCH; b++) {
            int t = (frand() < 0.5f)
                ? (token + (frand() < 0.5f ? -1 : 1) + VOCAB) % VOCAB
                : rand() % VOCAB;

            std::vector<float> w2(WORLD_DIM);
            world_gru_step(world, t, w2);

            float r = reward(t, token);
            float v_next = critic_value(w2);
            float A = r + GAMMA * v_next - v_now;

            cand.push_back(t);
            adv.push_back(A);
        }

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

        std::vector<float> world_next(WORLD_DIM);
        world_gru_step(world, next_token, world_next);

        float r = reward(next_token, token);
        float v_next = critic_value(world_next);
        float td = r + GAMMA * v_next - v_now;

        critic_backward(world, td);

        for (auto* t : {&Wc1,&Wc2}) {
            for (int i = 0; i < (int)t->data.size(); i++) {
                t->data[i] += LR * t->grad[i];
                t->grad[i] = 0.f;
            }
        }

        token = next_token;
        world = world_next;

        std::cout
            << "step " << step
            << " token=" << token
            << " V=" << v_next
            << " world=[";
        for (int i = 0; i < WORLD_DIM; i++)
            std::cout << world[i] << (i+1<WORLD_DIM?",":"");
        std::cout << "]\n";
    }
}