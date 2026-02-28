#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "tensor.h"

// =====================
// hyper params
// =====================
constexpr int VOCAB = 10;
constexpr int DIM   = 32;
constexpr int SEQ   = 3;

constexpr float LR    = 1e-2f;
constexpr float BETA1 = 0.9f;
constexpr float BETA2 = 0.999f;
constexpr float EPS   = 1e-8f;

// =====================
float randf() {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    return dist(rng);
}

// =====================
// Adam update
// =====================
void adam_update(Tensor& t, int step) {
    float b1t = 1.0f - std::pow(BETA1, step);
    float b2t = 1.0f - std::pow(BETA2, step);

    for (int i = 0; i < t.n; i++) {
        t.m[i] = BETA1 * t.m[i] + (1 - BETA1) * t.grad[i];
        t.v[i] = BETA2 * t.v[i] + (1 - BETA2) * t.grad[i] * t.grad[i];

        float mh = t.m[i] / b1t;
        float vh = t.v[i] / b2t;

        t.data[i] -= LR * mh / (std::sqrt(vh) + EPS);
    }
}

// =====================
// softmax + CE
// =====================
float softmax_ce(const std::vector<float>& logits, int tgt,
                 std::vector<float>& dlogits) {
    float m = *std::max_element(logits.begin(), logits.end());
    float s = 0;
    for (float v : logits) s += std::exp(v - m);

    float loss = 0;
    for (int i = 0; i < VOCAB; i++) {
        float p = std::exp(logits[i] - m) / s;
        dlogits[i] = p;
        if (i == tgt) {
            loss = -std::log(p + 1e-9f);
            dlogits[i] -= 1.0f;
        }
    }
    return loss;
}

// =====================
// main (training only)
// =====================
int main() {
    // ===== parameters =====
    Tensor embed(VOCAB * DIM);
    Tensor Wq(DIM * DIM), Wk(DIM * DIM), Wv(DIM * DIM);
    Tensor W1(DIM * DIM), W2(DIM * DIM);
    Tensor out(DIM * VOCAB);

    for (auto* t : {&embed,&Wq,&Wk,&Wv,&W1,&W2,&out})
        for (auto& v : t->data) v = randf();

    // ===== toy copy data =====
    std::vector<int> data = {1,2,3,1,2,3,1,2,3};

    int step = 1;

    // ===== training loop =====
    for (int epoch = 0; epoch < 400; epoch++) {
        float total_loss = 0;

        for (auto* t : {&embed,&Wq,&Wk,&Wv,&W1,&W2,&out})
            t->zero_grad();

        for (int i = 0; i + SEQ < (int)data.size(); i++) {
            // ----- embedding -----
            float X[SEQ][DIM];
            for (int t = 0; t < SEQ; t++)
                for (int d = 0; d < DIM; d++)
                    X[t][d] = embed.data[data[i+t]*DIM + d];

            // ----- Q K V -----
            float Q[SEQ][DIM]={}, K[SEQ][DIM]={}, V[SEQ][DIM]={};
            for (int t = 0; t < SEQ; t++)
                for (int d = 0; d < DIM; d++)
                    for (int k = 0; k < DIM; k++) {
                        Q[t][d] += X[t][k] * Wq.data[k*DIM+d];
                        K[t][d] += X[t][k] * Wk.data[k*DIM+d];
                        V[t][d] += X[t][k] * Wv.data[k*DIM+d];
                    }

            // ----- Attention (last token) -----
            float score[SEQ];
            for (int j = 0; j < SEQ; j++) {
                score[j] = 0;
                for (int d = 0; d < DIM; d++)
                    score[j] += Q[SEQ-1][d] * K[j][d];
                score[j] /= std::sqrt((float)DIM);
            }

            float sm = *std::max_element(score, score+SEQ);
            float ss = 0;
            float alpha[SEQ];
            for (int j = 0; j < SEQ; j++) {
                alpha[j] = std::exp(score[j]-sm);
                ss += alpha[j];
            }
            for (int j = 0; j < SEQ; j++) alpha[j] /= ss;

            float A[DIM]={};
            for (int j = 0; j < SEQ; j++)
                for (int d = 0; d < DIM; d++)
                    A[d] += alpha[j] * V[j][d];

            // ----- FFN -----
            float pre[DIM]={}, act[DIM]={}, H[DIM]={};
            for (int j = 0; j < DIM; j++) {
                for (int k = 0; k < DIM; k++)
                    pre[j] += A[k] * W1.data[k*DIM+j];
                act[j] = pre[j] > 0 ? pre[j] : 0;
            }
            for (int d = 0; d < DIM; d++)
                for (int j = 0; j < DIM; j++)
                    H[d] += act[j] * W2.data[j*DIM+d];

            // ----- output -----
            std::vector<float> logits(VOCAB,0);
            for (int v = 0; v < VOCAB; v++)
                for (int d = 0; d < DIM; d++)
                    logits[v] += H[d] * out.data[d*VOCAB+v];

            std::vector<float> dlogits(VOCAB);
            total_loss += softmax_ce(logits, data[i+SEQ], dlogits);

            // ----- backward (output) -----
            for (int d = 0; d < DIM; d++)
                for (int v = 0; v < VOCAB; v++)
                    out.grad[d*VOCAB+v] += H[d] * dlogits[v];

            float dH[DIM]={};
            for (int d = 0; d < DIM; d++)
                for (int v = 0; v < VOCAB; v++)
                    dH[d] += out.data[d*VOCAB+v] * dlogits[v];

            // ----- FFN backward -----
            float dact[DIM]={};
            for (int j = 0; j < DIM; j++) {
                for (int d = 0; d < DIM; d++)
                    W2.grad[j*DIM+d] += act[j] * dH[d];
                for (int d = 0; d < DIM; d++)
                    dact[j] += W2.data[j*DIM+d] * dH[d];
            }

            float dpre[DIM]={};
            for (int j = 0; j < DIM; j++)
                dpre[j] = pre[j] > 0 ? dact[j] : 0;

            float dA[DIM]={};
            for (int k = 0; k < DIM; k++)
                for (int j = 0; j < DIM; j++) {
                    W1.grad[k*DIM+j] += A[k] * dpre[j];
                    dA[k] += W1.data[k*DIM+j] * dpre[j];
                }

            // ----- V backward -----
            for (int j = 0; j < SEQ; j++)
                for (int d = 0; d < DIM; d++)
                    Wv.grad[d*DIM + d] += alpha[j] * dA[d];
        }

        for (auto* t : {&embed,&Wq,&Wk,&Wv,&W1,&W2,&out})
            adam_update(*t, step++);
        
        if (epoch % 20 == 0)
            std::cout << "epoch " << epoch << " loss=" << total_loss << "\n";
    }

    // ===== save weights =====
    save_tensor_data("embed.bin", embed);
    save_tensor_data("Wq.bin", Wq);
    save_tensor_data("Wk.bin", Wk);
    save_tensor_data("Wv.bin", Wv);
    save_tensor_data("W1.bin", W1);
    save_tensor_data("W2.bin", W2);
    save_tensor_data("out.bin", out);

    std::cout << "saved weights (*.bin)\n";
}