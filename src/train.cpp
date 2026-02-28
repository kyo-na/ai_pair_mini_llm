#include "model/transformer_stack4d.h"
#include "layers/embedding4d.h"
#include "layers/linear_vocab4d.h"
#include "loss/cross_entropy4d.h"
#include "optimizer/adamw.h"

#include <iostream>
#include <vector>

int main()
{
    std::cout << "train start\n";

    // ---- Model config ----
    int layers = 1;          // まずは1層で安定確認
    int heads = 1;           // まずは1 head
    int head_dim = 8;
    int ffn_hidden = 16;
    int vocab = 6;

    TransformerStack4D model(layers, heads, head_dim, ffn_hidden);
    Embedding4D embed(heads, head_dim, vocab);
    LinearVocab4D linear(heads, head_dim, vocab);

    // ★ 学習率上げる
    AdamW optimizer(0.1f);

    // ---- Toy dataset ----
    std::vector<int> input_ids  = {1,2,3,4};
    std::vector<int> target_ids = {2,3,4,5};

    int B = 1;
    int T = (int)input_ids.size();
    int V = vocab;

    int max_epoch = 300;

    for(int epoch=0; epoch<max_epoch; ++epoch)
    {
        // ===== Forward =====
        Tensor4D x = embed.forward(input_ids);   // (1,T,1,D)
        Tensor4D h = model.forward(x);           // (1,T,1,D)
        Tensor4D logits = linear.forward(h);     // (1,T,1,V)

        // ---- One-hot target ----
        Tensor4D target(1,T,1,V);
        for(int t=0; t<T; ++t)
            target.at(0,t,0,target_ids[t]) = 1.0f;

        CrossEntropy4D loss_fn;
        float loss = loss_fn.forward(logits, target);

        if(epoch % 10 == 0)
        {
            std::cout << "epoch "
                      << epoch
                      << " loss="
                      << loss
                      << "\n";
        }

        // ===== Backward =====
        Tensor4D grad = loss_fn.backward();
        Tensor4D grad_h = linear.backward(grad);
        Tensor4D grad_x = model.backward(grad_h);
        embed.backward(grad_x);

        // ---- Collect parameters ----
        auto p_model  = model.parameters();
        auto p_embed  = embed.parameters();
        auto p_linear = linear.parameters();

        p_model.insert(p_model.end(),
                       p_embed.begin(),
                       p_embed.end());
        p_model.insert(p_model.end(),
                       p_linear.begin(),
                       p_linear.end());

        optimizer.step(p_model);
    }

    // ===== 推論テスト =====
std::cout << "\n=== Inference Test ===\n";

Tensor4D x_test = embed.forward(input_ids);
Tensor4D h_test = model.forward(x_test);
Tensor4D logits_test = linear.forward(h_test);

// argmax
for(int t=0;t<T;++t)
{
    int best = 0;
    float bestv = -1e9f;

    for(int v=0; v<vocab; ++v)
    {
        float val = logits_test.at(0,t,0,v);
        if(val > bestv)
        {
            bestv = val;
            best = v;
        }
    }

    std::cout << input_ids[t]
              << " -> "
              << best
              << "\n";
}

    std::cout << "train done\n";
    return 0;
}
