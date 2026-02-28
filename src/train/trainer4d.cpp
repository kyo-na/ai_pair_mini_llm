#include "train/trainer4d.h"
#include <iostream>
#include <cmath>

Trainer4D::Trainer4D(
    TransformerStack4D* model,
    float lr,
    int seq_len)
    : model_(model),
      optimizer_(lr),
      scheduler_(lr),
      seq_len_(seq_len)
{}

void Trainer4D::train(
    std::vector<int>& data,
    int epochs)
{
    DataLoader4D loader(data, seq_len_);

    for(int epoch=0; epoch<epochs; ++epoch)
    {
        std::vector<int> input;
        std::vector<int> target;

        float total_loss = 0.0f;
        int steps = 0;

        loader = DataLoader4D(data, seq_len_);

        while(loader.next_batch(input, target))
        {
            Tensor4D x = model_->forward_ids(input);
            float loss = loss_.forward(x, target);

            Tensor4D grad = loss_.backward();
            model_->backward(grad);

            auto params = model_->parameters();
            train::optim::clip_grad_norm(params, 1.0f);

            optimizer_.step(params);

            total_loss += loss;
            steps++;
        }

        float avg_loss = total_loss / steps;
        float ppl = std::exp(avg_loss);

        std::cout << "epoch "
                  << epoch
                  << " loss="
                  << avg_loss
                  << " ppl="
                  << ppl
                  << "\n";

        scheduler_.step();
    }
}