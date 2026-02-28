#pragma once
#include <memory>
#include <vector>

#include "model/transformer_stack4d.h"
#include "loss/cross_entropy4d.h"
#include "optimizer/adamw.h"
#include "train/optim/grad_clip.h"
#include "train/optim/lr_scheduler.h"
#include "data/dataloader4d.h"

class Trainer4D {
public:
    Trainer4D(
        TransformerStack4D* model,
        float lr,
        int seq_len);

    void train(std::vector<int>& data,
               int epochs);

private:
    TransformerStack4D* model_;
    CrossEntropy4D loss_;
    AdamW optimizer_;
    LRScheduler scheduler_;
    int seq_len_;
};