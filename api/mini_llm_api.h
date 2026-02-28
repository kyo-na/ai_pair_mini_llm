#pragma once

#include "engine/mini_llm/include/tensor4d.h"

struct MiniLLMTask {
    Tensor4D* input;     // 必須
    Tensor4D* target;    // training=false なら null
    bool training;
    float lr;
};

struct MiniLLMResult {
    float loss;
    Tensor4D output;        // Transformer 出力
    Tensor4D world_state;   // World の内部状態
};

class MiniLLM {
public:
    MiniLLM();
    ~MiniLLM();

    MiniLLMResult forward(const MiniLLMTask& task);

private:
    class MiniAI* ai_;
    class World*  world_;
};