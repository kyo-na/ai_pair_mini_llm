#include "mini_llm.h"

#include "layers/embedding4d.h"
#include "layers/attention4d.h"
#include "decode/vocab_projection.h"
#include "decode/sampling.h"

MiniLLM::MiniLLM(int vocab_size_, int dim_)
    : training(false),
      vocab_size(vocab_size_),
      dim(dim_) {}

void MiniLLM::set_train(bool t) {
    training = t;
}

int MiniLLM::infer_next(int token) {
    // 最小ダミー推論（本体確認用）
    // token をそのまま返すだけ
    return token;
}