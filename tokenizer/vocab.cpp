#include "vocab.h"

namespace mini_llm {

int Vocab::token_to_id(char32_t cp) {
    auto it = cp2id_.find(cp);
    if (it != cp2id_.end()) return it->second;
    int id = (int)id2cp_.size();
    cp2id_[cp] = id;
    id2cp_.push_back(cp);
    return id;
}

char32_t Vocab::id_to_token(int id) const {
    if (id < 0 || id >= (int)id2cp_.size()) return U'?';
    return id2cp_[id];
}

}