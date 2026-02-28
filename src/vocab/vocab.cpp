#include "vocab/vocab.h"
#include <stdexcept>

Vocab::Vocab() {
    pad_id_ = add("<PAD>");
    bos_id_ = add("<BOS>");
    eos_id_ = add("<EOS>");
}

int Vocab::add(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    int new_id = (int)id_to_token_.size();
    token_to_id_[token] = new_id;
    id_to_token_.push_back(token);
    return new_id;
}

int Vocab::id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) {
        throw std::runtime_error("unknown token: " + token);
    }
    return it->second;
}

const std::string& Vocab::token(int id) const {
    if (id < 0 || (size_t)id >= id_to_token_.size()) {
        static std::string unk = "<UNK>";
        return unk;
    }
    return id_to_token_[id];
}