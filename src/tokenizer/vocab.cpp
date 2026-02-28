#include "tokenizer/vocab.h"
#include <sstream>

Vocab::Vocab() {
    add_token("<PAD>");
    add_token("<BOS>");
    add_token("<EOS>");
    add_token("<UNK>");
    add_token("hello");
    add_token("genki");
    add_token("?");
    add_token("yes");
}

void Vocab::add_token(const std::string& token) {
    int id = (int)id_to_token.size();
    id_to_token.push_back(token);
    token_to_id[token] = id;
}

int Vocab::encode_token(const std::string& token) const {
    auto it = token_to_id.find(token);
    if (it == token_to_id.end())
        return unk_id();
    return it->second;
}

std::string Vocab::decode_token(int id) const {
    if (id < 0 || id >= (int)id_to_token.size())
        return "<UNK>";
    return id_to_token[id];
}

std::vector<int> Vocab::encode(const std::string& text, bool add_special) const {
    std::vector<int> ids;

    if (add_special)
        ids.push_back(bos_id());

    // 超単純：空白区切り
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        ids.push_back(encode_token(token));
    }

    if (add_special)
        ids.push_back(eos_id());

    return ids;
}

std::string Vocab::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == bos_id() || id == eos_id() || id == pad_id())
            continue;
        result += decode_token(id);
        result += " ";
    }
    return result;
}