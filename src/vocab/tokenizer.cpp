#include "vocab/tokenizer.h"
#include "vocab/vocab.h"
#include <sstream>

Tokenizer::Tokenizer(Vocab& vocab)
    : vocab_(vocab) {}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;

    if (add_bos) {
        ids.push_back(vocab_.bos_id());
    }

    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        int id = vocab_.add(word);
        ids.push_back(id);
    }

    if (add_eos) {
        ids.push_back(vocab_.eos_id());
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        const std::string& tok = vocab_.token(id);
        if (tok == "<BOS>" || tok == "<EOS>" || tok == "<PAD>") continue;
        if (!out.empty()) out += " ";
        out += tok;
    }
    return out;
}