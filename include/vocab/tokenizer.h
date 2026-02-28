#pragma once
#include <string>
#include <vector>

class Vocab;

class Tokenizer {
public:
    explicit Tokenizer(Vocab& vocab);

    std::vector<int> encode(const std::string& text, bool add_bos = true, bool add_eos = true);
    std::string decode(const std::vector<int>& ids);

private:
    Vocab& vocab_;
};