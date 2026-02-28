#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class BPE4D {
public:
    BPE4D();

    void build_vocab(const std::vector<std::string>& corpus);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    int vocab_size() const;

private:
    std::unordered_map<std::string,int> token_to_id_;
    std::vector<std::string> id_to_token_;
};