#pragma once
#include <unordered_map>
#include <vector>

namespace mini_llm {

class Vocab {
public:
    int token_to_id(char32_t cp);
    char32_t id_to_token(int id) const;
    int size() const { return (int)id2cp_.size(); }

private:
    std::unordered_map<char32_t,int> cp2id_;
    std::vector<char32_t> id2cp_;
};

}