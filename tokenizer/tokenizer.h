#pragma once
#include <vector>
#include <string>
namespace mini_llm {
class Tokenizer {
public:
    std::vector<char32_t> encode(const std::string& s);
};
}