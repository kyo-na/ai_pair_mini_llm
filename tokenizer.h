#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Tokenizer {
    std::vector<uint32_t> encode(const std::string& s);
    std::string decode(const std::vector<uint32_t>& t);
};