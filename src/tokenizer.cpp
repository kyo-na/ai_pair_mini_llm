#include "tokenizer.h"

std::vector<uint32_t> utf8_to_codepoints(const std::string&);
std::string codepoints_to_utf8(const std::vector<uint32_t>&);

std::vector<uint32_t> Tokenizer::encode(const std::string& s){
    return utf8_to_codepoints(s);
}

std::string Tokenizer::decode(const std::vector<uint32_t>& t){
    return codepoints_to_utf8(t);
}