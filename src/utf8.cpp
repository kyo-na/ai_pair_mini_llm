#include <vector>
#include <string>
#include <cstdint>

std::vector<uint32_t> utf8_to_codepoints(const std::string& s){
    std::vector<uint32_t> out;
    for(unsigned char c: s) out.push_back(c);
    return out;
}

std::string codepoints_to_utf8(const std::vector<uint32_t>& v){
    std::string s;
    for(auto c: v) s.push_back((char)c);
    return s;
}