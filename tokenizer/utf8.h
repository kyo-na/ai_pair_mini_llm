#pragma once
#include <vector>
#include <string>

namespace mini_llm {
std::vector<char32_t> utf8_to_codepoints(const std::string& s);
std::string codepoint_to_utf8(char32_t cp);
}