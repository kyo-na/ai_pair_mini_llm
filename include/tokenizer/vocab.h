#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Vocab {
public:
    Vocab();

    int bos_id() const { return token_to_id.at("<BOS>"); }
    int eos_id() const { return token_to_id.at("<EOS>"); }
    int pad_id() const { return token_to_id.at("<PAD>"); }
    int unk_id() const { return token_to_id.at("<UNK>"); }

    int size() const { return (int)id_to_token.size(); }

    int encode_token(const std::string& token) const;
    std::string decode_token(int id) const;

    std::vector<int> encode(const std::string& text, bool add_special=true) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string,int> token_to_id;

    void add_token(const std::string& token);
};