#pragma once
#include <string>
#include <unordered_map>
#include <vector>

class Vocab {
public:
    Vocab();

    // 登録
    int add(const std::string& token);

    // 取得
    int id(const std::string& token) const;
    const std::string& token(int id) const;

    // special ids
    int pad_id() const { return pad_id_; }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

    size_t size() const { return id_to_token_.size(); }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    int pad_id_;
    int bos_id_;
    int eos_id_;
};