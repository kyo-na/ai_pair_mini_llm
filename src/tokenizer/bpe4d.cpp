#include "tokenizer/bpe4d.h"
#include <sstream>

BPE4D::BPE4D(){}

void BPE4D::build_vocab(const std::vector<std::string>& corpus)
{
    token_to_id_.clear();
    id_to_token_.clear();

    int id=0;
    for(const auto& line: corpus)
    {
        std::stringstream ss(line);
        std::string word;
        while(ss>>word)
        {
            if(token_to_id_.count(word)==0)
            {
                token_to_id_[word]=id++;
                id_to_token_.push_back(word);
            }
        }
    }
}

std::vector<int> BPE4D::encode(const std::string& text) const
{
    std::vector<int> out;
    std::stringstream ss(text);
    std::string word;
    while(ss>>word)
    {
        auto it=token_to_id_.find(word);
        if(it!=token_to_id_.end())
            out.push_back(it->second);
    }
    return out;
}

std::string BPE4D::decode(const std::vector<int>& tokens) const
{
    std::string out;
    for(int id: tokens)
    {
        if(id>=0 && id<id_to_token_.size())
            out+=id_to_token_[id]+" ";
    }
    return out;
}

int BPE4D::vocab_size() const
{
    return (int)id_to_token_.size();
}