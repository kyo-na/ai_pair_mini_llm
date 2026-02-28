#include "data/dataloader4d.h"

DataLoader4D::DataLoader4D(const std::vector<int>& data,int seq_len)
: data_(data), seq_len_(seq_len), pos_(0)
{}

bool DataLoader4D::next_batch(std::vector<int>& input,
                              std::vector<int>& target)
{
    if(pos_+seq_len_+1 >= data_.size())
        return false;

    input.assign(data_.begin()+pos_,
                 data_.begin()+pos_+seq_len_);

    target.assign(data_.begin()+pos_+1,
                  data_.begin()+pos_+seq_len_+1);

    pos_+=seq_len_;
    return true;
}