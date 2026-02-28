#pragma once
#include <vector>

class DataLoader4D {
public:
    DataLoader4D(const std::vector<int>& data, int seq_len);

    bool next_batch(std::vector<int>& input,
                    std::vector<int>& target);

private:
    std::vector<int> data_;
    int seq_len_;
    int pos_;
};