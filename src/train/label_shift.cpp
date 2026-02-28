#include "train/label_shift.h"

void label_shift(const std::vector<int>& input,
                 std::vector<int>& shifted_input,
                 std::vector<int>& shifted_target)
{
    shifted_input.clear();
    shifted_target.clear();

    for(size_t i=0;i+1<input.size();++i)
    {
        shifted_input.push_back(input[i]);
        shifted_target.push_back(input[i+1]);
    }
}