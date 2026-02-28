#include "train/label_shift4d.h"

std::vector<int> label_shift(
    const std::vector<int>& ids,
    int B,int T)
{
    std::vector<int> out;
    for(int b=0;b<B;++b)
    for(int t=1;t<T;++t)
        out.push_back(ids[b*T+t]);
    return out;
}