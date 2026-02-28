#include "decode/repetition_penalty4d.h"

void apply_repetition_penalty(
    Tensor4D& logits,
    const std::vector<int>& history,
    float penalty)
{
    for(int b=0;b<logits.B;++b)
    for(int t=0;t<logits.T;++t)
    for(int tok:history)
        logits.at(b,t,0,tok)/=penalty;
}