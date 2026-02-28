#include "decode/repetition_penalty.h"

void apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int>& history,
    float penalty
){
    if(penalty<=1.0f) return;
    for(int id:history){
        float& l=logits[id];
        if(l>0) l/=penalty;
        else    l*=penalty;
    }
}