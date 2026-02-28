#include "mask/padding_mask.h"

void apply_padding_mask(
    std::vector<float>& logits,
    const std::vector<int>& valid_len,
    int B,int T,int V
){
    const float NEG_INF = -1e9f;

    for(int b=0;b<B;++b){
        for(int t=valid_len[b]; t<T; ++t){
            for(int v=0;v<V;++v){
                logits[(b*T + t)*V + v] = NEG_INF;
            }
        }
    }
}