#include "mask/causal_mask.h"

void apply_causal_mask(
    std::vector<float>& logits,
    int B,int T,int V
){
    const float NEG_INF = -1e9f;

    for(int b=0;b<B;++b){
        for(int t=0;t<T;++t){
            for(int s=t+1;s<T;++s){
                for(int v=0;v<V;++v){
                    logits[(b*T + s)*V + v] = NEG_INF;
                }
            }
        }
    }
}