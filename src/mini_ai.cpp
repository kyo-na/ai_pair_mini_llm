#include "mini_ai.h"
#include <fstream>
#include <cmath>

MiniAI::MiniAI(int v,int d)
: vocab(v), dim(d),
  emb(v,d), proj(d,v)
{
    std::ifstream f1("engine/mini_llm/train/weights_emb.bin", std::ios::binary);
    f1.read((char*)emb.data.data(), emb.data.size()*sizeof(float));

    std::ifstream f2("engine/mini_llm/train/weights_proj.bin", std::ios::binary);
    f2.read((char*)proj.data.data(), proj.data.size()*sizeof(float));
}

uint32_t MiniAI::forward_token(uint32_t token){
    std::vector<float> h(dim,0.0f);
    for(int i=0;i<dim;i++)
        h[i]=emb(token,i);

    uint32_t best=0;
    float bestv=-1e9f;

    for(uint32_t v=0;v<vocab;v++){
        float s=0;
        for(int i=0;i<dim;i++)
            s+=h[i]*proj(i,v);
        if(s>bestv){bestv=s;best=v;}
    }
    return best;
}