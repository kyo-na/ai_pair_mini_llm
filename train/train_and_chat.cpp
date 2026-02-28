#include "../include/tensor4d.h"
#include "../include/adam.h"
#include "../include/softmax_ce4d.h"
#include <iostream>
#include <vector>
#include <map>
#include <string>

int main(){
    // ---- vocabulary ----
    std::vector<std::string> vocab = {
        "hi","hello","how","are","you","i","am","fine","bye"
    };
    std::map<std::string,int> tok;
    for(int i=0;i<vocab.size();i++) tok[vocab[i]]=i;
    int V=vocab.size(), D=8;

    // ---- dataset (conversation pairs) ----
    std::vector<int> data = {
        tok["hi"], tok["hello"],
        tok["how"], tok["are"],
        tok["you"], tok["i"],
        tok["am"], tok["fine"],
        tok["bye"], tok["bye"]
    };

    // ---- tensors ----
    Tensor4D emb(1,1,V,D);
    Tensor4D proj(1,1,D,V);

    Adam optE(emb.data.size());
    Adam optP(proj.data.size());

    // ---- training ----
    for(int epoch=0;epoch<300;epoch++){
        float loss_sum=0;
        for(int i=0;i+1<data.size();i++){
            int x=data[i], y=data[i+1];
            emb.zero_grad(); proj.zero_grad();

            // forward
            std::vector<float> h(D,0), logits(V,0);
            for(int d=0;d<D;d++)
                h[d]=emb.at(0,0,x,d);
            for(int v=0;v<V;v++)
                for(int d=0;d<D;d++)
                    logits[v]+=h[d]*proj.at(0,0,d,v);

            std::vector<float> dlogits;
            float loss=softmax_ce_4d(logits,V,y,dlogits);
            loss_sum+=loss;

            // backward proj
            for(int d=0;d<D;d++)
                for(int v=0;v<V;v++)
                    proj.grad[proj.idx(0,0,d,v)]+=h[d]*dlogits[v];

            // backward emb
            for(int d=0;d<D;d++){
                float g=0;
                for(int v=0;v<V;v++)
                    g+=proj.at(0,0,d,v)*dlogits[v];
                emb.grad[emb.idx(0,0,x,d)]+=g;
            }

            optP.update(proj);
            optE.update(emb);
        }
        if(epoch%50==0)
            std::cout<<"epoch "<<epoch<<" loss "<<loss_sum<<"\n";
    }

    emb.save("weights_emb.bin");
    proj.save("weights_proj.bin");

    // ---- chat ----
    std::cout<<"\n--- chat ---\n";
    std::string in;
    while(true){
        std::cout<<"> ";
        std::cin>>in;
        if(tok.find(in)==tok.end()) break;

        int x=tok[in];
        std::vector<float> h(D,0), logits(V,0);
        for(int d=0;d<D;d++)
            h[d]=emb.at(0,0,x,d);
        for(int v=0;v<V;v++)
            for(int d=0;d<D;d++)
                logits[v]+=h[d]*proj.at(0,0,d,v);

        int best=0;
        for(int v=1;v<V;v++)
            if(logits[v]>logits[best]) best=v;

        std::cout<<vocab[best]<<"\n";
    }
}