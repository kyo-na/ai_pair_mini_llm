#include <iostream>
#include "tensor4d.h"

int main(){
    Tensor4D emb(1,1,1,64), proj(1,1,1,64);
    emb.load("weights_emb.bin");
    proj.load("weights_proj.bin");

    std::cout<<"chat ready\n";
    for(int i=0;i<8;i++)
        std::cout<<proj.data[i]<<" ";
    std::cout<<"\n";
}