#include <iostream>
#include "../include/layers/linear4d.h"

int main() {
    Tensor4D x(1,8,1,1);
    for (int i=0;i<8;i++) x.at(0,i,0,0)=0.1f*i;

    Linear4D fc(8,8);
    OptimizerContext opt;
    opt.lr = 0.01f;

    for (int step=0;step<1000;step++) {
        auto y = fc.forward(x);

        // dummy loss: sum
        Tensor4D gy = y;
        for (auto& v : gy.data) v = 1.0f;

        fc.backward(gy,x);
        fc.step(opt);

        if (step%100==0)
            std::cout<<"step "<<step<<"\n";
    }
}