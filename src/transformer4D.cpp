#include "transformer4d.h"

Tensor4D Transformer4D::forward(const Tensor4D& x, World4D& world){
    auto y = attn.forward(x,world);
    world.step();
    return y;
}