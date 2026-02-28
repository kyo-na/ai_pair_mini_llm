#pragma once
#include "tensor4d.h"

struct WorldState {
    Tensor4D latent;   // (B,T,H,D)

    WorldState() = default;

    WorldState(int B, int T, int H, int D)
        : latent(B,T,H,D) {}

    void zero(){
        std::fill(latent.data.begin(),
                  latent.data.end(),
                  0.0f);
    }
};