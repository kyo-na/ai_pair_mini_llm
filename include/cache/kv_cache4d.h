#pragma once
#include "tensor4d.h"

class KVCache4D {
public:
    KVCache4D(int heads, int head_dim, int max_len);

    void reset();

    // append new step
    void append(const Tensor4D& K_new,
                const Tensor4D& V_new);

    Tensor4D& K();
    Tensor4D& V();

    int length() const;

private:
    int H_;
    int Dh_;
    int max_len_;

    int cur_len_;

    Tensor4D K_cache_; // (1, max_len, H, Dh)
    Tensor4D V_cache_; // (1, max_len, H, Dh)
};