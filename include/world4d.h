#pragma once
#include <vector>
#include <cstddef>

struct World4D {
    size_t B, T, D, C;
    size_t stride_B, stride_T, stride_D;
    std::vector<float> data;

    World4D() : B(0), T(0), D(0), C(0), stride_B(0), stride_T(0), stride_D(0) {}

    World4D(size_t b, size_t t, size_t d, size_t c)
        : B(b), T(t), D(d), C(c),
          stride_D(c),
          stride_T(d * c),
          stride_B(t * d * c),
          data(b * t * d * c, 0.0f) {}

    inline float& at(size_t b, size_t t, size_t d, size_t c) {
        return data[b * stride_B + t * stride_T + d * stride_D + c];
    }
    inline const float& at(size_t b, size_t t, size_t d, size_t c) const {
        return data[b * stride_B + t * stride_T + d * stride_D + c];
    }

    inline size_t batch()   const { return B; }
    inline size_t time()    const { return T; }
    inline size_t depth()   const { return D; }
    inline size_t channel() const { return C; }
};