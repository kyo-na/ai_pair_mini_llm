#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

struct Int8PerChannel {
    std::vector<int8_t> wq;   // [out * in]
    std::vector<float> scale; // [out]
    int in=0;
    int out=0;
};

inline Int8PerChannel quantize_symmetric_per_out(const std::vector<float>& w, int in, int out)
{
    Int8PerChannel q;
    q.in=in; q.out=out;
    q.wq.resize((size_t)in*out);
    q.scale.resize((size_t)out);

    for(int o=0;o<out;++o){
        float mx=0.0f;
        for(int i=0;i<in;++i){
            mx = std::max(mx, std::fabs(w[(size_t)o*in + i]));
        }
        float s = (mx <= 0.0f) ? 1.0f : (mx / 127.0f);
        q.scale[(size_t)o]=s;

        for(int i=0;i<in;++i){
            float v = w[(size_t)o*in + i] / s;
            int iv = (int)std::lrint(v);
            iv = std::min(127, std::max(-127, iv));
            q.wq[(size_t)o*in + i] = (int8_t)iv;
        }
    }
    return q;
}