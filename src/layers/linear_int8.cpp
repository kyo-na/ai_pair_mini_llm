#include "layers/linear_int8.h"
#include <cstdint>

LinearINT8::LinearINT8(int in, int out)
: in_(in), out_(out)
{
    q_.in=in; q_.out=out;
    q_.wq.assign((size_t)in*out, 0);
    q_.scale.assign((size_t)out, 1.0f);
}

void LinearINT8::quantize_from_float(const std::vector<float>& w)
{
    q_ = quantize_symmetric_per_out(w, in_, out_);
}

void LinearINT8::forward_vec(const float* x, float* y) const
{
    for(int o=0;o<out_;++o){
        int32_t acc=0;
        const int8_t* wrow = &q_.wq[(size_t)o*in_];
        for(int i=0;i<in_;++i){
            // x は float。ここは “最小実装”。次に x も int8 化するなら別途。
            acc += (int32_t)wrow[i] * (int32_t)x[i];
        }
        y[o] = (float)acc * q_.scale[(size_t)o];
    }
}