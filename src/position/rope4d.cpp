#include "position/rope4d.h"
#include <cmath>

void RoPE4D::apply(Tensor4D& x)
{
    int B = x.B;
    int T = x.T;
    int H = x.H;
    int D = x.D;

    for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
    {
        for (int d = 0; d + 1 < D; d += 2)
        {
            float theta = std::pow(10000.0f, -float(d) / D);
            float angle = t * theta;

            float cosA = std::cos(angle);
            float sinA = std::sin(angle);

            float x1 = x.at(b,t,h,d);
            float x2 = x.at(b,t,h,d+1);

            x.at(b,t,h,d)   = x1 * cosA - x2 * sinA;
            x.at(b,t,h,d+1) = x1 * sinA + x2 * cosA;
        }
    }
}