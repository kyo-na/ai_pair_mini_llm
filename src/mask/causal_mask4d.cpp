#include "mask/causal_mask4d.h"
#include <limits>

void CausalMask4D::apply(Tensor4D& scores)
{
    int B = scores.B;
    int T = scores.T;
    int H = scores.H;
    int Tk = scores.D; // DにTk入れてる設計なら

    float neg_inf = -1e9f;

    for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
    for (int tk = t+1; tk < Tk; ++tk)
    {
        scores.at(b,t,h,tk) = neg_inf;
    }
}