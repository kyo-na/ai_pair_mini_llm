#include "mask/padding_mask4d.h"

void PaddingMask4D::apply(
    Tensor4D& scores,
    const std::vector<int>& input_ids,
    int pad_id)
{
    int B = scores.B;
    int H = scores.H;
    int Tq = scores.T;
    int Tk = scores.D;

    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
    for (int t = 0; t < Tq; ++t)
    for (int k = 0; k < Tk; ++k)
    {
        if (k < (int)input_ids.size())
        {
            if (input_ids[k] == pad_id)
            {
                scores.at(b,h,t,k) += -1e9f;
            }
        }
    }
}