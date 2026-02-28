#include "world/world_loss.h"
#include <cassert>
#include <cmath>

static inline float clampf(float x, float lo, float hi){
    return std::max(lo, std::min(hi, x));
}

float WorldConsistencyLoss::forward(const Tensor4D& pred,
                                    const Tensor4D& actual)
{
    assert(pred.B == actual.B);
    assert(pred.T == actual.T);
    assert(pred.H == actual.H);
    assert(pred.D == actual.D);

    loss = 0.0f;
    grad = Tensor4D(pred.B, pred.T, pred.H, pred.D);

    // t=0 は比較しない（初期状態）
    for(int b=0;b<pred.B;b++){
        for(int t=1;t<pred.T;t++){
            for(int h=0;h<pred.H;h++){
                for(int d=0;d<pred.D;d++){
                    float diff =
                        pred.at(b,t,h,d) - actual.at(b,t,h,d);
                    loss += diff * diff;

                    // dL/dpred = 2 * diff
                    float g = 2.0f * diff;
                    grad.at(b,t,h,d) = clampf(g, -1.0f, 1.0f);
                }
            }
        }
    }

    // 正規化（スケール安定）
    float norm = (float)(pred.B * (pred.T-1) * pred.H * pred.D);
    if(norm > 0.0f) loss /= norm;

    return loss;
}

const Tensor4D& WorldConsistencyLoss::backward() const
{
    return grad;
}