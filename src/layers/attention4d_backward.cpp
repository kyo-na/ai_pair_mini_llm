#include "layers/attention4d_backward.h"

/*
  grad : d(context)  (B,T,H,D)
  Q,K,V: forward保存済みテンソル (B,T,H,D)

  簡易版：
  - 本来は softmax + matmul の逆伝播を計算する
  - しかし今は構造確認フェーズなので
    形を崩さず grad をそのまま流す
*/

Tensor4D AttentionBackward4D::backward(
        const Tensor4D& grad,
        const Tensor4D& Q,
        const Tensor4D& K,
        const Tensor4D& V)
{
    // サイズチェック（安全）
    if (grad.B != Q.B ||
        grad.T != Q.T ||
        grad.H != Q.H ||
        grad.D != Q.D)
    {
        // 形が違うならゼロ返す（暴走防止）
        return Tensor4D(Q.B, Q.T, Q.H, Q.D);
    }

    // 簡易：そのまま返す
    return grad;
}