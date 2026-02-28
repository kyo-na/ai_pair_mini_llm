#include "layers/attention4d.h"
#include "simd/dot_simd.h"
#include <cmath>
#include <algorithm>

static constexpr int TILE = 64;

Tensor4D Attention4D::forward_infer_fused(
    const Tensor4D& x_step,
    KVCacheRing4D* ring,
    InferContext& ctx)
{
    // x_step: (B,1,H,D)
    const int B = x_step.B;
    const int H = H_;
    const int D = D_;

    // arenaから一時バッファ確保（Q,K,V: B*H*D）
    float* qbuf = (float*)ctx.arena.alloc(sizeof(float)*B*H*D, 64);
    float* kbuf = (float*)ctx.arena.alloc(sizeof(float)*B*H*D, 64);
    float* vbuf = (float*)ctx.arena.alloc(sizeof(float)*B*H*D, 64);

    auto Q = [&](int b,int h,int d)->float& { return qbuf[(b*H+h)*D + d]; };
    auto K1= [&](int b,int h,int d)->float& { return kbuf[(b*H+h)*D + d]; };
    auto V1= [&](int b,int h,int d)->float& { return vbuf[(b*H+h)*D + d]; };

    // ---- projection（1step）----
    for(int b=0;b<B;++b)
    for(int h=0;h<H;++h)
    for(int d=0;d<D;++d){
        float q=0,k=0,v=0;
        for(int i=0;i<D;++i){
            float xi = x_step.at(b,0,h,i);
            q += xi * Wq_.at(0,0,i,d);
            k += xi * Wk_.at(0,0,i,d);
            v += xi * Wv_.at(0,0,i,d);
        }
        Q(b,h,d)=q;
        K1(b,h,d)=k;
        V1(b,h,d)=v;
    }

    // ---- ringに追加 ----
    {
        Tensor4D Kt(B,1,H,D);
        Tensor4D Vt(B,1,H,D);
        for(int b=0;b<B;++b)
        for(int h=0;h<H;++h)
        for(int d=0;d<D;++d){
            Kt.at(b,0,h,d)=K1(b,h,d);
            Vt.at(b,0,h,d)=V1(b,h,d);
        }
        ring->append_step(Kt,Vt);
    }

    const int L = ring->length();
    const float inv_sqrt = 1.0f / std::sqrt((float)D);

    Tensor4D context(B,1,H,D);

    // ---- fused softmax + sum(e*V) ----
    for(int b=0;b<B;++b)
    for(int h=0;h<H;++h)
    {
        const float* qptr = &Q(b,h,0);

        // pass1: max
        float max_s = -1e30f;
        for(int tile=0; tile<L; tile+=TILE){
            int end = std::min(tile+TILE, L);
            for(int t=tile; t<end; ++t){
                // K at logical t
                // ringは “物理に分散” してるので一旦テンポラリに積む（D小ならこれが速い）
                float* kt = (float*)ctx.arena.alloc(sizeof(float)*D, 64);
                for(int d=0; d<D; ++d) kt[d] = ring->K_at(b,t,h,d);

                float s = dot_f32_simd(qptr, kt, D) * inv_sqrt;
                max_s = std::max(max_s, s);
            }
        }

        // pass2: exp + denom + context accumulate
        float denom = 0.0f;
        for(int d=0; d<D; ++d) context.at(b,0,h,d)=0.0f;

        for(int tile=0; tile<L; tile+=TILE){
            int end = std::min(tile+TILE, L);
            for(int t=tile; t<end; ++t){
                float* kt = (float*)ctx.arena.alloc(sizeof(float)*D, 64);
                float* vt = (float*)ctx.arena.alloc(sizeof(float)*D, 64);
                for(int d=0; d<D; ++d){
                    kt[d] = ring->K_at(b,t,h,d);
                    vt[d] = ring->V_at(b,t,h,d);
                }

                float s = dot_f32_simd(qptr, kt, D) * inv_sqrt;
                float e = std::exp(s - max_s);
                denom += e;

                for(int d=0; d<D; ++d){
                    context.at(b,0,h,d) += e * vt[d];
                }
            }
        }

        float inv = 1.0f / std::max(denom, 1e-20f);
        for(int d=0; d<D; ++d){
            context.at(b,0,h,d) *= inv;
        }
    }

    // ---- output proj ----
    Tensor4D out(B,1,H,D);
    for(int b=0;b<B;++b)
    for(int h=0;h<H;++h)
    for(int d=0;d<D;++d){
        float s=0;
        for(int i=0;i<D;++i)
            s += context.at(b,0,h,i) * Wo_.at(0,0,i,d);
        out.at(b,0,h,d)=s;
    }

    return out;
}