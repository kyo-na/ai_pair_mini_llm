#include "layers/attention4d.h"
#include "cache/kv_cache4d.h"
#include <cmath>

/*
    Flash-style single step inference
    - input x: (1,1,1,Dmodel_)
    - KVCache: per-layer cache
    - no T^2 memory
*/

Tensor4D Attention4D::forward_flash(
    const Tensor4D& x,
    KVCache4D& cache)
{
    // x: (1,1,1,Dmodel_)
    int B = 1;
    int T = 1;

    // === 1️⃣ Project Q,K,V ===
    Tensor4D Q(1,1,H_,Dh_);
    Tensor4D K_new(1,1,H_,Dh_);
    Tensor4D V_new(1,1,H_,Dh_);

    for(int o=0;o<Dmodel_;++o)
    {
        float q=0,k=0,v=0;

        for(int i=0;i<Dmodel_;++i)
        {
            float xi = x.at(0,0,0,i);

            q += xi * Wq_.at(0,0,i,o);
            k += xi * Wk_.at(0,0,i,o);
            v += xi * Wv_.at(0,0,i,o);
        }

        int h = o / Dh_;
        int d = o % Dh_;

        Q.at(0,0,h,d)      = q;
        K_new.at(0,0,h,d)  = k;
        V_new.at(0,0,h,d)  = v;
    }

    // === 2️⃣ Append to cache ===
    cache.append(K_new, V_new);

    int L = cache.length();

    Tensor4D& K_all = cache.K();
    Tensor4D& V_all = cache.V();

    Tensor4D context(1,1,H_,Dh_);

    // === 3️⃣ Flash-style attention ===
    for(int h=0;h<H_;++h)
    {
        float max_s = -1e9f;

        // ---- pass 1: find max ----
        for(int t=0;t<L;++t)
        {
            float s=0;

            for(int d=0;d<Dh_;++d)
                s += Q.at(0,0,h,d)
                   * K_all.at(0,t,h,d);

            s /= std::sqrt((float)Dh_);

            if(s > max_s)
                max_s = s;
        }

        float denom = 0.0f;

        // ---- pass 2: accumulate ----
        for(int t=0;t<L;++t)
        {
            float s=0;

            for(int d=0;d<Dh_;++d)
                s += Q.at(0,0,h,d)
                   * K_all.at(0,t,h,d);

            s = std::exp(
                    s/std::sqrt((float)Dh_)
                    - max_s);

            denom += s;

            for(int d=0;d<Dh_;++d)
                context.at(0,0,h,d) +=
                    s * V_all.at(0,t,h,d);
        }

        // ---- normalize ----
        for(int d=0;d<Dh_;++d)
            context.at(0,0,h,d) /= denom;
    }

    // === 4️⃣ Merge heads + Wo ===
    Tensor4D out(1,1,1,Dmodel_);

    for(int o=0;o<Dmodel_;++o)
    {
        int h = o / Dh_;
        int d = o % Dh_;

        float merged = context.at(0,0,h,d);

        float sum=0;

        for(int i=0;i<Dmodel_;++i)
            sum += merged * Wo_.at(0,0,o,i);

        out.at(0,0,0,o) = sum;
    }

    return out;
}