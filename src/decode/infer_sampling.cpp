#include "decode/infer_sampling.h"
#include <algorithm>
#include <cmath>
#include <random>

static void apply_repetition(std::vector<float>& logits,
                             const std::vector<int32_t>& recent,
                             float penalty)
{
    if(penalty <= 1.0f) return;
    for(int32_t id : recent){
        if(id < 0 || id >= (int)logits.size()) continue;
        float& x = logits[(size_t)id];
        if(x > 0) x /= penalty;
        else      x *= penalty;
    }
}

static void softmax(std::vector<float>& p, float temp)
{
    float invT = (temp <= 0.0f) ? 1.0f : (1.0f / temp);
    float mx = -1e30f;
    for(float v : p) mx = std::max(mx, v * invT);
    float sum = 0.0f;
    for(float& v : p){
        v = std::exp(v * invT - mx);
        sum += v;
    }
    float inv = 1.0f / std::max(sum, 1e-20f);
    for(float& v : p) v *= inv;
}

int sample_next_token(const std::vector<float>& logits_in,
                      const std::vector<int32_t>& recent_tokens,
                      const InferSamplingConfig& cfg)
{
    std::vector<float> logits = logits_in;
    apply_repetition(logits, recent_tokens, cfg.repetition_penalty);

    // probs
    std::vector<float> p = logits;
    softmax(p, cfg.temperature);

    // top-k filter
    std::vector<int> idx(p.size());
    for(int i=0;i<(int)idx.size();++i) idx[i]=i;

    std::sort(idx.begin(), idx.end(),
              [&](int a,int b){ return p[a] > p[b]; });

    int k = cfg.top_k;
    if(k > 0 && k < (int)idx.size()){
        for(int i=k;i<(int)idx.size();++i) p[idx[i]] = 0.0f;
    }

    // top-p filter
    if(cfg.top_p > 0.0f && cfg.top_p < 1.0f){
        float cum = 0.0f;
        for(int i=0;i<(int)idx.size();++i){
            cum += p[idx[i]];
            if(cum > cfg.top_p){
                for(int j=i+1;j<(int)idx.size();++j)
                    p[idx[j]] = 0.0f;
                break;
            }
        }
    }

    // renormalize
    float sum=0.0f;
    for(float v:p) sum+=v;
    if(sum <= 0.0f){
        // fallback: argmax
        int best=0;
        for(int i=1;i<(int)logits.size();++i)
            if(logits[i] > logits[best]) best=i;
        return best;
    }
    for(float& v:p) v/=sum;

    std::mt19937 rng(cfg.rng_seed);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    float r = dist(rng);

    float c=0.0f;
    for(int i=0;i<(int)p.size();++i){
        c += p[i];
        if(r <= c) return i;
    }
    return (int)p.size()-1;
}