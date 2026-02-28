#include "infer/inference_engine.h"
#include <algorithm>
#include <cmath>
#include <random>

static std::mt19937 rng(1234);

InferenceEngine::InferenceEngine(
    int vocab,
    int layers,
    int heads,
    int head_dim,
    int ffn_hidden,
    int max_seq_len,
    float temperature,
    int top_k,
    float top_p)
: temperature_(temperature),
  top_k_(top_k),
  top_p_(top_p),
  embedding_(vocab,1,heads*head_dim),
  stack_(layers,heads,head_dim,ffn_hidden,max_seq_len),
  projection_(embedding_.weight())
{}

void InferenceEngine::reset()
{
    stack_.reset();
}

int InferenceEngine::step(int token_id)
{
    std::vector<int> ids = {token_id};
    Tensor4D x = embedding_.forward(ids);
    Tensor4D hidden = stack_.forward_step(x);
    Tensor4D logits = projection_.forward(hidden);

    int vocab = logits.D;
    std::vector<float> probs(vocab);

    float max_logit=-1e9f;

    for(int i=0;i<vocab;++i)
    {
        float v = logits.at(0,0,0,i) / temperature_;
        if(v>max_logit) max_logit=v;
        probs[i]=v;
    }

    float sum=0.0f;
    for(int i=0;i<vocab;++i)
    {
        probs[i]=std::exp(probs[i]-max_logit);
        sum+=probs[i];
    }

    for(int i=0;i<vocab;++i)
        probs[i]/=sum;

    // ---- Top-k ----
    if(top_k_>0 && top_k_<vocab)
    {
        std::vector<int> idx(vocab);
        for(int i=0;i<vocab;++i) idx[i]=i;

        std::partial_sort(
            idx.begin(),
            idx.begin()+top_k_,
            idx.end(),
            [&](int a,int b){
                return probs[a]>probs[b];
            });

        std::vector<float> new_probs(vocab,0.0f);
        float new_sum=0.0f;

        for(int i=0;i<top_k_;++i)
        {
            new_probs[idx[i]]=probs[idx[i]];
            new_sum+=probs[idx[i]];
        }

        for(int i=0;i<vocab;++i)
            new_probs[i]/=new_sum;

        probs=new_probs;
    }

    // ---- Top-p ----
    if(top_p_>0.0f && top_p_<1.0f)
    {
        std::vector<int> idx(vocab);
        for(int i=0;i<vocab;++i) idx[i]=i;

        std::sort(idx.begin(),idx.end(),
            [&](int a,int b){
                return probs[a]>probs[b];
            });

        float cumulative=0.0f;
        std::vector<float> new_probs(vocab,0.0f);

        for(int i=0;i<vocab;++i)
        {
            cumulative+=probs[idx[i]];
            new_probs[idx[i]]=probs[idx[i]];
            if(cumulative>=top_p_)
                break;
        }

        float new_sum=0.0f;
        for(float v:new_probs) new_sum+=v;

        for(int i=0;i<vocab;++i)
            new_probs[i]/=new_sum;

        probs=new_probs;
    }

    return sample(probs);
}

int InferenceEngine::sample(std::vector<float>& probs)
{
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    float r=dist(rng);

    float cumulative=0.0f;
    for(int i=0;i<(int)probs.size();++i)
    {
        cumulative+=probs[i];
        if(r<cumulative)
            return i;
    }

    return probs.size()-1;
}