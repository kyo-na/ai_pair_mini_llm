#pragma once
#include "embedding4d.h"
#include "transformer4d.h"
#include "linear4d.h"
#include "world4d.h"

struct Model4D {
    Embedding4D emb;
    Transformer4D tr;
    Linear4D proj;
    World4D world;

    Model4D(int vocab,int h,int d)
        : emb(vocab,d), tr(h,d), proj(d,vocab), world(1,128,h,d) {}

    int forward_token(int token);
};