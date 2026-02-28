#include "model4d.h"

int Model4D::forward_token(int token){
    auto x = emb.forward(token);
    auto y = tr.forward(x,world);
    return proj.forward(y);
}