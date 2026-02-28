#include "../../include/model/mini_AI.h"
#include "../../include/optimizer/adam.h"

namespace mini_llm {

void train_step(
    MiniAI& model,
    const std::vector<int>& input,
    int target
){
    auto logits = model.forward_ids(input);

    std::vector<float> dlogits(logits.size(),0);
    float loss = -std::log(logits[target]);

    for(size_t i=0;i<logits.size();i++){
        dlogits[i]=logits[i];
    }
    dlogits[target]-=1.0f;

    model.backward(dlogits);
    model.step();
}

}