#pragma once
#include <vector>

namespace mini_llm {

class Attention {
public:
    Attention(int dim);

    // context: [t-n ... t]
    std::vector<float> forward(
        const std::vector<std::vector<float>>& context);

private:
    int dim_;
};

}