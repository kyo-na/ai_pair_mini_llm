#pragma once
#include <vector>

namespace mini_llm {

class Embedding {
public:
    Embedding(int vocab, int dim);
    const std::vector<float>& forward(int id);
    void backward(int id, const std::vector<float>& grad);
    void step(float lr);

private:
    int vocab_, dim_;
    std::vector<float> w_, g_, tmp_;
};

}