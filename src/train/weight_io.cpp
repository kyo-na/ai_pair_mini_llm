#include <fstream>
#include <stdexcept>
#include "../../include/tensor4d.h"

namespace mini_llm {

void load_weights(const char* path, Tensor4D& w){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("weight load failed");

    size_t sz;
    f.read((char*)&sz, sizeof(sz));
    if(sz != w.data.size())
        throw std::runtime_error("weight size mismatch");

    f.read((char*)w.data.data(), sz*sizeof(float));
}

void save_weights(const char* path, const Tensor4D& w){
    std::ofstream f(path, std::ios::binary);
    size_t sz = w.data.size();
    f.write((char*)&sz, sizeof(sz));
    f.write((char*)w.data.data(), sz*sizeof(float));
}

}