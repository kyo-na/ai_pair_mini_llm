#pragma once

class MiniLLM {
public:
    MiniLLM(int vocab_size, int dim);

    void set_train(bool t);
    int infer_next(int token);

private:
    bool training;
    int vocab_size;
    int dim;
};