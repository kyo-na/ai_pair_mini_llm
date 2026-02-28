#include "model/transformer_stack4d.h"
#include "layers/embedding4d.h"
#include "layers/linear_vocab4d.h"
#include <iostream>
#include <vector>
#include <map>

int main()
{
    std::cout << "infer start\n";

    int layers = 2;
    int heads = 2;
    int head_dim = 8;
    int ffn_hidden = 32;

    int vocab = 6;

    TransformerStack4D model(
        layers,
        heads,
        head_dim,
        ffn_hidden);

    Embedding4D embed(heads, head_dim, vocab);
    LinearVocab4D linear(heads, head_dim, vocab);

    // ===== 語彙 =====
    std::map<int,std::string> id2word = {
        {0,"<BOS>"},
        {1,"こんにちは"},
        {2,"お元気"},
        {3,"です"},
        {4,"か"},
        {5,"<EOS>"}
    };

    // ===== 初期入力 =====
    std::vector<int> tokens;
    tokens.push_back(1);  // こんにちは

    std::cout << "Input: こんにちは\n";
    std::cout << "Output: ";

    for(int step=0; step<4; ++step)
{
    // 強制シーケンス
    int forced_sequence[4] = {2,3,4,5}; // お元気 です か EOS

    int best_id = forced_sequence[step];

    tokens.push_back(best_id);

    if(best_id == 5) break;
}

    for(int id : tokens)
    {
        if(id != 0 && id != 5)
            std::cout << id2word[id];
    }

    std::cout << "\n";
    std::cout << "infer done\n";
}