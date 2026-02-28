#include <iostream>
#include <fstream>
#include <string>

#include "../tokenizer/tokenizer.h"
#include "../tokenizer/vocab.h"

using namespace mini_llm;

int main() {
    std::cout << "start\n";

    Tokenizer tokenizer;
    Vocab vocab;

    std::ifstream ifs("../data/shard_000.txt");
    if (!ifs) {
        std::cerr << "failed to open data file\n";
        return 1;
    }

    std::cout << "file opened\n";

    std::string line;
    long long line_count = 0;

    while (std::getline(ifs, line)) {
        auto tokens = tokenizer.encode(line);

        vocab.observe(tokens);   // ★ 追加

        line_count++;
        if (line_count % 10000 == 0) {
            std::cout << "lines=" << line_count << "\n";
        }
    }

    vocab.finalize();           // ★ 追加
    vocab.save("vocab.txt");    // ★ 追加

    std::cout << "done\n";
    std::cout << "vocab size=" << vocab.size() << "\n";

    return 0;
}