#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand((unsigned)std::time(nullptr));

    std::ofstream ofs("../data/shard_000.txt");
    if (!ofs) return 1;

    for (int i = 0; i < 200000; i++) {
        int a = std::rand() % 10;
        int b = std::rand() % 10;
        ofs << "if state " << a << " then state " << b << ".\n";
    }

    return 0;
}