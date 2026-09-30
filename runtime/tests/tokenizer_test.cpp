#include "tokenizer.h"
#include <cstdio>
#include <fstream>
#include <iterator>

// CPU-only comparison driver. The prompt is a UTF-8 file so Windows argument
// encoding cannot change multilingual text before it reaches the tokenizer.
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    GGUFFile model;
    Tokenizer tokenizer;
    if (!model.open(argv[1]) || !tokenizer.load(model)) return 1;
    std::ifstream input(argv[2],std::ios::binary);
    if (!input) return 1;
    std::string text{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
    auto ids=tokenizer.encode(text);
    std::printf("[");
    for(size_t i=0;i<ids.size();++i) std::printf("%s%d",i?",":"",ids[i]);
    std::puts("]");
    return 0;
}
