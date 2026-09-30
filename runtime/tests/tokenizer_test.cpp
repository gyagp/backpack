#include "tokenizer.h"
#include <cstdio>
#include <fstream>
#include <iterator>

// CPU-only comparison driver. The prompt is a UTF-8 file so Windows argument
// encoding cannot change multilingual text before it reaches the tokenizer.
int main(int argc, char** argv) {
    if (argc == 1) {
        GGUFFile metadata;
        metadata.metadata["tokenizer.ggml.tokens"] = std::vector<std::string>{
            "text","<eos>","<bos>","<turn|>","<eom>","<control>"};
        metadata.metadata["tokenizer.ggml.eos_token_id"] = uint32_t(1);
        metadata.metadata["tokenizer.ggml.eot_token_id"] = uint32_t(3);
        metadata.metadata["tokenizer.ggml.eom_token_id"] = uint32_t(4);
        Tokenizer tokenizer;
        if (!tokenizer.load(metadata) || !tokenizer.is_end_token(1) ||
            !tokenizer.is_end_token(3) || !tokenizer.is_end_token(4) ||
            tokenizer.is_end_token(-1) || tokenizer.is_end_token(0) ||
            tokenizer.is_end_token(2) || tokenizer.is_end_token(5)) return 1;
        metadata.metadata.erase("tokenizer.ggml.eot_token_id");
        metadata.metadata.erase("tokenizer.ggml.eom_token_id");
        if (!tokenizer.load(metadata) || !tokenizer.is_end_token(1) ||
            tokenizer.is_end_token(3) || tokenizer.is_end_token(4)) return 1;
        metadata.metadata["tokenizer.ggml.eot_token_id"] = uint32_t(1);
        if (!tokenizer.load(metadata) || !tokenizer.is_end_token(1) ||
            tokenizer.is_end_token(-1)) return 1;
        std::puts("PASS GGUF end-token metadata, non-stop controls, reload, and duplicate IDs");
        return 0;
    }
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
