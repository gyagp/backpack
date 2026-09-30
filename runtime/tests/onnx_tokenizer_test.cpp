#include "onnx_tokenizer.h"
#include <cstdio>
#include <stdexcept>

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    try {
        OnnxTokenizer tokenizer;
        if(!tokenizer.load(argv[1]))return 1;
        for(const auto* token:{"<|im_end|>","<|endoftext|>"}) {
            auto ids=tokenizer.encode(token);
            if(ids.size()!=1 || !tokenizer.is_end_token(ids[0]) || tokenizer.decode_token(ids[0])!=token)
                throw std::runtime_error("Qwen end token metadata or added-token decoding is wrong");
        }
        auto start=tokenizer.encode("<|im_start|>");
        if(start.size()!=1 || tokenizer.is_end_token(start[0]) || tokenizer.decode_token(start[0])!="<|im_start|>")
            throw std::runtime_error("Non-terminal chat token was misclassified");
        auto word=tokenizer.encode("hello");
        for(auto id:word)if(tokenizer.is_end_token(id))throw std::runtime_error("Ordinary text became a stop token");
        std::puts("PASS nested EOS metadata, multiple end tokens, and added-token roundtrip");
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
