#include "inference.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static std::string gguf_get_string_or(const GGUFFile& file, const std::string& key,
                                       const std::string& fallback) {
    auto it = file.metadata.find(key);
    if (it == file.metadata.end()) return fallback;
    try {
        return std::get<std::string>(it->second);
    } catch (...) {
        return fallback;
    }
}

static std::string gguf_get_quant_or(const GGUFFile& file, const std::string& fallback) {
    auto it = file.metadata.find("general.file_type");
    if (it == file.metadata.end()) return fallback;
    try {
        uint32_t ft = std::get<uint32_t>(it->second);
        switch (ft) {
            case 2:  return "Q4_0";
            case 3:  return "Q4_1";
            case 7:  return "Q8_0";
            case 15: return "Q4_K_M";
            case 17: return "Q5_K_M";
            case 19: return "Q6_K";
            default: return "type_" + std::to_string(ft);
        }
    } catch (...) {
        return fallback;
    }
}

static std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: bench_backpack <model.gguf>\n");
        return 1;
    }

    const std::string gguf_path = argv[1];
    const std::string prompt = "Hello, how are you?";

    GenerateParams params{};
    params.max_tokens = 128;
    params.temperature = 0.0f;

    MmapFile mmap(gguf_path);
    auto gguf = GGUFFile::parse(mmap.data(), mmap.size());

    std::string model_name = gguf_get_string_or(gguf, "general.name", "unknown");
    std::string quant = gguf_get_quant_or(gguf, "unknown");

    auto result = generate(gguf_path, prompt, params);

    uint32_t prefill_toks = 0;
    {
        auto vocab = extract_vocab(gguf);
        BPETokenizer tokenizer(vocab);
        auto formatted = format_chat({{"user", prompt}}, true);
        prefill_toks = static_cast<uint32_t>(tokenizer.encode(formatted).size());
    }
    uint32_t decode_toks = static_cast<uint32_t>(result.tokens.size());

    std::printf("{\"model\":\"%s\",\"quant\":\"%s\","
                "\"prefill_toks\":%u,\"decode_toks\":%u,"
                "\"prefill_tok_s\":%.2f,\"decode_tok_s\":%.2f}\n",
                escape_json(model_name).c_str(),
                escape_json(quant).c_str(),
                prefill_toks, decode_toks,
                result.prefill_tok_per_sec,
                result.decode_tok_per_sec);

    return 0;
}
