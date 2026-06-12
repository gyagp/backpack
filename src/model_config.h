#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

#include "gguf_parser.h"

struct ModelConfig {
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t hidden_dim = 0;
    uint32_t intermediate_dim = 0;
    uint32_t vocab_size = 0;
    uint32_t context_length = 0;
    float rope_theta = 10000.0f;
};

inline uint32_t gguf_get_u32(const GGUFFile& file, const std::string& key) {
    auto it = file.metadata.find(key);
    if (it == file.metadata.end())
        throw std::runtime_error("ModelConfig: missing key " + key);
    return std::get<uint32_t>(it->second);
}

inline float gguf_get_f32(const GGUFFile& file, const std::string& key) {
    auto it = file.metadata.find(key);
    if (it == file.metadata.end())
        throw std::runtime_error("ModelConfig: missing key " + key);
    return std::get<float>(it->second);
}

inline std::string detect_arch_prefix(const GGUFFile& file) {
    auto it = file.metadata.find("general.architecture");
    if (it != file.metadata.end()) {
        const auto& arch = std::get<std::string>(it->second);
        if (arch == "qwen2") return "qwen2.";
        if (arch == "llama") return "llama.";
        if (arch == "phi3") return "phi3.";
        if (arch == "gemma2") return "gemma2.";
        if (arch == "gemma3" || arch == "gemma") return "gemma3.";
        if (arch == "nemotron") return "nemotron.";
        if (arch == "granite") return "granite.";
        if (arch == "internlm2") return "internlm2.";
        if (arch == "gpt-oss") return "gpt-oss.";
        return arch + ".";
    }
    if (file.metadata.count("llama.block_count")) return "llama.";
    if (file.metadata.count("phi3.block_count")) return "phi3.";
    throw std::runtime_error("ModelConfig: cannot detect architecture");
}

inline ModelConfig parse_model_config(const GGUFFile& file) {
    std::string p = detect_arch_prefix(file);

    ModelConfig cfg;
    cfg.n_layers         = gguf_get_u32(file, p + "block_count");
    cfg.n_heads          = gguf_get_u32(file, p + "attention.head_count");
    cfg.n_kv_heads       = gguf_get_u32(file, p + "attention.head_count_kv");
    cfg.hidden_dim       = gguf_get_u32(file, p + "embedding_length");
    cfg.intermediate_dim = gguf_get_u32(file, p + "feed_forward_length");
    cfg.vocab_size       = gguf_get_u32(file, p + "vocab_size");
    cfg.context_length   = gguf_get_u32(file, p + "context_length");

    auto it = file.metadata.find(p + "rope.freq_base");
    if (it != file.metadata.end())
        cfg.rope_theta = std::get<float>(it->second);

    return cfg;
}
