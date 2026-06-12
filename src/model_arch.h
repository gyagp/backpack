#pragma once

#include <cstdint>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>

enum class TensorRole {
    TOKEN_EMBED,
    OUTPUT_NORM,
    OUTPUT,

    ATTN_NORM,
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_OUTPUT,

    FFN_NORM,
    FFN_GATE,
    FFN_DOWN,
    FFN_UP,
};

struct TensorMapping {
    TensorRole role;
    int layer;  // -1 for non-layer tensors
};

enum class ModelArch {
    QWEN2,
    QWEN3,
    PHI4,
    GEMMA2,
    GEMMA3,
    LLAMA,
    NEMOTRON,
    GRANITE,
    INTERNLM2,
    GPT_OSS,
};

inline std::optional<TensorMapping> map_tensor_name(const std::string& name) {
    static const std::unordered_map<std::string, TensorRole> global_tensors = {
        {"token_embd.weight",  TensorRole::TOKEN_EMBED},
        {"output_norm.weight", TensorRole::OUTPUT_NORM},
        {"output.weight",      TensorRole::OUTPUT},
    };

    auto git = global_tensors.find(name);
    if (git != global_tensors.end())
        return TensorMapping{git->second, -1};

    static const std::regex blk_re(R"(blk\.(\d+)\.(\w+(?:\.\w+)*)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, blk_re))
        return std::nullopt;

    int layer = std::stoi(m[1].str());
    std::string part = m[2].str();

    static const std::unordered_map<std::string, TensorRole> layer_tensors = {
        {"attn_norm",   TensorRole::ATTN_NORM},
        {"attn_q",      TensorRole::ATTN_Q},
        {"attn_k",      TensorRole::ATTN_K},
        {"attn_v",      TensorRole::ATTN_V},
        {"attn_output", TensorRole::ATTN_OUTPUT},
        {"ffn_norm",    TensorRole::FFN_NORM},
        {"ffn_gate",    TensorRole::FFN_GATE},
        {"ffn_down",    TensorRole::FFN_DOWN},
        {"ffn_up",      TensorRole::FFN_UP},
    };

    auto lit = layer_tensors.find(part);
    if (lit != layer_tensors.end())
        return TensorMapping{lit->second, layer};

    return std::nullopt;
}

inline ModelArch detect_model_arch(const std::string& arch_string) {
    if (arch_string == "qwen2")
        return ModelArch::QWEN2;
    if (arch_string == "qwen3")
        return ModelArch::QWEN3;
    if (arch_string == "phi3" || arch_string == "phi4")
        return ModelArch::PHI4;
    if (arch_string == "gemma2")
        return ModelArch::GEMMA2;
    if (arch_string == "gemma3" || arch_string == "gemma")
        return ModelArch::GEMMA3;
    if (arch_string == "llama")
        return ModelArch::LLAMA;
    if (arch_string == "nemotron")
        return ModelArch::NEMOTRON;
    if (arch_string == "granite")
        return ModelArch::GRANITE;
    if (arch_string == "internlm2")
        return ModelArch::INTERNLM2;
    if (arch_string == "gpt-oss")
        return ModelArch::GPT_OSS;
    throw std::runtime_error("ModelArch: unsupported architecture: " + arch_string);
}
