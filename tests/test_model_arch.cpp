#include <cassert>
#include <cstdio>
#include <string>

#include "../src/model_arch.h"
#include "../src/gguf_parser.h"
#include "../src/model_config.h"

void test_global_tensors() {
    auto m = map_tensor_name("token_embd.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::TOKEN_EMBED);
    assert(m->layer == -1);

    m = map_tensor_name("output_norm.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::OUTPUT_NORM);
    assert(m->layer == -1);

    m = map_tensor_name("output.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::OUTPUT);
    assert(m->layer == -1);

    printf("  PASS: global tensors\n");
}

void test_layer_tensors() {
    auto m = map_tensor_name("blk.0.attn_q.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::ATTN_Q);
    assert(m->layer == 0);

    m = map_tensor_name("blk.15.attn_k.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::ATTN_K);
    assert(m->layer == 15);

    m = map_tensor_name("blk.3.attn_v.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::ATTN_V);
    assert(m->layer == 3);

    m = map_tensor_name("blk.7.attn_output.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::ATTN_OUTPUT);
    assert(m->layer == 7);

    m = map_tensor_name("blk.2.attn_norm.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::ATTN_NORM);
    assert(m->layer == 2);

    m = map_tensor_name("blk.5.ffn_gate.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::FFN_GATE);
    assert(m->layer == 5);

    m = map_tensor_name("blk.5.ffn_down.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::FFN_DOWN);
    assert(m->layer == 5);

    m = map_tensor_name("blk.5.ffn_up.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::FFN_UP);
    assert(m->layer == 5);

    m = map_tensor_name("blk.5.ffn_norm.weight");
    assert(m.has_value());
    assert(m->role == TensorRole::FFN_NORM);
    assert(m->layer == 5);

    printf("  PASS: layer tensors\n");
}

void test_qwen3_tensors() {
    // Qwen3-1.7B uses "llama" arch in GGUF; tensor names follow standard blk.N.* convention
    auto m = map_tensor_name("token_embd.weight");
    assert(m.has_value() && m->role == TensorRole::TOKEN_EMBED);

    m = map_tensor_name("blk.0.attn_q.weight");
    assert(m.has_value() && m->role == TensorRole::ATTN_Q && m->layer == 0);

    m = map_tensor_name("blk.27.ffn_gate.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_GATE && m->layer == 27);

    m = map_tensor_name("blk.27.ffn_down.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_DOWN && m->layer == 27);

    m = map_tensor_name("blk.27.ffn_up.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_UP && m->layer == 27);

    m = map_tensor_name("output_norm.weight");
    assert(m.has_value() && m->role == TensorRole::OUTPUT_NORM);

    m = map_tensor_name("output.weight");
    assert(m.has_value() && m->role == TensorRole::OUTPUT);

    assert(detect_model_arch("qwen3") == ModelArch::QWEN3);

    printf("  PASS: Qwen3 tensor mapping\n");
}

void test_phi4_tensors() {
    // Phi-4-mini uses "phi3" arch in GGUF; same standard blk.N.* tensor names
    auto m = map_tensor_name("token_embd.weight");
    assert(m.has_value() && m->role == TensorRole::TOKEN_EMBED);

    m = map_tensor_name("blk.0.attn_q.weight");
    assert(m.has_value() && m->role == TensorRole::ATTN_Q && m->layer == 0);

    m = map_tensor_name("blk.31.attn_output.weight");
    assert(m.has_value() && m->role == TensorRole::ATTN_OUTPUT && m->layer == 31);

    m = map_tensor_name("blk.31.ffn_down.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_DOWN && m->layer == 31);

    m = map_tensor_name("output_norm.weight");
    assert(m.has_value() && m->role == TensorRole::OUTPUT_NORM);

    assert(detect_model_arch("phi3") == ModelArch::PHI4);

    printf("  PASS: Phi-4 tensor mapping\n");
}

void test_unknown_tensor() {
    auto m = map_tensor_name("blk.0.unknown_thing.weight");
    assert(!m.has_value());

    m = map_tensor_name("garbage");
    assert(!m.has_value());

    printf("  PASS: unknown tensor returns nullopt\n");
}

void test_unsupported_arch_throws() {
    bool threw = false;
    try { detect_model_arch("mamba"); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    printf("  PASS: unsupported arch throws\n");
}

void test_qwen2_arch() {
    assert(detect_model_arch("qwen2") == ModelArch::QWEN2);

    auto m = map_tensor_name("token_embd.weight");
    assert(m.has_value() && m->role == TensorRole::TOKEN_EMBED);

    m = map_tensor_name("blk.0.attn_q.weight");
    assert(m.has_value() && m->role == TensorRole::ATTN_Q && m->layer == 0);

    m = map_tensor_name("blk.23.ffn_gate.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_GATE && m->layer == 23);

    printf("  PASS: Qwen2 arch detection and tensor mapping\n");
}

void test_qwen2_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("qwen2");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "qwen2.");

    printf("  PASS: Qwen2 arch prefix detection\n");
}

void test_gemma2_arch() {
    assert(detect_model_arch("gemma2") == ModelArch::GEMMA2);

    auto m = map_tensor_name("token_embd.weight");
    assert(m.has_value() && m->role == TensorRole::TOKEN_EMBED);

    m = map_tensor_name("blk.0.attn_q.weight");
    assert(m.has_value() && m->role == TensorRole::ATTN_Q && m->layer == 0);

    m = map_tensor_name("blk.25.ffn_gate.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_GATE && m->layer == 25);

    m = map_tensor_name("blk.25.ffn_down.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_DOWN && m->layer == 25);

    m = map_tensor_name("blk.25.ffn_up.weight");
    assert(m.has_value() && m->role == TensorRole::FFN_UP && m->layer == 25);

    m = map_tensor_name("output_norm.weight");
    assert(m.has_value() && m->role == TensorRole::OUTPUT_NORM);

    m = map_tensor_name("output.weight");
    assert(m.has_value() && m->role == TensorRole::OUTPUT);

    printf("  PASS: Gemma2 arch detection and tensor mapping\n");
}

void test_gemma2_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("gemma2");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "gemma2.");

    printf("  PASS: Gemma2 arch prefix detection\n");
}

void test_gemma3_arch() {
    assert(detect_model_arch("gemma3") == ModelArch::GEMMA3);
    assert(detect_model_arch("gemma") == ModelArch::GEMMA3);
    printf("  PASS: Gemma3 arch detection (gemma3 and gemma strings)\n");
}

void test_gemma3_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;

    file.metadata["general.architecture"] = std::string("gemma3");
    assert(detect_arch_prefix(file) == "gemma3.");

    file.metadata["general.architecture"] = std::string("gemma");
    assert(detect_arch_prefix(file) == "gemma3.");

    printf("  PASS: Gemma3 arch prefix detection\n");
}

void test_llama_arch() {
    assert(detect_model_arch("llama") == ModelArch::LLAMA);
    printf("  PASS: LLAMA arch detection\n");
}

void test_llama_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("llama");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "llama.");

    printf("  PASS: LLAMA arch prefix detection\n");
}

void test_nemotron_arch() {
    assert(detect_model_arch("nemotron") == ModelArch::NEMOTRON);
    printf("  PASS: NEMOTRON arch detection\n");
}

void test_nemotron_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("nemotron");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "nemotron.");

    printf("  PASS: NEMOTRON arch prefix detection\n");
}

void test_granite_arch() {
    assert(detect_model_arch("granite") == ModelArch::GRANITE);
    printf("  PASS: GRANITE arch detection\n");
}

void test_granite_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("granite");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "granite.");

    printf("  PASS: GRANITE arch prefix detection\n");
}

void test_internlm2_arch() {
    assert(detect_model_arch("internlm2") == ModelArch::INTERNLM2);
    printf("  PASS: INTERNLM2 arch detection\n");
}

void test_internlm2_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("internlm2");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "internlm2.");

    printf("  PASS: INTERNLM2 arch prefix detection\n");
}

void test_gpt_oss_arch() {
    assert(detect_model_arch("gpt-oss") == ModelArch::GPT_OSS);
    printf("  PASS: GPT_OSS arch detection\n");
}

void test_gpt_oss_arch_prefix() {
    GGUFFile file;
    file.version = 3;
    file.tensor_count = 0;
    file.metadata_kv_count = 1;
    file.metadata["general.architecture"] = std::string("gpt-oss");

    std::string prefix = detect_arch_prefix(file);
    assert(prefix == "gpt-oss.");

    printf("  PASS: GPT_OSS arch prefix detection\n");
}

int main() {
    printf("test_model_arch:\n");
    test_global_tensors();
    test_layer_tensors();
    test_qwen3_tensors();
    test_phi4_tensors();
    test_unknown_tensor();
    test_unsupported_arch_throws();
    test_qwen2_arch();
    test_qwen2_arch_prefix();
    test_gemma2_arch();
    test_gemma2_arch_prefix();
    test_gemma3_arch();
    test_gemma3_arch_prefix();
    test_llama_arch();
    test_llama_arch_prefix();
    test_nemotron_arch();
    test_nemotron_arch_prefix();
    test_granite_arch();
    test_granite_arch_prefix();
    test_internlm2_arch();
    test_internlm2_arch_prefix();
    test_gpt_oss_arch();
    test_gpt_oss_arch_prefix();
    printf("All model arch tests passed.\n");
    return 0;
}
