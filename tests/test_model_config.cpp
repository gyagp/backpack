#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../src/model_config.h"

class GGUFBuilder {
public:
    std::vector<uint8_t> buf;

    template<typename T> void write(T val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    void write_string(const std::string& s) {
        write<uint64_t>(s.size());
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void write_kv_uint32(const std::string& key, uint32_t val) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
        write<uint32_t>(val);
    }

    void write_kv_float32(const std::string& key, float val) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
        write<float>(val);
    }

    void write_kv_string(const std::string& key, const std::string& val) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
        write_string(val);
    }
};

GGUFFile build_llama_gguf() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);  // tensor_count
    b.write<uint64_t>(9);  // metadata_kv_count

    b.write_kv_string("general.architecture", "llama");
    b.write_kv_uint32("llama.block_count", 32);
    b.write_kv_uint32("llama.attention.head_count", 32);
    b.write_kv_uint32("llama.attention.head_count_kv", 8);
    b.write_kv_uint32("llama.embedding_length", 4096);
    b.write_kv_uint32("llama.feed_forward_length", 11008);
    b.write_kv_uint32("llama.vocab_size", 32000);
    b.write_kv_uint32("llama.context_length", 4096);
    b.write_kv_float32("llama.rope.freq_base", 10000.0f);

    return GGUFParser::parse(b.buf.data(), b.buf.size());
}

GGUFFile build_phi3_gguf() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);
    b.write<uint64_t>(9);

    b.write_kv_string("general.architecture", "phi3");
    b.write_kv_uint32("phi3.block_count", 32);
    b.write_kv_uint32("phi3.attention.head_count", 32);
    b.write_kv_uint32("phi3.attention.head_count_kv", 32);
    b.write_kv_uint32("phi3.embedding_length", 3072);
    b.write_kv_uint32("phi3.feed_forward_length", 8192);
    b.write_kv_uint32("phi3.vocab_size", 32064);
    b.write_kv_uint32("phi3.context_length", 4096);
    b.write_kv_float32("phi3.rope.freq_base", 10000.0f);

    return GGUFParser::parse(b.buf.data(), b.buf.size());
}

void test_llama_config() {
    auto file = build_llama_gguf();
    auto cfg = parse_model_config(file);

    assert(cfg.n_layers == 32);
    assert(cfg.n_heads == 32);
    assert(cfg.n_kv_heads == 8);
    assert(cfg.hidden_dim == 4096);
    assert(cfg.intermediate_dim == 11008);
    assert(cfg.vocab_size == 32000);
    assert(cfg.context_length == 4096);
    assert(cfg.rope_theta == 10000.0f);
    printf("  PASS: llama config\n");
}

void test_phi3_config() {
    auto file = build_phi3_gguf();
    auto cfg = parse_model_config(file);

    assert(cfg.n_layers == 32);
    assert(cfg.n_heads == 32);
    assert(cfg.n_kv_heads == 32);
    assert(cfg.hidden_dim == 3072);
    assert(cfg.intermediate_dim == 8192);
    assert(cfg.vocab_size == 32064);
    assert(cfg.context_length == 4096);
    printf("  PASS: phi3 config\n");
}

void test_rope_theta_default() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);
    b.write<uint64_t>(8);

    b.write_kv_string("general.architecture", "llama");
    b.write_kv_uint32("llama.block_count", 32);
    b.write_kv_uint32("llama.attention.head_count", 32);
    b.write_kv_uint32("llama.attention.head_count_kv", 8);
    b.write_kv_uint32("llama.embedding_length", 4096);
    b.write_kv_uint32("llama.feed_forward_length", 11008);
    b.write_kv_uint32("llama.vocab_size", 32000);
    b.write_kv_uint32("llama.context_length", 4096);

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    auto cfg = parse_model_config(file);
    assert(cfg.rope_theta == 10000.0f);
    printf("  PASS: rope_theta default\n");
}

void test_auto_detect_prefix() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);
    b.write<uint64_t>(7);

    b.write_kv_uint32("llama.block_count", 32);
    b.write_kv_uint32("llama.attention.head_count", 32);
    b.write_kv_uint32("llama.attention.head_count_kv", 8);
    b.write_kv_uint32("llama.embedding_length", 4096);
    b.write_kv_uint32("llama.feed_forward_length", 11008);
    b.write_kv_uint32("llama.vocab_size", 32000);
    b.write_kv_uint32("llama.context_length", 4096);

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    auto cfg = parse_model_config(file);
    assert(cfg.n_layers == 32);
    printf("  PASS: auto-detect prefix without general.architecture\n");
}

void test_missing_key_throws() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);
    b.write<uint64_t>(1);

    b.write_kv_string("general.architecture", "llama");

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    bool threw = false;
    try { parse_model_config(file); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    printf("  PASS: missing key throws\n");
}

int main() {
    printf("test_model_config:\n");
    test_llama_config();
    test_phi3_config();
    test_rope_theta_default();
    test_auto_detect_prefix();
    test_missing_key_throws();
    printf("All model config tests passed.\n");
    return 0;
}
