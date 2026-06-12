#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

#include "../src/gguf_parser.h"
#include "../src/tokenizer.h"

class TokenizerGGUFBuilder {
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

    void write_kv_array_string(const std::string& key, const std::vector<std::string>& vals) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
        write<uint64_t>(vals.size());
        for (const auto& v : vals) write_string(v);
    }

    void write_kv_array_float32(const std::string& key, const std::vector<float>& vals) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
        write<uint64_t>(vals.size());
        for (auto v : vals) write<float>(v);
    }

    void write_kv_array_int32(const std::string& key, const std::vector<int32_t>& vals) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::INT32));
        write<uint64_t>(vals.size());
        for (auto v : vals) write<int32_t>(v);
    }

    void write_header(uint64_t kv_count) {
        write<uint32_t>(GGUF_MAGIC);
        write<uint32_t>(3);
        write<uint64_t>(0); // tensor_count
        write<uint64_t>(kv_count);
    }
};

void test_extract_vocab_full() {
    TokenizerGGUFBuilder b;
    b.write_header(4);

    std::vector<std::string> tokens = {"<s>", "</s>", "hello", "world"};
    std::vector<float> scores = {0.0f, 0.0f, -1.5f, -2.0f};
    std::vector<int32_t> types = {3, 3, 1, 1};
    std::vector<std::string> merges = {"h e", "l l", "he ll"};

    b.write_kv_array_string("tokenizer.ggml.tokens", tokens);
    b.write_kv_array_float32("tokenizer.ggml.scores", scores);
    b.write_kv_array_int32("tokenizer.ggml.token_type", types);
    b.write_kv_array_string("tokenizer.ggml.merges", merges);

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    auto vocab = extract_vocab(file);

    assert(vocab.tokens.size() == 4);
    assert(vocab.tokens[0] == "<s>");
    assert(vocab.tokens[1] == "</s>");
    assert(vocab.tokens[2] == "hello");
    assert(vocab.tokens[3] == "world");

    assert(vocab.scores.size() == 4);
    assert(vocab.scores[0] == 0.0f);
    assert(vocab.scores[2] == -1.5f);

    assert(vocab.token_types.size() == 4);
    assert(vocab.token_types[0] == 3);
    assert(vocab.token_types[2] == 1);

    assert(vocab.merges.size() == 3);
    assert(vocab.merges[0] == "h e");
    assert(vocab.merges[2] == "he ll");

    printf("  PASS: extract_vocab full\n");
}

void test_extract_vocab_no_merges() {
    TokenizerGGUFBuilder b;
    b.write_header(3);

    b.write_kv_array_string("tokenizer.ggml.tokens", {"a", "b"});
    b.write_kv_array_float32("tokenizer.ggml.scores", {1.0f, 2.0f});
    b.write_kv_array_int32("tokenizer.ggml.token_type", {1, 1});

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    auto vocab = extract_vocab(file);

    assert(vocab.tokens.size() == 2);
    assert(vocab.scores.size() == 2);
    assert(vocab.token_types.size() == 2);
    assert(vocab.merges.empty());

    printf("  PASS: extract_vocab no merges (optional)\n");
}

void test_extract_vocab_missing_tokens() {
    TokenizerGGUFBuilder b;
    b.write_header(0);

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());

    bool threw = false;
    try { extract_vocab(file); }
    catch (const std::runtime_error& e) { threw = true; }
    assert(threw);

    printf("  PASS: extract_vocab throws on missing tokens\n");
}

void test_extract_vocab_missing_scores() {
    TokenizerGGUFBuilder b;
    b.write_header(1);
    b.write_kv_array_string("tokenizer.ggml.tokens", {"a"});

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());

    bool threw = false;
    try { extract_vocab(file); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);

    printf("  PASS: extract_vocab throws on missing scores\n");
}

void test_encode_basic() {
    TokenizerVocab vocab;
    vocab.tokens = {"h", "e", "l", "o", "he", "ll", "hello"};
    vocab.scores = {0, 0, 0, 0, 0, 0, 0};
    vocab.token_types = {1, 1, 1, 1, 1, 1, 1};
    vocab.merges = {"h e", "l l", "he ll", "hell o"};

    BPETokenizer tok(vocab);
    auto ids = tok.encode("hello");
    // Merges: h e -> he, l l -> ll, he ll -> hell, hell o -> hello
    assert(ids.size() == 1);
    assert(ids[0] == 6); // "hello" is token 6
    printf("  PASS: encode basic (hello -> single token)\n");
}

void test_encode_partial_merge() {
    TokenizerVocab vocab;
    vocab.tokens = {"h", "e", "l", "o", "he"};
    vocab.scores = {0, 0, 0, 0, 0};
    vocab.token_types = {1, 1, 1, 1, 1};
    vocab.merges = {"h e"}; // only one merge rule

    BPETokenizer tok(vocab);
    auto ids = tok.encode("hello");
    // h e -> he, then "he", "l", "l", "o" with no more merges
    assert(ids.size() == 4);
    assert(ids[0] == 4); // "he"
    assert(ids[1] == 2); // "l"
    assert(ids[2] == 2); // "l"
    assert(ids[3] == 3); // "o"
    printf("  PASS: encode partial merge\n");
}

void test_encode_empty() {
    TokenizerVocab vocab;
    vocab.tokens = {"a"};
    vocab.scores = {0};
    vocab.token_types = {1};
    BPETokenizer tok(vocab);
    auto ids = tok.encode("");
    assert(ids.empty());
    printf("  PASS: encode empty string\n");
}

void test_encode_byte_fallback() {
    TokenizerVocab vocab;
    // Only byte-fallback tokens, no regular char tokens
    vocab.tokens = {"<0x61>", "<0x62>"}; // 'a'=0x61, 'b'=0x62
    vocab.scores = {0, 0};
    vocab.token_types = {1, 1};
    BPETokenizer tok(vocab);
    auto ids = tok.encode("ab");
    // 'a' and 'b' not in vocab as single chars -> byte fallback
    assert(ids.size() == 2);
    assert(ids[0] == 0); // <0x61>
    assert(ids[1] == 1); // <0x62>
    printf("  PASS: encode byte fallback\n");
}

void test_decode_basic() {
    TokenizerVocab vocab;
    vocab.tokens = {"hello", " ", "world"};
    vocab.scores = {0, 0, 0};
    vocab.token_types = {1, 1, 1};
    BPETokenizer tok(vocab);
    auto result = tok.decode({0, 1, 2});
    assert(result == "hello world");
    printf("  PASS: decode basic\n");
}

void test_decode_byte_fallback() {
    TokenizerVocab vocab;
    vocab.tokens = {"<0x48>", "<0x69>"}; // 'H'=0x48, 'i'=0x69
    vocab.scores = {0, 0};
    vocab.token_types = {1, 1};
    BPETokenizer tok(vocab);
    auto result = tok.decode({0, 1});
    assert(result == "Hi");
    printf("  PASS: decode byte fallback\n");
}

void test_decode_skip_out_of_range() {
    TokenizerVocab vocab;
    vocab.tokens = {"ok"};
    vocab.scores = {0};
    vocab.token_types = {1};
    BPETokenizer tok(vocab);
    auto result = tok.decode({0, 999});
    assert(result == "ok");
    printf("  PASS: decode skips out-of-range IDs\n");
}

void test_encode_decode_roundtrip() {
    TokenizerVocab vocab;
    vocab.tokens = {"h", "e", "l", "o", "he", "ll", "hello"};
    vocab.scores = {0, 0, 0, 0, 0, 0, 0};
    vocab.token_types = {1, 1, 1, 1, 1, 1, 1};
    vocab.merges = {"h e", "l l", "he ll", "hell o"};

    BPETokenizer tok(vocab);
    std::string input = "hello";
    auto ids = tok.encode(input);
    auto output = tok.decode(ids);
    assert(output == input);
    printf("  PASS: encode/decode roundtrip\n");
}

int main() {
    printf("test_tokenizer:\n");
    test_extract_vocab_full();
    test_extract_vocab_no_merges();
    test_extract_vocab_missing_tokens();
    test_extract_vocab_missing_scores();
    test_encode_basic();
    test_encode_partial_merge();
    test_encode_empty();
    test_encode_byte_fallback();
    test_decode_basic();
    test_decode_byte_fallback();
    test_decode_skip_out_of_range();
    test_encode_decode_roundtrip();
    printf("All tokenizer tests passed.\n");
    return 0;
}
