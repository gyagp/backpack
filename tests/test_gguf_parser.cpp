#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

#include "../src/gguf_parser.h"

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

    void write_kv_bool(const std::string& key, bool val) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::BOOL));
        write<uint8_t>(val ? 1 : 0);
    }

    void write_kv_array_uint32(const std::string& key, const std::vector<uint32_t>& vals) {
        write_string(key);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
        write<uint64_t>(vals.size());
        for (auto v : vals) write<uint32_t>(v);
    }

    void write_tensor_info(const std::string& name, const std::vector<uint64_t>& dims, uint32_t dtype, uint64_t offset) {
        write_string(name);
        write<uint32_t>(static_cast<uint32_t>(dims.size()));
        for (auto d : dims) write<uint64_t>(d);
        write<uint32_t>(dtype);
        write<uint64_t>(offset);
    }
};

void test_basic_header() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);  // version
    b.write<uint64_t>(0);  // tensor_count
    b.write<uint64_t>(0);  // metadata_kv_count

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    assert(file.version == 3);
    assert(file.tensor_count == 0);
    assert(file.metadata_kv_count == 0);
    assert(file.metadata.empty());
    printf("  PASS: basic header\n");
}

void test_invalid_magic() {
    GGUFBuilder b;
    b.write<uint32_t>(0xDEADBEEF);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);
    b.write<uint64_t>(0);

    bool threw = false;
    try { GGUFParser::parse(b.buf.data(), b.buf.size()); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    printf("  PASS: invalid magic rejected\n");
}

void test_metadata_types() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);  // tensor_count
    b.write<uint64_t>(5);  // 5 kv pairs

    b.write_kv_uint32("block_count", 32);
    b.write_kv_float32("rope_theta", 10000.0f);
    b.write_kv_string("arch", "llama");
    b.write_kv_bool("use_bias", false);
    b.write_kv_array_uint32("dims", {4096, 11008, 32000});

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    assert(file.metadata_kv_count == 5);
    assert(file.metadata.size() == 5);

    assert(std::get<uint32_t>(file.metadata["block_count"]) == 32);
    assert(std::get<float>(file.metadata["rope_theta"]) == 10000.0f);
    assert(std::get<std::string>(file.metadata["arch"]) == "llama");
    assert(std::get<bool>(file.metadata["use_bias"]) == false);

    auto& arr = std::get<GGUFArray>(file.metadata["dims"]);
    assert(arr.element_type == GGUFValueType::UINT32);
    assert(arr.values.size() == 3);
    assert(std::get<uint32_t>(arr.values[0]) == 4096);
    assert(std::get<uint32_t>(arr.values[1]) == 11008);
    assert(std::get<uint32_t>(arr.values[2]) == 32000);

    printf("  PASS: metadata types\n");
}

void test_int_types() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);
    b.write<uint64_t>(6);

    // uint8
    b.write_string("u8"); b.write<uint32_t>(0); b.write<uint8_t>(255);
    // int8
    b.write_string("i8"); b.write<uint32_t>(1); b.write<int8_t>(-1);
    // uint16
    b.write_string("u16"); b.write<uint32_t>(2); b.write<uint16_t>(1000);
    // int16
    b.write_string("i16"); b.write<uint32_t>(3); b.write<int16_t>(-500);
    // uint64
    b.write_string("u64"); b.write<uint32_t>(10); b.write<uint64_t>(0xFFFFFFFFFFULL);
    // float64
    b.write_string("f64"); b.write<uint32_t>(12); b.write<double>(3.14159265358979);

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    assert(std::get<uint8_t>(file.metadata["u8"]) == 255);
    assert(std::get<int8_t>(file.metadata["i8"]) == -1);
    assert(std::get<uint16_t>(file.metadata["u16"]) == 1000);
    assert(std::get<int16_t>(file.metadata["i16"]) == -500);
    assert(std::get<uint64_t>(file.metadata["u64"]) == 0xFFFFFFFFFFULL);
    assert(std::get<double>(file.metadata["f64"]) == 3.14159265358979);

    printf("  PASS: int/float types\n");
}

void test_tensor_info_parsing() {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(3);  // 3 tensors
    b.write<uint64_t>(1);  // 1 kv pair

    b.write_kv_uint32("block_count", 32);

    b.write_tensor_info("weight.embed", {4096, 32000}, 0, 0);
    b.write_tensor_info("attn.q", {4096, 4096}, 1, 524288000);
    b.write_tensor_info("ffn.gate", {4096, 11008}, 15, 558956544);

    auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
    assert(file.tensor_count == 3);
    assert(file.tensors.size() == 3);

    assert(file.tensors[0].name == "weight.embed");
    assert(file.tensors[0].ndims == 2);
    assert(file.tensors[0].dimensions[0] == 4096);
    assert(file.tensors[0].dimensions[1] == 32000);
    assert(file.tensors[0].dtype == GGUFDType::F32);
    assert(file.tensors[0].offset == 0);

    assert(file.tensors[1].name == "attn.q");
    assert(file.tensors[1].dtype == GGUFDType::F16);
    assert(file.tensors[1].offset == 524288000);

    assert(file.tensors[2].name == "ffn.gate");
    assert(file.tensors[2].dtype == GGUFDType::Q4_K_M);

    assert(file.data_offset % 32 == 0);
    assert(file.data_offset > 0);

    printf("  PASS: tensor info parsing\n");
}

void test_data_offset_alignment() {
    for (int pad = 0; pad < 32; pad++) {
        GGUFBuilder b;
        b.write<uint32_t>(GGUF_MAGIC);
        b.write<uint32_t>(3);
        b.write<uint64_t>(1);
        b.write<uint64_t>(0);

        std::string name(pad, 'x');
        b.write_tensor_info(name, {256}, 8, 0);

        auto file = GGUFParser::parse(b.buf.data(), b.buf.size());
        assert(file.data_offset % 32 == 0);
        assert(file.tensors[0].dtype == GGUFDType::Q8_0);
    }
    printf("  PASS: data offset alignment\n");
}

int main() {
    printf("test_gguf_parser:\n");
    test_basic_header();
    test_invalid_magic();
    test_metadata_types();
    test_int_types();
    test_tensor_info_parsing();
    test_data_offset_alignment();
    printf("All GGUF parser tests passed.\n");
    return 0;
}
