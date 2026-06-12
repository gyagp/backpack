#include "../src/onnx_inference.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/onnx_loader.h"

static void write_varint(std::vector<uint8_t>& buf, uint64_t v) {
    while (v >= 0x80) { buf.push_back(static_cast<uint8_t>(v & 0x7F) | 0x80); v >>= 7; }
    buf.push_back(static_cast<uint8_t>(v));
}
static void write_tag(std::vector<uint8_t>& buf, uint32_t field, uint32_t wire_type) {
    write_varint(buf, (static_cast<uint64_t>(field) << 3) | wire_type);
}
static void write_string_field(std::vector<uint8_t>& buf, uint32_t field, const std::string& s) {
    write_tag(buf, field, 2);
    write_varint(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}
static void write_varint_field(std::vector<uint8_t>& buf, uint32_t field, uint64_t v) {
    write_tag(buf, field, 0);
    write_varint(buf, v);
}
static void write_bytes_field(std::vector<uint8_t>& buf, uint32_t field, const std::vector<uint8_t>& data) {
    write_tag(buf, field, 2);
    write_varint(buf, data.size());
    buf.insert(buf.end(), data.begin(), data.end());
}

static std::vector<uint8_t> build_initializer(const std::string& name, int32_t data_type,
                                               const std::vector<int64_t>& dims,
                                               const void* raw, size_t raw_size) {
    std::vector<uint8_t> tensor;
    {
        std::vector<uint8_t> packed;
        for (auto d : dims) write_varint(packed, static_cast<uint64_t>(d));
        write_bytes_field(tensor, 1, packed);
    }
    write_varint_field(tensor, 2, static_cast<uint64_t>(data_type));
    write_string_field(tensor, 8, name);
    {
        std::vector<uint8_t> rd(static_cast<const uint8_t*>(raw),
                                 static_cast<const uint8_t*>(raw) + raw_size);
        write_bytes_field(tensor, 13, rd);
    }
    return tensor;
}

static std::vector<uint8_t> build_node(const std::string& op_type, const std::string& name,
                                        const std::vector<std::string>& inputs,
                                        const std::vector<std::string>& outputs) {
    std::vector<uint8_t> node;
    for (auto& in : inputs) write_string_field(node, 1, in);
    for (auto& out : outputs) write_string_field(node, 2, out);
    write_string_field(node, 3, name);
    write_string_field(node, 4, op_type);
    return node;
}

static std::vector<uint8_t> build_test_model() {
    float W_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float B_data[2] = {0.5f, -0.5f};
    auto W_init = build_initializer("W", 1, {2, 2}, W_data, sizeof(W_data));
    auto B_init = build_initializer("B", 1, {2}, B_data, sizeof(B_data));
    auto matmul_node = build_node("MatMul", "matmul0", {"X", "W"}, {"matmul_out"});
    auto add_node = build_node("Add", "add0", {"matmul_out", "B"}, {"Y"});

    std::vector<uint8_t> X_vi;
    write_string_field(X_vi, 1, "X");
    std::vector<uint8_t> Y_vi;
    write_string_field(Y_vi, 1, "Y");

    std::vector<uint8_t> graph;
    write_bytes_field(graph, 1, matmul_node);
    write_bytes_field(graph, 1, add_node);
    write_string_field(graph, 2, "test_graph");
    write_bytes_field(graph, 5, W_init);
    write_bytes_field(graph, 5, B_init);
    write_bytes_field(graph, 11, X_vi);
    write_bytes_field(graph, 12, Y_vi);

    std::vector<uint8_t> model;
    write_varint_field(model, 1, 7);
    write_string_field(model, 3, "backpack_test");
    write_string_field(model, 4, "1.0");
    write_bytes_field(model, 7, graph);
    std::vector<uint8_t> opset;
    write_varint_field(opset, 2, 13);
    write_bytes_field(model, 2, opset);
    return model;
}

int main() {
    int passed = 0, failed = 0;

    // AC1: onnx_inference.h compiles and GenerateResult is usable from real header
    {
        std::cout << "Test: GenerateResult from real header... ";
        GenerateResult r{};
        r.tokens.push_back(42);
        r.prefill_tok_per_sec = 100.0;
        r.decode_tok_per_sec = 50.0;
        assert(r.tokens[0] == 42);
        assert(r.prefill_tok_per_sec == 100.0);
        assert(r.decode_tok_per_sec == 50.0);
        std::cout << "PASSED" << std::endl;
        passed++;
    }

    // AC2: OnnxGenerateParams from real header has correct defaults
    {
        std::cout << "Test: OnnxGenerateParams defaults... ";
        OnnxGenerateParams p;
        assert(p.max_tokens == 128);
        assert(p.temperature == 0.0f);
        assert(p.top_k == 40);
        assert(p.top_p == 0.9f);
        assert(p.shader_dir == "src/shaders");
        std::cout << "PASSED" << std::endl;
        passed++;
    }

    // AC3: generate_onnx function signature is correct (function pointer assignment)
    {
        std::cout << "Test: generate_onnx function signature... ";
        using GenFn = GenerateResult(*)(const std::string&,
                                        const std::vector<uint32_t>&,
                                        const OnnxGenerateParams&);
        GenFn fn = &generate_onnx;
        assert(fn != nullptr);
        (void)fn;
        std::cout << "PASSED" << std::endl;
        passed++;
    }

    // AC4: ONNX model can be loaded end-to-end
    {
        std::cout << "Test: ONNX model loading... ";
        auto model_bytes = build_test_model();
        const char* tmpenv = std::getenv("TEMP");
        if (!tmpenv) tmpenv = std::getenv("TMP");
        if (!tmpenv) tmpenv = "/tmp";
        std::string path = std::string(tmpenv) + "/test_onnx_inf_ac4.onnx";
        {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());
        }
        auto model = onnx::load_onnx(path);
        assert(model.graph.nodes.size() == 2);
        assert(model.graph.initializers.size() == 2);
        assert(!model.graph.inputs.empty());
        assert(!model.graph.outputs.empty());
        std::remove(path.c_str());
        std::cout << "PASSED" << std::endl;
        passed++;
    }

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed > 0 ? 1 : 0;
}
