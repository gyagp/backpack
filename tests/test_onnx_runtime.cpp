#include "../src/onnx_runtime.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define ASSERT_NEAR(a, b, eps, msg) do { \
    float _a = (a), _b = (b); \
    if (std::fabs(_a - _b) > (eps)) { \
        std::fprintf(stderr, "FAIL: %s: expected %f, got %f (diff %e)\n", msg, _b, _a, std::fabs(_a - _b)); \
        return 1; \
    } \
} while(0)

// --- Protobuf helpers (same as test_onnx_loader.cpp) ---

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

static std::vector<uint8_t> build_tensor_shape(const std::vector<int64_t>& dims) {
    std::vector<uint8_t> shape_buf;
    for (auto d : dims) {
        std::vector<uint8_t> dim_buf;
        write_varint_field(dim_buf, 1, static_cast<uint64_t>(d));
        write_bytes_field(shape_buf, 1, dim_buf);
    }
    return shape_buf;
}

static std::vector<uint8_t> build_type_proto(int32_t elem_type, const std::vector<int64_t>& dims) {
    std::vector<uint8_t> tensor_type;
    write_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
    auto shape = build_tensor_shape(dims);
    write_bytes_field(tensor_type, 2, shape);
    std::vector<uint8_t> type_proto;
    write_bytes_field(type_proto, 1, tensor_type);
    return type_proto;
}

static std::vector<uint8_t> build_value_info(const std::string& name, int32_t elem_type, const std::vector<int64_t>& dims) {
    std::vector<uint8_t> vi;
    write_string_field(vi, 1, name);
    auto tp = build_type_proto(elem_type, dims);
    write_bytes_field(vi, 2, tp);
    return vi;
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

// Build test model: Y = MatMul(X, W) + B
// X=[1,2] (1x2), W=identity (2x2), B=[0.5, -0.5] (2,)
// Expected: Y = [1*1+2*0+0.5, 1*0+2*1-0.5] = [1.5, 1.5]
static std::vector<uint8_t> build_test_model() {
    float W_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float B_data[2] = {0.5f, -0.5f};

    auto W_init = build_initializer("W", 1, {2, 2}, W_data, sizeof(W_data));
    auto B_init = build_initializer("B", 1, {2}, B_data, sizeof(B_data));

    auto matmul_node = build_node("MatMul", "matmul0", {"X", "W"}, {"matmul_out"});
    auto add_node = build_node("Add", "add0", {"matmul_out", "B"}, {"Y"});

    auto X_vi = build_value_info("X", 1, {1, 2});
    auto Y_vi = build_value_info("Y", 1, {1, 2});

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
    {
        std::vector<uint8_t> opset;
        write_varint_field(opset, 2, 13);
        write_bytes_field(model, 2, opset);
    }
    return model;
}

int main() {
    // Build and save test model
    auto model_bytes = build_test_model();
    const char* tmpenv = std::getenv("TEMP");
    if (!tmpenv) tmpenv = std::getenv("TMP");
    if (!tmpenv) tmpenv = "/tmp";
    std::string path = std::string(tmpenv) + "/test_onnx_runtime.onnx";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());
    }

    // Load model
    auto model = onnx::load_onnx(path);
    assert(model.graph.nodes.size() == 2);
    std::cout << "Model loaded: " << model.graph.nodes.size() << " nodes\n";

    // Create GPU context and runtime
    auto ctx = create_gpu_context();
    std::string shader_dir = "src/shaders";
    onnx_runtime::OnnxRuntime runtime(ctx, shader_dir);

    // Input: X = [1.0, 2.0]
    std::unordered_map<std::string, std::vector<float>> inputs;
    inputs["X"] = {1.0f, 2.0f};

    auto outputs = runtime.run(model.graph, inputs);

    // Expected: Y = MatMul([1,2], I_2x2) + [0.5, -0.5] = [1.5, 1.5]
    assert(outputs.count("Y"));
    auto& Y = outputs["Y"];
    assert(Y.size() == 2);

    std::cout << "Y = [" << Y[0] << ", " << Y[1] << "]\n";
    ASSERT_NEAR(Y[0], 1.5f, 1e-4f, "Y[0]");
    ASSERT_NEAR(Y[1], 1.5f, 1e-4f, "Y[1]");

    std::remove(path.c_str());
    std::cout << "All ONNX runtime tests passed!\n";
    return 0;
}
