#include "../src/onnx_loader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static void write_varint(std::vector<uint8_t>& buf, uint64_t v) {
    while (v >= 0x80) {
        buf.push_back(static_cast<uint8_t>(v & 0x7F) | 0x80);
        v >>= 7;
    }
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
    for (auto& in : inputs)  write_string_field(node, 1, in);
    for (auto& out : outputs) write_string_field(node, 2, out);
    write_string_field(node, 3, name);
    write_string_field(node, 4, op_type);
    return node;
}

static std::vector<uint8_t> build_initializer(const std::string& name, int32_t data_type,
                                               const std::vector<int64_t>& dims,
                                               const void* raw, size_t raw_size) {
    std::vector<uint8_t> tensor;
    // dims (field 1, packed)
    {
        std::vector<uint8_t> packed;
        for (auto d : dims) write_varint(packed, static_cast<uint64_t>(d));
        write_bytes_field(tensor, 1, packed);
    }
    write_varint_field(tensor, 2, static_cast<uint64_t>(data_type));
    write_string_field(tensor, 8, name);
    // raw_data (field 13)
    {
        std::vector<uint8_t> rd(static_cast<const uint8_t*>(raw),
                                 static_cast<const uint8_t*>(raw) + raw_size);
        write_bytes_field(tensor, 13, rd);
    }
    return tensor;
}

// Build a minimal ONNX model: Y = MatMul(X, W) + B
// Graph: 2 nodes (MatMul, Add), 1 input (X), 1 output (Y), 2 initializers (W, B)
static std::vector<uint8_t> build_test_model() {
    // Initializer data
    float W_data[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // 2x2 identity
    float B_data[2] = {0.5f, -0.5f};

    auto W_init = build_initializer("W", 1/*FLOAT*/, {2, 2}, W_data, sizeof(W_data));
    auto B_init = build_initializer("B", 1/*FLOAT*/, {2}, B_data, sizeof(B_data));

    auto matmul_node = build_node("MatMul", "matmul0", {"X", "W"}, {"matmul_out"});
    auto add_node    = build_node("Add", "add0", {"matmul_out", "B"}, {"Y"});

    auto X_vi = build_value_info("X", 1/*FLOAT*/, {1, 2});
    auto Y_vi = build_value_info("Y", 1/*FLOAT*/, {1, 2});

    // Graph
    std::vector<uint8_t> graph;
    write_bytes_field(graph, 1, matmul_node);
    write_bytes_field(graph, 1, add_node);
    write_string_field(graph, 2, "test_graph");
    write_bytes_field(graph, 5, W_init);
    write_bytes_field(graph, 5, B_init);
    write_bytes_field(graph, 11, X_vi);
    write_bytes_field(graph, 12, Y_vi);

    // Model
    std::vector<uint8_t> model;
    write_varint_field(model, 1, 7); // ir_version
    write_string_field(model, 3, "backpack_test");
    write_string_field(model, 4, "1.0");
    write_bytes_field(model, 7, graph);

    // Opset import (field 2): version=13
    {
        std::vector<uint8_t> opset;
        write_varint_field(opset, 2, 13);
        write_bytes_field(model, 2, opset);
    }

    return model;
}

int main() {
    auto model_bytes = build_test_model();

    // Write to temp file
    std::string tmp_dir;
    const char* tmpenv = std::getenv("TEMP");
    if (!tmpenv) tmpenv = std::getenv("TMP");
    if (!tmpenv) tmpenv = "/tmp";
    tmp_dir = tmpenv;
    std::string path = tmp_dir + "/test_model_tmp.onnx";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());
    }

    // Load and verify
    auto model = onnx::load_onnx(path);

    // ir_version
    assert(model.ir_version == 7);
    std::cout << "ir_version: " << model.ir_version << " OK\n";

    // producer
    assert(model.producer_name == "backpack_test");
    assert(model.producer_version == "1.0");
    std::cout << "producer: " << model.producer_name << " v" << model.producer_version << " OK\n";

    // opset
    assert(model.opset_version == 13);
    std::cout << "opset_version: " << model.opset_version << " OK\n";

    // Graph name
    assert(model.graph.name == "test_graph");

    // Nodes
    assert(model.graph.nodes.size() == 2);
    assert(model.graph.nodes[0].op_type == "MatMul");
    assert(model.graph.nodes[0].name == "matmul0");
    assert(model.graph.nodes[0].inputs.size() == 2);
    assert(model.graph.nodes[0].inputs[0] == "X");
    assert(model.graph.nodes[0].inputs[1] == "W");
    assert(model.graph.nodes[0].outputs.size() == 1);
    assert(model.graph.nodes[0].outputs[0] == "matmul_out");

    assert(model.graph.nodes[1].op_type == "Add");
    assert(model.graph.nodes[1].name == "add0");
    std::cout << "nodes: " << model.graph.nodes.size() << " OK\n";

    // Inputs
    assert(model.graph.inputs.size() == 1);
    assert(model.graph.inputs[0].name == "X");
    assert(model.graph.inputs[0].elem_type == onnx::FLOAT);
    assert(model.graph.inputs[0].shape.dims.size() == 2);
    assert(model.graph.inputs[0].shape.dims[0] == 1);
    assert(model.graph.inputs[0].shape.dims[1] == 2);
    std::cout << "inputs: " << model.graph.inputs.size()
              << " shape=[" << model.graph.inputs[0].shape.dims[0]
              << "," << model.graph.inputs[0].shape.dims[1] << "] OK\n";

    // Outputs
    assert(model.graph.outputs.size() == 1);
    assert(model.graph.outputs[0].name == "Y");
    assert(model.graph.outputs[0].elem_type == onnx::FLOAT);
    assert(model.graph.outputs[0].shape.dims.size() == 2);
    assert(model.graph.outputs[0].shape.dims[0] == 1);
    assert(model.graph.outputs[0].shape.dims[1] == 2);
    std::cout << "outputs: " << model.graph.outputs.size()
              << " shape=[" << model.graph.outputs[0].shape.dims[0]
              << "," << model.graph.outputs[0].shape.dims[1] << "] OK\n";

    // Initializers
    assert(model.graph.initializers.size() == 2);
    auto& W = model.graph.initializers[0];
    assert(W.name == "W");
    assert(W.data_type == onnx::FLOAT);
    assert(W.dims.size() == 2);
    assert(W.dims[0] == 2);
    assert(W.dims[1] == 2);
    assert(W.raw_data != nullptr);
    assert(W.raw_data_size == 16);

    auto& B = model.graph.initializers[1];
    assert(B.name == "B");
    assert(B.data_type == onnx::FLOAT);
    assert(B.dims.size() == 1);
    assert(B.dims[0] == 2);
    assert(B.raw_data_size == 8);

    // Verify raw data content
    float w_vals[4];
    std::memcpy(w_vals, W.raw_data, 16);
    assert(w_vals[0] == 1.0f);
    assert(w_vals[1] == 0.0f);
    assert(w_vals[2] == 0.0f);
    assert(w_vals[3] == 1.0f);

    float b_vals[2];
    std::memcpy(b_vals, B.raw_data, 8);
    assert(b_vals[0] == 0.5f);
    assert(b_vals[1] == -0.5f);

    std::cout << "initializers: " << model.graph.initializers.size() << " with correct shapes and data OK\n";

    // Cleanup
    std::remove(path.c_str());

    std::cout << "All ONNX loader tests passed!\n";
    return 0;
}
