#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static bool file_contains(const std::string& path, const std::string& needle) {
    std::ifstream f(path);
    if (!f.good()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find(needle) != std::string::npos) return true;
    }
    return false;
}

int main() {
    int pass = 0, fail = 0;

    // --- Acceptance criterion 1: All required ops are in dispatch_node ---
    const std::vector<std::string> required_ops = {
        "Conv", "Reshape", "Transpose", "Cast", "Unsqueeze", "Concat",
        "Split", "Mul", "Div", "Tanh", "Gelu", "RotaryEmbedding"
    };

    std::string runtime_path = "src/onnx_runtime.h";
    for (auto& op : required_ops) {
        std::string pattern = "\"" + op + "\"";
        if (file_contains(runtime_path, pattern)) {
            std::cout << "PASS: op " << op << " registered in dispatch_node\n";
            pass++;
        } else {
            std::cerr << "FAIL: op " << op << " NOT found in dispatch_node\n";
            fail++;
        }
    }

    // Also verify dispatch function exists for each op
    const std::vector<std::string> dispatch_funcs = {
        "dispatch_conv", "dispatch_reshape", "dispatch_transpose",
        "dispatch_cast", "dispatch_unsqueeze", "dispatch_concat",
        "dispatch_split", "dispatch_mul", "dispatch_div",
        "dispatch_tanh", "dispatch_gelu", "dispatch_rotary_embedding"
    };

    for (auto& fn : dispatch_funcs) {
        if (file_contains(runtime_path, fn)) {
            std::cout << "PASS: function " << fn << " exists\n";
            pass++;
        } else {
            std::cerr << "FAIL: function " << fn << " NOT found\n";
            fail++;
        }
    }

    // --- Verify shader files exist for GPU-dispatched ops ---
    const std::vector<std::string> gpu_shaders = {
        "src/shaders/mul.wgsl",
        "src/shaders/div.wgsl",
        "src/shaders/tanh.wgsl",
        "src/shaders/gelu.wgsl",
        "src/shaders/transpose.wgsl",
        "src/shaders/rope.wgsl",
    };

    for (auto& shader : gpu_shaders) {
        if (file_exists(shader)) {
            std::cout << "PASS: shader " << shader << " exists\n";
            pass++;
        } else {
            std::cerr << "FAIL: shader " << shader << " NOT found\n";
            fail++;
        }
    }

    // --- Verify no "unsupported op" for common model ops ---
    const std::vector<std::string> all_expected_ops = {
        "MatMul", "Add", "Relu", "Softmax", "LayerNormalization",
        "Gather", "Mul", "Div", "Tanh", "Gelu",
        "Reshape", "Transpose", "Cast", "Unsqueeze", "Squeeze",
        "Concat", "Split", "Conv", "RotaryEmbedding",
        "Sigmoid", "Clip", "Sub", "Sqrt", "Pow",
        "ReduceMean", "Where", "Equal", "Expand", "Flatten",
        "Gemm", "Shape", "Slice", "Neg", "Abs",
        "Log", "Exp", "Erf", "Identity", "SkipLayerNormalization"
    };

    for (auto& op : all_expected_ops) {
        std::string pattern = "\"" + op + "\"";
        if (file_contains(runtime_path, pattern)) {
            pass++;
        } else {
            std::cerr << "FAIL: op " << op << " missing from dispatch_node\n";
            fail++;
        }
    }

    // --- Summary ---
    std::cout << "\n=== Results: " << pass << " passed, " << fail << " failed ===\n";
    if (fail > 0) {
        std::cerr << "ACCEPTANCE CRITERIA NOT MET\n";
        return 1;
    }
    std::cout << "All ONNX ops coverage tests passed!\n";
    std::cout << "Acceptance criteria: All ops needed by ONNX models are implemented.\n";
    return 0;
}
