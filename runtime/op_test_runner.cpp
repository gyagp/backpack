/**
 * op_test_runner.cpp — Run a single ONNX model through GraphExecutor
 * and dump output tensors as binary files for comparison.
 *
 * Usage:
 *   op_test_runner <model.onnx> <input_dir> <output_dir>
 *
 * Input files:  <input_dir>/<tensor_name>.bin  (raw bytes, little-endian)
 *               <input_dir>/manifest.json      (shapes + dtypes)
 * Output files: <output_dir>/<tensor_name>.bin  (raw bytes)
 *               <output_dir>/manifest.json      (shapes + dtypes)
 */

#include "gpu_context.h"
#include "graph_executor.h"
#include "json_parser.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static TensorDtype parseDtype(const std::string& s) {
    if (s == "float32") return TensorDtype::Float32;
    if (s == "float16") return TensorDtype::Float16;
    if (s == "int32")   return TensorDtype::Int32;
    if (s == "int64")   return TensorDtype::Int64;
    if (s == "uint8")   return TensorDtype::UInt8;
    if (s == "bool")    return TensorDtype::Bool;
    return TensorDtype::Float32;
}

static const char* dtypeStr(TensorDtype d) {
    switch (d) {
        case TensorDtype::Float32: return "float32";
        case TensorDtype::Float16: return "float16";
        case TensorDtype::Int32:   return "int32";
        case TensorDtype::Int64:   return "int64";
        case TensorDtype::UInt8:   return "uint8";
        case TensorDtype::Int8:    return "int8";
        case TensorDtype::Bool:    return "bool";
    }
    return "float32";
}

static size_t dtypeSize(TensorDtype d) {
    switch (d) {
        case TensorDtype::Float32: case TensorDtype::Int32: return 4;
        case TensorDtype::Float16: return 2;
        case TensorDtype::Int64: return 8;
        case TensorDtype::UInt8: case TensorDtype::Int8: case TensorDtype::Bool: return 1;
    }
    return 4;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.onnx> <input_dir> <output_dir>\n", argv[0]);
        return 1;
    }

    std::string modelPath = argv[1];
    std::string inputDir = argv[2];
    std::string outputDir = argv[3];

    // Create output dir
    fs::create_directories(outputDir);

    // 1. Init GPU
    GPUContext gpu;
    if (!gpu.init(WGPUBackendType_Vulkan)) {
        // Fallback to D3D12
        if (!gpu.init(WGPUBackendType_D3D12)) {
            fprintf(stderr, "Failed to init GPU\n");
            return 1;
        }
    }

    // 2. Load model
    GraphExecutor executor;
    if (!executor.Load(gpu, modelPath)) {
        fprintf(stderr, "Failed to load: %s\n", modelPath.c_str());
        return 1;
    }

    // 3. Read input manifest
    std::string manifestPath = (fs::path(inputDir) / "manifest.json").string();
    std::ifstream mf(manifestPath);
    if (!mf.is_open()) {
        fprintf(stderr, "Cannot open: %s\n", manifestPath.c_str());
        return 1;
    }
    std::string mfStr{std::istreambuf_iterator<char>(mf),
                       std::istreambuf_iterator<char>()};
    mf.close();
    auto manifest = json_parse(mfStr);

    // 4. Build input tensors
    std::unordered_map<std::string, GpuTensor> inputTensors;
    std::unordered_map<std::string, GpuTensor*> inputs;

    auto& inputs_json = manifest["inputs"];
    for (int64_t i = 0; i < inputs_json.size(); i++) {
        auto& entry = inputs_json[i];
        std::string name = entry["name"].as_string();
        TensorDtype dtype = parseDtype(entry["dtype"].as_string());

        std::vector<int64_t> shape;
        auto& shape_json = entry["shape"];
        for (int64_t j = 0; j < shape_json.size(); j++)
            shape.push_back(shape_json[j].as_int());

        // Read binary data
        std::string binPath = (fs::path(inputDir) / (name + ".bin")).string();
        // Replace special chars in filename
        for (auto& c : binPath) if (c == '/' || c == ':') c = '_';
        binPath = (fs::path(inputDir) / (name + ".bin")).string();

        std::ifstream bin(binPath, std::ios::binary);
        if (!bin.is_open()) {
            fprintf(stderr, "Cannot open input: %s\n", binPath.c_str());
            // Create zero tensor
            int64_t nel = 1;
            for (auto d : shape) nel *= d;
            size_t bytes = nel * dtypeSize(dtype);
            if (bytes == 0) bytes = 4;
            std::vector<uint8_t> zeros(bytes, 0);
            auto& t = inputTensors[name];
            t.shape = shape;
            t.dtype = dtype;
            t.buffer = gpu.createBuffer(name, bytes);
            gpu.writeBuffer(t.buffer, zeros.data(), bytes);
            t.cpuData = zeros;
            inputs[name] = &inputTensors[name];
            continue;
        }

        bin.seekg(0, std::ios::end);
        size_t fileSize = (size_t)bin.tellg();
        bin.seekg(0);
        std::vector<uint8_t> data(fileSize);
        bin.read(reinterpret_cast<char*>(data.data()), fileSize);
        bin.close();

        auto& t = inputTensors[name];
        t.shape = shape;
        t.dtype = dtype;
        size_t bufSize = fileSize > 0 ? fileSize : 4;
        t.buffer = gpu.createBuffer(name, bufSize);
        if (fileSize > 0)
            gpu.writeBuffer(t.buffer, data.data(), fileSize);
        t.cpuData = data;
        inputs[name] = &inputTensors[name];
    }

    // 5. Prepare output placeholders
    std::unordered_map<std::string, GpuTensor> outputTensors;
    std::unordered_map<std::string, GpuTensor*> outputs;
    std::vector<std::string> outputNames;

    auto& graph = executor.GetGraph();
    for (auto& out : graph.outputs) {
        outputNames.push_back(out.name);
        auto& t = outputTensors[out.name];
        t.shape = out.shape;
        t.dtype = out.dtype;
        // Estimate output size (may be overridden by Execute)
        int64_t nel = 1;
        for (auto d : out.shape) nel *= std::max<int64_t>(d, 1);
        size_t bytes = nel * dtypeSize(out.dtype);
        if (bytes == 0) bytes = 4;
        t.buffer = gpu.createBuffer(out.name + "_out", bytes);
        outputs[out.name] = &outputTensors[out.name];
    }

    // 6. Execute
    auto t0 = std::chrono::steady_clock::now();
    executor.Execute(inputs, outputs);
    executor.FlushPendingWork();
    gpu.waitForQueue();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Execute: %.1fms\n", ms);

    // Debug: check output tensor states
    for (auto& name : outputNames) {
        auto* t = outputs[name];
        if (!t) { fprintf(stderr, "  out '%s': null ptr\n", name.c_str()); continue; }
        fprintf(stderr, "  out '%s': valid=%d buf=%p bufsize=%llu dtype=%d shape=[",
                name.c_str(), t->IsValid(), (void*)t->buffer.handle,
                (unsigned long long)t->buffer.size, (int)t->dtype);
        for (auto d : t->shape) fprintf(stderr, "%lld,", (long long)d);
        fprintf(stderr, "] cpu=%d nel=%lld\n", t->isCpuOnly,
                (long long)t->ElementCount());
    }

    // 7. Write outputs
    std::string outManifest = "{\n  \"outputs\": [\n";
    for (size_t i = 0; i < outputNames.size(); i++) {
        auto& name = outputNames[i];
        auto* t = outputs[name];
        if (!t || !t->IsValid()) {
            fprintf(stderr, "Output '%s' not valid\n", name.c_str());
            continue;
        }

        // Read back data
        int64_t nel = t->ElementCount();
        size_t bytes = nel * dtypeSize(t->dtype);
        if (bytes == 0) bytes = 4;

        std::vector<uint8_t> data;
        if (t->isCpuOnly && !t->cpuData.empty()) {
            data = t->cpuData;
        } else if (t->buffer.handle) {
            auto rb = gpu.readBuffer(t->buffer, bytes);
            data.assign(rb.begin(), rb.end());
        }

        // Write binary
        std::string binPath = (fs::path(outputDir) / (name + ".bin")).string();
        std::ofstream bin(binPath, std::ios::binary);
        if (bin.is_open() && !data.empty()) {
            bin.write(reinterpret_cast<const char*>(data.data()), data.size());
            bin.close();
        }

        // Manifest entry
        outManifest += "    {\"name\": \"" + name + "\", \"dtype\": \"" +
                       dtypeStr(t->dtype) + "\", \"shape\": [";
        for (size_t j = 0; j < t->shape.size(); j++) {
            if (j > 0) outManifest += ", ";
            outManifest += std::to_string(t->shape[j]);
        }
        outManifest += "]}";
        if (i + 1 < outputNames.size()) outManifest += ",";
        outManifest += "\n";
    }
    outManifest += "  ]\n}\n";

    std::ofstream omf(fs::path(outputDir) / "manifest.json");
    omf << outManifest;
    omf.close();

    printf("OK\n");
    return 0;
}
