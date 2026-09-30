// Native ORT reference for original stateful ONNX graphs. State outputs stay on
// WebGPU between calls; only final logits are copied to CPU for greedy sampling.
#include <onnxruntime_cxx_api.h>
#include "json_parser.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <unordered_map>
#ifdef _WIN32
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#endif

namespace fs = std::filesystem;
static void logNativeMemory(const char* stage) {
#ifdef _WIN32
    IDXGIFactory1* factory=nullptr;
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))return;
    for(UINT index=0;;++index){
        IDXGIAdapter1* adapter=nullptr;if(factory->EnumAdapters1(index,&adapter)!=S_OK)break;
        DXGI_ADAPTER_DESC1 description{};adapter->GetDesc1(&description);
        if(std::wstring(description.Description).find(L"RTX 5080")!=std::wstring::npos){
            IDXGIAdapter3* budget=nullptr;
            if(SUCCEEDED(adapter->QueryInterface(IID_PPV_ARGS(&budget)))){
                DXGI_QUERY_VIDEO_MEMORY_INFO local{},shared{};
                if(SUCCEEDED(budget->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&local)) &&
                   SUCCEEDED(budget->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&shared)))
                    std::cerr<<"[native-dxgi] "<<stage<<" local="<<local.CurrentUsage
                             <<" nonlocal="<<shared.CurrentUsage<<" budget="<<local.Budget<<"\n";
                budget->Release();
            }
        }
        adapter->Release();
    }
    factory->Release();
#endif
}
struct TensorData {
    ONNXTensorElementDataType type;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};
static size_t elementSize(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
    default: throw std::runtime_error("Unsupported tensor type");
    }
}
static float halfValue(uint16_t h) {
    const float sign = h & 0x8000 ? -1.0f : 1.0f;
    const int exponent = (h >> 10) & 31, mantissa = h & 1023;
    if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
    return sign * std::ldexp(float(exponent ? 1024 + mantissa : mantissa), exponent ? exponent - 25 : -24);
}
static TensorData integers(std::vector<int64_t> shape, const std::vector<int64_t>& values) {
    TensorData result{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::move(shape), std::vector<uint8_t>(values.size() * 8)};
    if (!values.empty()) std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
    return result;
}
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        const fs::path model = fs::absolute(argv[1]), output = fs::absolute(argv[3]);
        std::ifstream input(argv[2]);
        const auto request = json_parse(std::string{std::istreambuf_iterator<char>(input), {}});
        const int warmups = request.has("warmup_runs") ? request["warmup_runs"].as_int() : 0;
        const int repetitions = request.has("repetitions") ? request["repetitions"].as_int() : 1;
        const int maxTokens = request["max_new_tokens"].as_int();
        const int capacity = request["max_seq_len"].as_int();
        const bool capture = request.has("graph_capture") && request["graph_capture"].as_bool();
        const bool profile = request.has("profile_prefix");
        const bool nativeMemory = request.has("native_memory") && request["native_memory"].as_bool();
        if (warmups < 0 || repetitions < 1 || maxTokens < 1 || capacity < 1)
            throw std::runtime_error("Invalid workload counts");
        std::vector<std::vector<int64_t>> batches;
        size_t inputTokens = 0;
        for (const auto& batch : request["batches"].as_array()) {
            std::vector<int64_t> ids;
            for (const auto& id : batch.as_array()) ids.push_back(id.as_int());
            if (ids.empty()) throw std::runtime_error("Empty batch");
            inputTokens += ids.size(); batches.push_back(std::move(ids));
        }
        if (batches.empty() || inputTokens + maxTokens > size_t(capacity))
            throw std::runtime_error("Workload exceeds capacity");
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resident-reference");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(8);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* keys[] = {"dawnBackendType", "powerPreference", "enableGraphCapture", "validationMode"};
        const char* values[] = {"D3D12", "high-performance", capture ? "1" : "0", "basic"};
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(options, "WebGPU", keys, values, 4));
        if (profile) options.EnableProfiling(fs::path(request["profile_prefix"].as_string()).c_str());
        Ort::Session session(env, (model / "text.onnx").c_str(), options);
        Ort::SessionOptions embeddingOptions;
        embeddingOptions.SetIntraOpNumThreads(8);
        embeddingOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session embedding(env, (model / "embedding.onnx").c_str(), embeddingOptions);
        if(nativeMemory)logNativeMemory("loaded");
        Ort::AllocatorWithDefaultOptions allocator;
        auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::MemoryInfo gpu("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        std::vector<std::string> names, outputNames, embeddingNames;
        std::vector<TensorData> initial, embeddingInitial;
        auto metadata = [&](Ort::Session& source, auto& list, auto& tensors) {
            for (size_t i = 0; i < source.GetInputCount(); ++i) {
                list.emplace_back(source.GetInputNameAllocated(i, allocator).get());
                auto type = source.GetInputTypeInfo(i);
                auto info = type.GetTensorTypeAndShapeInfo();
                tensors.push_back({info.GetElementType(), info.GetShape(), {}});
            }
        };
        metadata(session, names, initial);
        metadata(embedding, embeddingNames, embeddingInitial);
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            outputNames.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto tensor = [&](TensorData& data) {
            return Ort::Value::CreateTensor(cpu, data.bytes.data(), data.bytes.size(),
                                           data.shape.data(), data.shape.size(), data.type);
        };
        std::unordered_map<std::string, Ort::Value> states;
        Ort::IoBinding binding(session);
        size_t position = 0, forwardCalls = 0, gpuStateChecks = 0;
        auto run = [&](const std::vector<int64_t>& ids) {
            const int64_t count = int64_t(ids.size());
            if (position + count > size_t(capacity)) throw std::runtime_error("Context overflow");
            std::vector<TensorData> embeddingData;
            for (size_t i = 0; i < embeddingNames.size(); ++i) {
                if (embeddingNames[i] == "input_ids") embeddingData.push_back(integers({1, count}, ids));
                else if (embeddingNames[i] == "image_features") {
                    auto data = embeddingInitial[i];
                    if (data.shape.size() != 2 || data.shape[1] <= 0)
                        throw std::runtime_error("Unsupported image feature shape");
                    data.shape[0] = 0; embeddingData.push_back(std::move(data));
                } else throw std::runtime_error("Unknown embedding input");
            }
            std::vector<Ort::Value> embeddingInputs;
            std::vector<const char*> embeddingRawNames;
            for (size_t i = 0; i < embeddingData.size(); ++i) {
                embeddingInputs.push_back(tensor(embeddingData[i]));
                embeddingRawNames.push_back(embeddingNames[i].c_str());
            }
            const char* embeddingOutput = "inputs_embeds";
            auto embedded = embedding.Run(Ort::RunOptions{nullptr}, embeddingRawNames.data(),
                embeddingInputs.data(), embeddingInputs.size(), &embeddingOutput, 1);
            binding.ClearBoundInputs(); binding.ClearBoundOutputs();
            std::vector<TensorData> storage; storage.reserve(names.size());
            std::vector<Ort::Value> inputs; inputs.reserve(names.size());
            for (size_t i = 0; i < names.size(); ++i) {
                const auto& name = names[i];
                if (name == "inputs_embeds") { binding.BindInput(name.c_str(), embedded[0]); continue; }
                auto found = states.find(name);
                if (found != states.end()) { binding.BindInput(name.c_str(), found->second); continue; }
                if (name == "attention_mask")
                    storage.push_back(integers({1, int64_t(position) + count}, std::vector<int64_t>(position + count, 1)));
                else if (name == "position_ids") {
                    const int64_t axes = initial[i].shape.size() == 3 ? initial[i].shape[0] : 1;
                    std::vector<int64_t> positions(size_t(axes * count));
                    for (int64_t a = 0; a < axes; ++a) for (int64_t t = 0; t < count; ++t)
                        positions[size_t(a * count + t)] = int64_t(position) + t;
                    storage.push_back(integers(axes == 1 ? std::vector<int64_t>{1, count} :
                                              std::vector<int64_t>{axes, 1, count}, positions));
                } else if (name.rfind("past", 0) == 0) {
                    auto data = initial[i];
                    if (request.has("state_shapes") && request["state_shapes"].has(name)) {
                        data.shape.clear();
                        for (const auto& dim : request["state_shapes"][name].as_array()) data.shape.push_back(dim.as_int());
                    }
                    const bool kv = name.find(".key") != std::string::npos || name.find(".value") != std::string::npos;
                    size_t elements = 1;
                    for (size_t d = 0; d < data.shape.size(); ++d) {
                        if (data.shape[d] < 0) data.shape[d] = kv && d == 2 ? 0 : 1;
                        elements *= size_t(data.shape[d]);
                    }
                    data.bytes.resize(elements * elementSize(data.type), 0);
                    storage.push_back(std::move(data));
                } else throw std::runtime_error("Unknown decoder input: " + name);
                inputs.push_back(tensor(storage.back())); binding.BindInput(name.c_str(), inputs.back());
            }
            for (const auto& name : outputNames) binding.BindOutput(name.c_str(), name == "logits" ? cpu : gpu);
            session.Run(Ort::RunOptions{nullptr}, binding);
            binding.SynchronizeOutputs();
            auto outputs = binding.GetOutputValues();
            int32_t next = -1; float best = -INFINITY;
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto& name = outputNames[i];
                auto memory = outputs[i].GetTensorMemoryInfo();
                if (name == "logits") {
                    if (memory.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) throw std::runtime_error("Logits not on CPU");
                    auto info = outputs[i].GetTensorTypeAndShapeInfo();
                    const auto type = info.GetElementType();
                    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                        throw std::runtime_error("Unsupported logits type");
                    const size_t vocab = size_t(info.GetShape().back()), offset = info.GetElementCount() - vocab;
                    const auto* row = static_cast<const uint8_t*>(outputs[i].GetTensorRawData()) + offset * elementSize(type);
                    for (size_t j = 0; j < vocab; ++j) {
                        const float value = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ?
                            reinterpret_cast<const float*>(row)[j] : halfValue(reinterpret_cast<const uint16_t*>(row)[j]);
                        if (!std::isfinite(value)) throw std::runtime_error("Non-finite logits");
                        if (value > best) { best = value; next = int32_t(j); }
                    }
                } else {
                    if (name.rfind("present.", 0) != 0 || memory.GetDeviceType() != OrtMemoryInfoDeviceType_GPU)
                        throw std::runtime_error("State output is not a GPU tensor: " + name);
                    std::string past = "past_key_values." + name.substr(8);
                    if (std::find(names.begin(), names.end(), past) == names.end())
                        throw std::runtime_error("Unknown state output: " + name);
                    states.insert_or_assign(past, std::move(outputs[i])); ++gpuStateChecks;
                }
            }
            position += ids.size(); ++forwardCalls;
            if (next < 0) throw std::runtime_error("No prediction");
            return next;
        };
        using Clock = std::chrono::steady_clock;
        struct Result { std::vector<int32_t> tokens; double prefillMs, decodeMs; size_t calls, position; };
        std::vector<Result> results;
        std::vector<int32_t> expected;
        for (int repetition = 0; repetition < warmups + repetitions; ++repetition) {
            binding.ClearBoundInputs(); binding.ClearBoundOutputs(); states.clear(); position = 0; forwardCalls = 0;
            const auto start = Clock::now();
            int32_t next = -1;
            for (const auto& batch : batches) next = run(batch);
            const auto first = Clock::now();
            if(nativeMemory)logNativeMemory("prefilled");
            Result result; result.tokens.reserve(maxTokens);
            for (int i = 0; i < maxTokens; ++i) {
                result.tokens.push_back(next);
                bool stop = false;
                for (const auto& id : request["stop_ids"].as_array()) if (id.as_int() == next) stop = true;
                if (stop || i + 1 == maxTokens) break;
                next = run({next});
                if(nativeMemory && i==7)logNativeMemory("decode8");
            }
            const auto end = Clock::now();
            if(nativeMemory)logNativeMemory("generated");
            result.prefillMs = std::chrono::duration<double, std::milli>(first - start).count();
            result.decodeMs = std::chrono::duration<double, std::milli>(end - first).count();
            result.calls = forwardCalls; result.position = position;
            if (result.calls != batches.size() + result.tokens.size() - 1 ||
                result.position != inputTokens + result.tokens.size() - 1)
                throw std::runtime_error("Unexpected workload counts");
            if (repetition && result.tokens != expected) throw std::runtime_error("Continuation changed after reset");
            expected = result.tokens;
            std::cerr << "[resident-reference] repetition=" << repetition << " tokens=" << result.tokens.size()
                      << " prefill_ms=" << result.prefillMs << " decode_ms=" << result.decodeMs << "\n";
            if (repetition >= warmups) results.push_back(std::move(result));
        }
        std::ofstream file(output);
        file << std::setprecision(12) << "{\"ort_version\":" << std::quoted(Ort::GetVersionString())
             << ",\"memory_diagnostics\":" << (nativeMemory ? "true" : "false")
             << ",\"graph_capture\":" << (capture ? "true" : "false") << ",\"profiling\":" << (profile ? "true" : "false")
             << ",\"warmup_runs\":" << warmups << ",\"gpu_state_checks\":" << gpuStateChecks
             << ",\"input_tokens\":" << inputTokens << ",\"max_seq_len\":" << capacity << ",\"prefill_batches\":" << batches.size()
             << ",\"sampling\":\"CPU argmax of CPU-bound logits\",\"runs\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            file << (i ? "," : "") << "{\"prefill_ms\":" << r.prefillMs << ",\"decode_ms\":" << r.decodeMs
                 << ",\"prefill_tok_s\":" << inputTokens * 1000.0 / r.prefillMs
                 << ",\"decode_tok_s\":" << (r.tokens.size() > 1 ? (r.tokens.size() - 1) * 1000.0 / r.decodeMs : 0)
                 << ",\"forward_calls\":" << r.calls << ",\"final_position\":" << r.position << ",\"tokens\":[";
            for (size_t j = 0; j < r.tokens.size(); ++j) file << (j ? "," : "") << r.tokens[j];
            file << "]}";
        }
        file << "]}\n";
        if (!file) throw std::runtime_error("Cannot write result");
        if (profile) session.EndProfilingAllocated(allocator);
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
}
