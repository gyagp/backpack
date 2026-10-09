/**
 * lm_session.cpp — Implementation of bp::LmSession (Layer 2 LM API).
 *
 * Moves OnnxLlmContext (GenericOnnx) and LlmContext (Standard) from
 * apps/llm/main.cpp into the runtime as internal state of LmSession::Impl.
 */

#include "lm_session.h"

// Internal headers (not exposed in the public API)
#include "gpu_context.h"
#include "model_runner.h"
#include "execution_context.h"
#include "graph_executor.h"
#include "tokenizer.h"
#include "onnx_tokenizer.h"
#include <wgsl_shaders.h>
#include "json_parser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>

namespace fs = std::filesystem;

namespace bp {

// ═══════════════════════════════════════════════════════════════════════════
// Path Resolution (moved from apps/llm/main.cpp)
// ═══════════════════════════════════════════════════════════════════════════

static bool isStandardOnnxDir(const std::string& path) {
    if (!fs::is_directory(path)) return false;
    if (!fs::exists(fs::path(path) / "config.json")) return false;
    if (!fs::exists(fs::path(path) / "tokenizer.json")) return false;
    for (auto& e : fs::directory_iterator(path))
        if (e.is_regular_file() && e.path().extension() == ".onnx") return true;
    return false;
}

static bool isNonStandardArch(const std::string& path) {
    std::string cfgPath = (fs::path(path) / "config.json").string();
    if (!fs::exists(cfgPath)) return false;
    std::ifstream f(cfgPath);
    std::string s{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    if (std::getenv("BP_QWEN35_SPECIALIZED_ONNX") &&
        (s.find("\"qwen3_5_text\"") != std::string::npos ||
         s.find("\"qwen3_5\"") != std::string::npos))
        return false;
    return s.find("\"layer_types\"") != std::string::npos ||
           s.find("\"conv_L_cache\"") != std::string::npos;
}

static std::string findOnnxFile(const std::string& dir) {
    const auto manifest = fs::path(dir) / "genai_config.json";
    if (fs::exists(manifest)) {
        std::ifstream input(manifest);
        const auto config = json_parse(std::string{std::istreambuf_iterator<char>(input), {}});
        if (config.has("model") && config["model"].has("decoder") && config["model"]["decoder"].has("filename")) {
            const auto decoder = fs::path(dir) / config["model"]["decoder"]["filename"].as_string();
            if (!fs::exists(decoder)) throw std::runtime_error("Missing GenAI decoder: " + decoder.string());
            return decoder.string();
        }
    }
    std::string best;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".onnx") continue;
        auto name = e.path().filename().string();
        if (name.find("q4f16") != std::string::npos) return e.path().string();
        if (name.find("q4") != std::string::npos) best = e.path().string();
        if (best.empty()) best = e.path().string();
    }
    return best;
}

static std::string findGenaiDecoderDir(const std::string& dir) {
    fs::path root(dir);
    if (!fs::exists(root / "genai_config.json")) return {};
    fs::path decoder = root / "decoder";
    if (fs::exists(decoder / "model.onnx")) return decoder.string();
    return {};
}

static std::string resolvePath(const std::string& path, std::string& format,
                               const std::string& formatOverride = "") {
    if (!formatOverride.empty()) {
        if (formatOverride == "gguf") {
            format = "gguf";
            if (fs::is_directory(path)) {
                for (auto& e : fs::recursive_directory_iterator(path))
                    if (e.is_regular_file() && e.path().extension() == ".gguf")
                        return e.path().string();
            }
            return path;
        }
        if (formatOverride == "onnx") {
            if (fs::is_directory(path)) {
                if (auto decoder = findGenaiDecoderDir(path); !decoder.empty()) {
                    format = "onnx";
                    return decoder;
                }
                format = isNonStandardArch(path) ? "onnx_generic" : "onnx";
                if (format == "onnx_generic") return findOnnxFile(path);
                return path;
            }
            std::string dir = fs::path(path).parent_path().string();
            format = isNonStandardArch(dir) ? "onnx_generic" : "onnx";
            return format == "onnx" ? dir : path;
        }
    }
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf") {
        if (fs::exists(path)) { format = "gguf"; return path; }
    }
    if (path.size() > 5 && path.substr(path.size() - 5) == ".onnx") {
        if (fs::exists(path)) {
            std::string dir = fs::path(path).parent_path().string();
            format = isNonStandardArch(dir) ? "onnx_generic" : "onnx";
            return format == "onnx" ? dir : path;
        }
    }
    if (fs::is_directory(path)) {
        if (auto decoder = findGenaiDecoderDir(path); !decoder.empty()) {
            format = "onnx";
            return decoder;
        }
        if (isStandardOnnxDir(path)) {
            format = isNonStandardArch(path) ? "onnx_generic" : "onnx";
            if (format == "onnx_generic") return findOnnxFile(path);
            return path;
        }
        for (auto& e : fs::directory_iterator(path)) {
            if (!e.is_directory()) continue;
            if (isStandardOnnxDir(e.path().string())) {
                format = isNonStandardArch(e.path().string()) ? "onnx_generic" : "onnx";
                if (format == "onnx_generic") return findOnnxFile(e.path().string());
                return e.path().string();
            }
        }
        std::string best;
        for (auto& e : fs::recursive_directory_iterator(path))
            if (e.is_regular_file() && e.path().extension() == ".gguf")
                { best = e.path().string(); if (best.find("Q8_0") != std::string::npos) break; }
        if (!best.empty()) { format = "gguf"; return best; }
    }
    format = "gguf";
    return path;
}

// ═══════════════════════════════════════════════════════════════════════════
// Sampling (internal)
// ═══════════════════════════════════════════════════════════════════════════

static int32_t sampleToken(const float* logits, uint32_t vocabSize,
                           float temperature, int topK, std::mt19937& rng) {
    if (temperature <= 0.0f)
        return (int32_t)(std::max_element(logits, logits + vocabSize) - logits);

    if (topK > 0 && topK < (int)vocabSize) {
        int k = topK;
        std::vector<int32_t> indices(vocabSize);
        for (uint32_t i = 0; i < vocabSize; i++) indices[i] = (int32_t)i;
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        std::vector<float> probs(k);
        float maxVal = logits[indices[0]];
        for (int i = 0; i < k; i++)
            probs[i] = logits[indices[i]] / temperature - maxVal / temperature;
        float sum = 0.0f;
        for (int i = 0; i < k; i++) { probs[i] = std::exp(probs[i]); sum += probs[i]; }
        for (int i = 0; i < k; i++) probs[i] /= sum;
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return indices[dist(rng)];
    }

    std::vector<float> probs(vocabSize);
    float maxVal = *std::max_element(logits, logits + vocabSize);
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocabSize; i++) {
        probs[i] = std::exp((logits[i] - maxVal) / temperature);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < vocabSize; i++) probs[i] /= sum;
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

static int32_t argmax(const float* logits, int64_t n) {
    float mx = -1e30f;
    int32_t idx = 0;
    for (int64_t i = 0; i < n; i++) {
        if (logits[i] > mx) { mx = logits[i]; idx = (int32_t)i; }
    }
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════
// GenericOnnxState (moved from OnnxLlmContext in apps/llm/main.cpp)
// ═══════════════════════════════════════════════════════════════════════════

struct GenericOnnxState {
    GPUContext* gpu = nullptr;
    GraphExecutor executor;
    ExecutionContext execCtx;
    OnnxTokenizer tokenizer;
    std::string modelPath;
    std::string modelDir;
    GraphExecutor embeddingMetadata;
    const OnnxInitData* cpuEmbedding = nullptr;
    GPUBuffer embeddingsBuf;
    std::vector<int64_t> mediaTokenIds;
    TensorDtype logitsDtype = TensorDtype::Float32;

    void LoadCpuEmbedding() {
        const auto& inputs = executor.GetGraph().inputs;
        const auto found = std::find_if(inputs.begin(), inputs.end(), [](const auto& input) { return input.name == "inputs_embeds"; });
        if (found == inputs.end()) return;
        std::ifstream input(fs::path(modelDir) / "genai_config.json");
        if (!input) throw std::runtime_error("inputs_embeds requires a GenAI embedding manifest");
        const auto config = json_parse(std::string{std::istreambuf_iterator<char>(input), {}});
        const auto& model = config["model"];
        const auto file = fs::path(modelDir) / model["embedding"]["filename"].as_string();
        if (!embeddingMetadata.Load(*gpu, file.string(), false)) throw std::runtime_error("Cannot read embedding metadata");
        const OnnxGraphNode* gather = nullptr;
        for (const auto& node : embeddingMetadata.GetGraph().nodes) {
            if (node.opType == "Gather" && node.inputs.size() >= 2 && node.inputs[1] == "input_ids" &&
                node.GetInt("axis", 0) == 0 && embeddingMetadata.GetInitData(node.inputs[0])) {
                if (gather) throw std::runtime_error("Ambiguous CPU token embedding graph");
                gather = &node;
            }
        }
        if (!gather || gather->outputs.size() != 1) throw std::runtime_error("Unsupported CPU embedding graph");
        // The supported text-only graph is Gather -> ScatterND(media). With no
        // media placeholders, ScatterND has zero indices and leaves Gather intact.
        bool outputMatches = false;
        const auto& graph = embeddingMetadata.GetGraph();
        if (graph.outputs.size() != 1) throw std::runtime_error("CPU embedding requires one output");
        for (const auto& output : graph.outputs) {
            if (output.name == gather->outputs[0]) outputMatches = true;
            for (const auto& node : graph.nodes) {
                if (node.opType == "ScatterND" && node.inputs.size() == 3 && node.inputs[0] == gather->outputs[0] &&
                    !node.outputs.empty() && node.outputs[0] == output.name) {
                    auto producer = [&](const std::string& name) -> const OnnxGraphNode* {
                        for (const auto& candidate : graph.nodes)
                            if (std::find(candidate.outputs.begin(), candidate.outputs.end(), name) != candidate.outputs.end()) return &candidate;
                        return nullptr;
                    };
                    const auto* transpose = producer(node.inputs[1]);
                    if (!transpose || transpose->opType != "Transpose" || transpose->inputs.size() != 1) continue;
                    const auto permutation = transpose->attrIntLists.find("perm");
                    if (permutation != transpose->attrIntLists.end() && permutation->second != std::vector<int64_t>{1,0}) continue;
                    const auto* condition = producer(transpose->inputs[0]);
                    if (!condition || condition->opType != "NonZero" || condition->inputs.size() != 1) continue;
                    condition = producer(condition->inputs[0]);
                    size_t depth = 0;
                    while (condition && !condition->inputs.empty() && depth++ < graph.nodes.size() &&
                           (condition->opType == "Expand" || condition->opType == "Unsqueeze" || condition->opType == "Reshape" || condition->opType == "Identity"))
                        condition = producer(condition->inputs[0]);
                    if (!condition || condition->opType != "Equal" || condition->inputs.size() != 2) continue;
                    const auto constantName = condition->inputs[0] == "input_ids" ? condition->inputs[1] :
                        condition->inputs[1] == "input_ids" ? condition->inputs[0] : std::string{};
                    auto* constant = embeddingMetadata.GetWeightTensor(constantName);
                    if (!constant || constant->ElementCount() != 1) continue;
                    int64_t mediaToken = -1;
                    if (constant->dtype == TensorDtype::Int64 && constant->cpuData.size() == 8)
                        memcpy(&mediaToken, constant->cpuData.data(), 8);
                    else if (constant->dtype == TensorDtype::Int32 && constant->cpuData.size() == 4) {
                        int32_t value; memcpy(&value, constant->cpuData.data(), 4); mediaToken = value;
                    }
                    if (mediaToken < 0 || mediaToken >= vocabSize) continue;
                    mediaTokenIds.push_back(mediaToken);
                    outputMatches = true;
                }
            }
        }
        if (!outputMatches) throw std::runtime_error("Embedding graph transforms token values after Gather");
        cpuEmbedding = embeddingMetadata.GetInitData(gather->inputs[0]);
        if (!cpuEmbedding || !cpuEmbedding->data || cpuEmbedding->shape != std::vector<int64_t>{vocabSize, hiddenSize} ||
            cpuEmbedding->dtype != found->dtype ||
            (cpuEmbedding->dtype != TensorDtype::Float16 && cpuEmbedding->dtype != TensorDtype::Float32) ||
            cpuEmbedding->size != uint64_t(vocabSize) * hiddenSize * GpuTensor::DtypeSizeOf(cpuEmbedding->dtype))
            throw std::runtime_error("Invalid CPU embedding table");
        for (const char* name : {"image_token_id", "video_token_id"})
            if (model.has(name)) mediaTokenIds.push_back(model[name].as_int());
        fprintf(stderr, "  CPU-mapped embedding: %lld x %lld, %.1f MiB; text rows uploaded on demand\n",
                static_cast<long long>(vocabSize), static_cast<long long>(hiddenSize), cpuEmbedding->size / 1048576.0);
    }

    void PrepareEmbeddings(const int32_t* tokens, uint32_t count, GpuTensor& tensor) {
        if (!cpuEmbedding) return;
        const size_t rowBytes = hiddenSize * GpuTensor::DtypeSizeOf(cpuEmbedding->dtype);
        std::vector<uint8_t> rows(size_t(count) * rowBytes);
        for (uint32_t i = 0; i < count; ++i) {
            if (tokens[i] < 0 || tokens[i] >= vocabSize ||
                std::find(mediaTokenIds.begin(), mediaTokenIds.end(), tokens[i]) != mediaTokenIds.end())
                throw std::runtime_error("CPU embedding accepts in-range text tokens only; media inputs are unsupported");
            memcpy(rows.data() + i * rowBytes, cpuEmbedding->data + size_t(tokens[i]) * rowBytes, rowBytes);
        }
        gpu->writeBuffer(embeddingsBuf, rows.data(), rows.size());
        tensor.shape = {1, count, hiddenSize}; tensor.dtype = cpuEmbedding->dtype; tensor.buffer = embeddingsBuf;
    }

    std::vector<float> ReadLogits(uint64_t count) {
        const auto bytes = gpu->mapReadbackBuffer(count * GpuTensor::DtypeSizeOf(logitsDtype));
        std::vector<float> values(count);
        if (logitsDtype == TensorDtype::Float32) memcpy(values.data(), bytes.data(), count * sizeof(float));
        else if (logitsDtype == TensorDtype::Float16) {
            const auto* half = reinterpret_cast<const uint16_t*>(bytes.data());
            for (size_t i = 0; i < count; ++i) {
                const uint16_t bits = half[i]; const int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
                const float sign = bits & 0x8000 ? -1.0f : 1.0f;
                values[i] = exponent == 31 ? (mantissa ? NAN : sign * INFINITY) :
                    sign * std::ldexp(float(exponent ? 1024 + mantissa : mantissa), exponent ? exponent - 25 : -24);
            }
        } else throw std::runtime_error("Unsupported logits dtype");
        return values;
    }

    // Sampling reads the already computed logits without advancing model state.
    std::vector<float> CurrentLogits() {
        const uint64_t count = vocabSize;
        const auto bytes = gpu->readBuffer(logitsBuf, count * GpuTensor::DtypeSizeOf(logitsDtype));
        std::vector<float> values(count);
        if (logitsDtype == TensorDtype::Float32) memcpy(values.data(), bytes.data(), count * sizeof(float));
        else if (logitsDtype == TensorDtype::Float16) {
            const auto* half = reinterpret_cast<const uint16_t*>(bytes.data());
            for (size_t i = 0; i < count; ++i) {
                const uint16_t bits = half[i]; const int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
                const float sign = bits & 0x8000 ? -1.0f : 1.0f;
                values[i] = exponent == 31 ? (mantissa ? NAN : sign * INFINITY) :
                    sign * std::ldexp(float(exponent ? 1024 + mantissa : mantissa), exponent ? exponent - 25 : -24);
            }
        } else throw std::runtime_error("Unsupported logits dtype");
        return values;
    }

    int64_t hiddenSize = 0, numLayers = 0, vocabSize = 0;
    int64_t numHeads = 0, numKvHeads = 0, headDim = 0;
    int64_t convLCache = 3;
    int64_t maxSeqLen = 0;
    int64_t intermediateSize = 0;
    int64_t convChannels = 0;
    int64_t recurrentHeads = 0, recurrentKeyDim = 0, recurrentValueDim = 0;
    uint32_t positionAxes = 1;
    uint32_t diagnosticStep = 0;
    void DumpReferenceState() {
        const char* directory = std::getenv("BP_DUMP_ONNX_STATE_DIR");
        if (!directory || !*directory) return;
        const fs::path folder(directory);
        fs::create_directories(folder);
        std::ofstream metadata(folder / ("state-" + std::to_string(diagnosticStep) + ".json"));
        if (!metadata) throw std::runtime_error("Cannot create ONNX state metadata");
        metadata << "{";
        bool first = true;
        auto dump = [&](const std::string& name, const GpuTensor& tensor, bool lastRow = false) {
            const size_t bytes = tensor.ByteSize();
            auto data = gpu->readBuffer(tensor.buffer, (bytes + 3) & ~size_t(3));
            const size_t size = lastRow ? size_t(vocabSize) * tensor.DtypeSize() : bytes;
            if (size > bytes) throw std::runtime_error("Invalid diagnostic tensor shape");
            std::ofstream file(folder / (name + "-" + std::to_string(diagnosticStep) + ".bin"),
                               std::ios::binary);
            file.write(reinterpret_cast<const char*>(data.data() + bytes - size), size);
            if (!file) throw std::runtime_error("Cannot write ONNX state tensor");
            if (!first) metadata << ",";
            first = false;
            metadata << "\"" << name << "\":{\"type\":"
                     << (tensor.dtype == TensorDtype::Float16 ? 10 : 1)
                     << ",\"count\":" << size / tensor.DtypeSize() << "}";
        };
        // Replay uses alternating captured output buffers; tensorStore_ and the
        // ordinary cache maps no longer identify the current parity's outputs.
        const bool replay = fastDecodeCaptured && qwenActiveVariant >= 0;
        GpuTensor logits;
        logits.buffer = logitsBuf;
        logits.shape = {1, 1, vocabSize};
        logits.dtype = logitsDtype;
        dump("logits", logits);
        if (replay) {
            const auto& variant = qwenCaptureVariants[qwenActiveVariant];
            if (variant.diagnosticConv.IsValid()) dump("present.0.conv", variant.diagnosticConv);
            if (variant.diagnosticRecurrent.IsValid()) dump("present.0.recurrent", variant.diagnosticRecurrent);
        } else {
            auto conv = convState.find("past_key_values.0.conv_state");
            if (conv != convState.end()) dump("present.0.conv", conv->second);
            auto recurrent = recurrentState.find("past_key_values.0.recurrent_state");
            if (recurrent != recurrentState.end()) dump("present.0.recurrent", recurrent->second);
        }
        for (const char* kind : {"key", "value"}) {
            auto it = kvState.find(std::string("past_key_values.3.") + kind);
            if (it == kvState.end()) continue;
            GpuTensor cache = it->second;
            cache.shape = {static_cast<int64_t>(cache.buffer.size / cache.DtypeSize())};
            dump(std::string("cache.3.") + kind, cache);
        }
        metadata << "}\n";
        ++diagnosticStep;
    }
    std::unordered_map<std::string, std::string> cacheAliases;

    void ExecuteGraph(const std::unordered_map<std::string, GpuTensor*>& inputs,
                      std::unordered_map<std::string, GpuTensor*>& outputs) {
        if (cacheAliases.empty()) {
            executor.Execute(execCtx, inputs, outputs);
            if (outputs.count("logits")) logitsDtype = outputs.at("logits")->dtype;
            return;
        }
        auto rename = [&](const auto& tensors) {
            std::unordered_map<std::string, GpuTensor*> mapped;
            for (auto& [name, tensor] : tensors) {
                auto alias = cacheAliases.find(name);
                mapped[alias == cacheAliases.end() ? name : alias->second] = tensor;
            }
            return mapped;
        };
        auto mappedInputs = rename(inputs), mappedOutputs = rename(outputs);
        executor.Execute(execCtx, mappedInputs, mappedOutputs);
        if (outputs.count("logits")) logitsDtype = outputs.at("logits")->dtype;
    }
    int64_t moeIntermediateSize = 0;
    int64_t numExperts = 0, numExpertsPerTok = 0;
    float normEps = 1e-5f;
    float ropeTheta = 1000000.0f;
    std::vector<std::string> layerTypes;
    std::string arch;

    uint32_t pos = 0;
    uint32_t prefillChunkSize = 0;
    std::unordered_map<std::string, GpuTensor> convState;
    std::unordered_map<std::string, GpuTensor> recurrentState;
    std::unordered_map<std::string, GpuTensor> kvState;
    std::vector<int> convLayerIndices, attnLayerIndices;

    GPUBuffer idsBuf, prefillIdsBuf, positionBuf, maskBuf, nlkBuf, logitsBuf;
    std::vector<GPUBuffer> convOutBufs, convOutAltBufs;
    std::vector<GPUBuffer> recurrentOutBufs, recurrentOutAltBufs;
    uint32_t maskBufCapacity = 0;

    bool fastDecodeEnabled = false;
    bool fastDecodeCaptured = false;
    bool prefillDone = false;
    bool benchWarmupDone = false;
    bool decodePlanInitialized = false;
    int decodeWarmupRemaining = 0;
    bool nlkWritten = false;
    std::vector<GPUBuffer> convCastF16Bufs;
    std::vector<GPUBuffer> capturedConvOutputBufs;
    std::vector<GPUBuffer> capturedRecurrentInputBufs;
    std::vector<GPUBuffer> capturedRecurrentOutputBufs;
    std::vector<WGPUBindGroup> convCastBindGroups;
    const CompiledPipeline* convCastPipeline = nullptr;
    uint32_t convCastWorkgroups = 0;
    struct CaptureVariant {
        std::vector<ExecutionContext::CapturedFlush> flushes;
        std::vector<GPUBuffer> temporaryBuffers;
        std::vector<ExecutionContext::CapturedWrite> writes;
        std::vector<ExecutionContext::ReplayParamUpdate> params;
        std::vector<ExecutionContext::ReplayScalarUpdate> scalars;
        std::vector<ExecutionContext::CapturedTokenIdBuf> tokenIds;
        std::vector<WGPUBuffer> skipBuffers;
        GPUBuffer logits;
        GpuTensor diagnosticConv, diagnosticRecurrent;
    } qwenCaptureVariants[4];
    int qwenCapturedVariants = 0;
    int qwenActiveVariant = -1;
    int qwenNextReplayVariant = 0;
    bool requestGpuGreedyToken = false;
    int32_t lastGpuGreedyToken = -1;

    bool IsNvidiaQwenCapture() const {
        return arch == "qwen3_5_text" &&
            gpu->adapterName.find("NVIDIA") != std::string::npos;
    }

    void ClassifyNvidiaQwenCaptureWrites() {
        if (!IsNvidiaQwenCapture() || qwenCapturedVariants != 2) return;
        auto& a = qwenCaptureVariants[0].writes;
        auto& b = qwenCaptureVariants[1].writes;
        // Pairing is deliberately strict. Any graph/order/size ambiguity keeps
        // both writes on the replay path. Only parameter-pool buffers can be
        // immutable: other CPU-produced buffers may be shared and overwritten
        // between captures. Handles differ because the two captures
        // intentionally own disjoint parameter-pool ranges.
        if (a.size() == b.size()) {
            for (size_t i = 0; i < a.size(); ++i) {
                const bool sameSite = a[i].opName == b[i].opName &&
                    a[i].offset == b[i].offset &&
                    a[i].data.size() == b[i].data.size() &&
                    execCtx.IsParamPoolBuffer(a[i].handle) &&
                    execCtx.IsParamPoolBuffer(b[i].handle);
                if (sameSite && a[i].data == b[i].data) {
                    a[i].replay = false;
                    b[i].replay = false;
                }
            }
        }
        if (std::getenv("BP_EXEC_STATS")) {
            size_t captured = a.size() + b.size();
            size_t replayed = 0;
            for (const auto& w : a) replayed += w.replay;
            for (const auto& w : b) replayed += w.replay;
            fprintf(stderr,
                "  [fast decode writes] %zu captured, %zu replayed, %zu immutable\n",
                captured, replayed, captured - replayed);
        }
    }

    // Shape-specialized prefill capture used by reused-generator benchmarks.
    // The reset snapshots retain the stable zero-state input buffers while
    // the ordinary state maps advance to the captured outputs.
    bool prefillCaptureReady = false;
    uint32_t prefillCaptureTokens = 0;
    std::unordered_map<std::string, GpuTensor> resetConvState;
    std::unordered_map<std::string, GpuTensor> resetRecurrentState;
    std::unordered_map<std::string, GpuTensor> resetKvState;
    bool stateBuffersAllocated = false;
    std::unordered_map<std::string, GpuTensor> capturedPrefillConvState;
    std::unordered_map<std::string, GpuTensor> capturedPrefillRecurrentState;

    void SwapCaptureVariant(int index) {
        auto& variant = qwenCaptureVariants[index];
        execCtx.capturedFlushes_.swap(variant.flushes);
        execCtx.capturedTemporaryBuffers_.swap(variant.temporaryBuffers);
        execCtx.capturedWrites_.swap(variant.writes);
        execCtx.replayParamUpdates_.swap(variant.params);
        execCtx.replayScalarUpdates_.swap(variant.scalars);
        execCtx.capturedTokenIdBufs_.swap(variant.tokenIds);
        execCtx.replaySkipBuffers_.swap(variant.skipBuffers);
    }

    void StoreCurrentQwenCapture(int index) {
        SwapCaptureVariant(index);
        qwenActiveVariant = -1;
    }

    void ActivateQwenCapture(int index) {
        if (qwenActiveVariant == index) return;
        if (qwenActiveVariant >= 0) SwapCaptureVariant(qwenActiveVariant);
        SwapCaptureVariant(index);
        qwenActiveVariant = index;
        logitsBuf = qwenCaptureVariants[index].logits;
    }

    bool Load(GPUContext& gpuCtx, const std::string& onnxPath, int64_t maxSeqOverride) {
        gpu = &gpuCtx;
        execCtx.gpu = &gpuCtx;
        modelPath = onnxPath;
        modelDir = fs::path(onnxPath).parent_path().string();
        if (maxSeqOverride > 0) maxSeqLen = maxSeqOverride;

        if (!executor.Load(gpuCtx, onnxPath)) return false;
        // Current exports shorten recurrent cache names, while older exports
        // use past_key_values.N.conv_state/recurrent_state. Keep internal
        // state ownership unchanged and bind the graph's actual I/O names.
        cacheAliases.clear();
        auto discoverAliases = [&](const auto& entries, bool input) {
            const std::string prefix = input ? "past." : "present.";
            for (const auto& entry : entries) {
                if (entry.name.rfind(prefix, 0) != 0) continue;
                auto dot = entry.name.find('.', prefix.size());
                if (dot == std::string::npos) continue;
                auto layer = entry.name.substr(prefix.size(), dot - prefix.size());
                auto kind = entry.name.substr(dot + 1);
                if (kind == "conv") kind = "conv_state";
                else if (kind == "recurrent") kind = "recurrent_state";
                else if (kind != "key" && kind != "value") continue;
                const std::string legacy = (input ? "past_key_values." : "present.") + layer + "." + kind;
                if (legacy != entry.name) cacheAliases[legacy] = entry.name;
            }
        };
        discoverAliases(executor.GetGraph().inputs, true);
        discoverAliases(executor.GetGraph().outputs, false);
        for(const auto& input:executor.GetGraph().inputs)
            if(input.name=="position_ids" && input.shape.size()==3 && input.shape[0]==3) positionAxes=3;
        if (!tokenizer.load(modelDir)) return false;

        std::string cfgPath = (fs::path(modelDir) / "config.json").string();
        if (!fs::exists(cfgPath)) {
            fprintf(stderr, "OnnxLlm: missing config.json\n");
            return false;
        }
        std::ifstream cfgFile(cfgPath);
        std::string cfgStr{std::istreambuf_iterator<char>(cfgFile),
                           std::istreambuf_iterator<char>()};
        cfgFile.close();
        auto cfgRoot = json_parse(cfgStr);
        const JsonValue& cfg = cfgRoot.has("text_config")
            ? cfgRoot["text_config"] : cfgRoot;

        hiddenSize = cfg.has("hidden_size") ? cfg["hidden_size"].as_int() : 2048;
        numLayers = cfg.has("num_hidden_layers") ? cfg["num_hidden_layers"].as_int() : 24;
        vocabSize = cfg.has("vocab_size") ? cfg["vocab_size"].as_int() : 65536;
        numKvHeads = cfg.has("num_key_value_heads") ? cfg["num_key_value_heads"].as_int() : 8;
        numHeads = cfg.has("num_attention_heads") ? cfg["num_attention_heads"].as_int() : 32;
        headDim = cfg.has("head_dim") ? cfg["head_dim"].as_int() : hiddenSize / numHeads;
        convLCache = cfg.has("conv_L_cache") ? cfg["conv_L_cache"].as_int() : 3;
        intermediateSize = cfg.has("intermediate_size") ? cfg["intermediate_size"].as_int() : 7168;
        recurrentHeads = cfg.has("linear_num_value_heads")
            ? cfg["linear_num_value_heads"].as_int() : 0;
        recurrentKeyDim = cfg.has("linear_key_head_dim")
            ? cfg["linear_key_head_dim"].as_int() : 0;
        recurrentValueDim = cfg.has("linear_value_head_dim")
            ? cfg["linear_value_head_dim"].as_int() : 0;
        if (cfg.has("linear_conv_kernel_dim")) {
            const int64_t keyHeads = cfg.has("linear_num_key_heads")
                ? cfg["linear_num_key_heads"].as_int() : recurrentHeads;
            // The convolution packs Q, K, and V. Q/K use key-head geometry,
            // while V uses value-head geometry; this is not generally 3H.
            convChannels = 2 * keyHeads * recurrentKeyDim +
                           recurrentHeads * recurrentValueDim;
        } else {
            convChannels = hiddenSize;
        }
        if (cfg.has("linear_conv_kernel_dim"))
            convLCache = cfg["linear_conv_kernel_dim"].as_int() - 1;
        moeIntermediateSize = cfg.has("moe_intermediate_size") ? cfg["moe_intermediate_size"].as_int() : 1792;
        numExperts = cfg.has("num_experts") ? cfg["num_experts"].as_int() : 32;
        numExpertsPerTok = cfg.has("num_experts_per_tok") ? cfg["num_experts_per_tok"].as_int() : 4;
        if (cfg.has("norm_eps")) normEps = (float)cfg["norm_eps"].as_number();
        else if (cfg.has("rms_norm_eps")) normEps = (float)cfg["rms_norm_eps"].as_number();
        if (cfg.has("rope_parameters")) {
            auto& rp = cfg["rope_parameters"];
            if (rp.has("rope_theta")) ropeTheta = (float)rp["rope_theta"].as_number();
        }
        arch = cfg.has("model_type") ? cfg["model_type"].as_string() : "onnx";
        LoadCpuEmbedding();

        int64_t modelMaxSeq = cfg.has("max_position_embeddings") ? cfg["max_position_embeddings"].as_int() : 4096;

        if (cfg.has("layer_types") && cfg["layer_types"].is_array()) {
            for (int64_t i = 0; i < cfg["layer_types"].size(); i++) {
                std::string lt = cfg["layer_types"][i].as_string();
                layerTypes.push_back(lt);
                if (lt == "conv" || lt == "linear_attention")
                    convLayerIndices.push_back((int)i);
                else if (lt == "full_attention" || lt == "sliding_attention")
                    attnLayerIndices.push_back((int)i);
            }
        }
        if (!attnLayerIndices.empty())
            fprintf(stderr, "  Attention layers: %zu (of %zu total)\n",
                    attnLayerIndices.size(), layerTypes.size());

        if (maxSeqLen <= 0) {
            int64_t nAttn = std::max((int64_t)1, (int64_t)attnLayerIndices.size());
            uint64_t maxBuf = gpuCtx.adapterLimits.maxBufferSize;
            int64_t perBufLimit = (int64_t)(maxBuf / (numKvHeads * headDim * 4));
            int64_t budgetBytes = (int64_t)(maxBuf / 4);
            int64_t perBufFromBudget = budgetBytes / (nAttn * 2 * numKvHeads * headDim * 4);
            int64_t computed = std::min(perBufLimit, perBufFromBudget);
            int64_t rounded = 1;
            while (rounded * 2 <= computed) rounded *= 2;
            maxSeqLen = std::min(modelMaxSeq, std::max(rounded, (int64_t)4096));
        } else {
            maxSeqLen = std::min(modelMaxSeq, maxSeqLen);
        }

        ResetCaches();
        return true;
    }

    void RestoreInitialState() {
        convState = resetConvState;
        recurrentState = resetRecurrentState;
        kvState = resetKvState;
        for (auto* states : {&convState, &recurrentState}) {
            for (auto& [name, t] : *states) {
                std::vector<uint8_t> zeros(t.ByteSize(), 0);
                gpu->writeBuffer(t.buffer, zeros.data(), zeros.size());
            }
        }
        for (auto& [name, t] : kvState)
            t.shape = {1, numKvHeads, 0, headDim};
    }

    void ResetCaches(bool preserveWarmGraph = false) {
        pos = 0;
        prefillDone = false;
        decodePlanInitialized = false;
        decodeWarmupRemaining = 0;
        nlkWritten = false;
        if (preserveWarmGraph && stateBuffersAllocated) {
            RestoreInitialState();
            return;
        }
        convState.clear();
        recurrentState.clear();
        kvState.clear();
        if (qwenActiveVariant >= 0) {
            SwapCaptureVariant(qwenActiveVariant);
            qwenActiveVariant = -1;
        }
        for (int i = 0; i < qwenCapturedVariants; i++) {
            SwapCaptureVariant(i);
            execCtx.ReleaseCaptured();
        }
        qwenCapturedVariants = 0;
        qwenNextReplayVariant = 0;
        execCtx.ReleaseCaptured();
        fastDecodeCaptured = false;
        prefillCaptureReady = false;
        prefillCaptureTokens = 0;
        capturedPrefillConvState.clear();
        capturedPrefillRecurrentState.clear();
        // Captured NVIDIA Qwen variants reserve disjoint parameter-pool
        // ranges. They are reusable only after every captured bind group has
        // been released above.
        execCtx.ResetParamPoolCursors();
        execCtx.InvalidateWarmCaches();

        // These input/output allocations belong to the session, not to the
        // executor's shape-specific tensor plan. A reset restores the initial
        // state and reuses both output parities instead of losing their handles.
        // Qwen3.8-27B previously leaked 463 MiB on every conversation reset.
        if (stateBuffersAllocated) {
            RestoreInitialState();
            return;
        }

        for (size_t ci = 0; ci < convLayerIndices.size(); ci++) {
            int idx = convLayerIndices[ci];
            std::string name = "past_key_values." + std::to_string(idx) + ".conv_state";
            GpuTensor t;
            t.shape = {1, convChannels, convLCache};
            t.dtype = TensorDtype::Float32;
            const size_t elements = (size_t)(convChannels * convLCache);
            size_t bytes = elements * 4;
            if (fastDecodeEnabled && ci < convCastF16Bufs.size()) {
                // Replay feeds the previous f32 convolution output through
                // cache_cast_f16 into this stable captured input buffer.
                // Initialize it in that same dtype and byte width; writing
                // f32 zeros here overran the f16 allocation for Qwen 3.5.
                t.buffer = convCastF16Bufs[ci];
                if (arch == "qwen3_5_text") {
                    t.dtype = TensorDtype::Float32;
                    std::vector<float> zeros(elements, 0.0f);
                    gpu->writeBuffer(t.buffer, zeros.data(), elements * 4);
                } else {
                    t.dtype = TensorDtype::Float16;
                    std::vector<uint16_t> zeros(elements, 0);
                    gpu->writeBuffer(t.buffer, zeros.data(), elements * 2);
                }
            } else {
                t.buffer = gpu->createBuffer(name, bytes);
                std::vector<float> zeros(elements, 0.0f);
                gpu->writeBuffer(t.buffer, zeros.data(), bytes);
            }
            convState[name] = t;

            if (recurrentHeads > 0 && recurrentKeyDim > 0 && recurrentValueDim > 0) {
                std::string recurrentName = "past_key_values." + std::to_string(idx) +
                                            ".recurrent_state";
                GpuTensor recurrent;
                recurrent.shape = {1, recurrentHeads, recurrentKeyDim, recurrentValueDim};
                recurrent.dtype = TensorDtype::Float32;
                size_t recurrentBytes = (size_t)recurrentHeads * recurrentKeyDim *
                                        recurrentValueDim * 4;
                recurrent.buffer = gpu->createBuffer(recurrentName, recurrentBytes);
                std::vector<float> recurrentZeros(recurrentBytes / 4, 0.0f);
                gpu->writeBuffer(recurrent.buffer, recurrentZeros.data(), recurrentBytes);
                recurrentState[recurrentName] = recurrent;
            }
        }

        size_t kvBytes = (size_t)(numKvHeads * maxSeqLen * headDim * 4);
        for (int idx : attnLayerIndices) {
            for (const char* suffix : {".key", ".value"}) {
                std::string name = "past_key_values." + std::to_string(idx) + suffix;
                GpuTensor t;
                t.shape = {1, numKvHeads, 0, headDim};
                t.dtype = TensorDtype::Float32;
                t.kvCacheCapacity = maxSeqLen;
                t.buffer = gpu->createBuffer(name, kvBytes);
                kvState[name] = t;
            }
        }

        if (cpuEmbedding) embeddingsBuf = gpu->createBuffer("text_embeddings",
            uint64_t(maxSeqLen) * hiddenSize * GpuTensor::DtypeSizeOf(cpuEmbedding->dtype));
        idsBuf = gpu->createBuffer("ids", 8);
        prefillIdsBuf = gpu->createBuffer("prefill_ids", maxSeqLen * 8);
        positionBuf = gpu->createBuffer("position_ids", maxSeqLen * positionAxes * 8);
        nlkBuf = gpu->createBuffer("nlk", 8);
        logitsBuf = gpu->createBuffer("logits_out", vocabSize * 4);
        maskBuf = gpu->createBuffer("mask", maxSeqLen * 8);
        maskBufCapacity = (uint32_t)maxSeqLen;
        convOutBufs.resize(convLayerIndices.size());
        convOutAltBufs.resize(convLayerIndices.size());
        recurrentOutBufs.resize(convLayerIndices.size());
        recurrentOutAltBufs.resize(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "conv_out_" + std::to_string(convLayerIndices[i]);
            convOutBufs[i] = gpu->createBuffer(name, convChannels * convLCache * 4);
            convOutAltBufs[i] = gpu->createBuffer(name + "_alt", convChannels * convLCache * 4);
            if (recurrentHeads > 0) {
                const size_t bytes = static_cast<size_t>(recurrentHeads) *
                    recurrentKeyDim * recurrentValueDim * 4;
                recurrentOutBufs[i] = gpu->createBuffer(
                    "recurrent_out_" + std::to_string(convLayerIndices[i]), bytes);
                recurrentOutAltBufs[i] = gpu->createBuffer(
                    "recurrent_out_alt_" + std::to_string(convLayerIndices[i]), bytes);
            }
        }
        resetConvState = convState;
        resetRecurrentState = recurrentState;
        resetKvState = kvState;
        stateBuffersAllocated = true;
    }

    std::vector<float> RunPrefillStep(int64_t tokenId) {
        pos++;
        return runExecutePath(tokenId);
    }

    int32_t RunPrefillBatch(const int32_t* tokenIds, uint32_t T) {
        if (T == 0) return -1;
        if (uint64_t(pos) + T > uint64_t(maxSeqLen))
            throw std::runtime_error("Prompt exceeds the configured context capacity");
        if (prefillChunkSize && T > prefillChunkSize) {
            int32_t next = -1;
            for (uint32_t offset = 0; offset < T; offset += prefillChunkSize)
                next = RunPrefillBatch(tokenIds + offset, std::min(prefillChunkSize, T - offset));
            return next;
        }
        // Intel's Qwen 4B graph has one known bad dynamic shape at M=4, but
        // the normal 64-token bounded path is conformant. Keep an override for
        // bounded kernel-isolation experiments without penalizing production.
        const bool intelQwen = arch == "qwen3_5_text" &&
            gpu->adapterName.find("Intel") != std::string::npos;
        uint32_t intelQwenChunk = 64;
        if (const char* value = std::getenv("BP_INTEL_QWEN_PREFILL_CHUNK"))
            intelQwenChunk = std::max(1, atoi(value));
        if (intelQwen && numLayers != 24 && T > intelQwenChunk) {
            int32_t result = -1;
            for (uint32_t offset = 0; offset < T; offset += intelQwenChunk)
                result = RunPrefillBatch(tokenIds + offset,
                    std::min(intelQwenChunk, T - offset));
            return result;
        }
        if (std::getenv("BP_GENERIC_SERIAL_PREFILL")) {
            std::vector<float> logits;
            for (uint32_t i = 0; i < T; ++i)
                logits = RunPrefillStep(tokenIds[i]);
            return argmax(logits.data(), (int64_t)logits.size());
        }
        if (T == 1) {
            auto logits = RunPrefillStep(tokenIds[0]);
            return argmax(logits.data(), (int64_t)logits.size());
        }

        // All cared Windows adapters use the sequence-causal
        // Conv/LinearAttention graph. GQA selects the vendor-validated
        // reduction in attention.cpp; none of these paths requires subgroup
        // matrices, which are unavailable through Windows WebGPU.
        const bool deviceQwenBatch = arch == "qwen3_5_text" &&
            (gpu->adapterName.find("NVIDIA") != std::string::npos ||
             gpu->adapterName.find("AMD") != std::string::npos ||
             gpu->adapterName.find("Intel") != std::string::npos) &&
            !std::getenv("BP_QWEN_SERIAL_PREFILL");
        if (arch == "qwen3_5_text" && !deviceQwenBatch) {
            prefillDone = true;
            std::vector<float> logits;
            for (uint32_t i = 0; i < T; ++i)
                logits = RunStep(tokenIds[i]);
            return argmax(logits.data(), (int64_t)logits.size());
        }

        pos += T;
        std::unordered_map<std::string, GpuTensor*> inputs;

        std::vector<int64_t> ids64(T);
        for (uint32_t i = 0; i < T; i++) ids64[i] = tokenIds[i];
        GpuTensor idT;
        idT.shape = {1, (int64_t)T};
        idT.dtype = TensorDtype::Int64;
        idT.buffer = prefillIdsBuf;
        gpu->writeBuffer(idT.buffer, ids64.data(), T * 8);
        idT.cpuData.resize(T * 8);
        memcpy(idT.cpuData.data(), ids64.data(), T * 8);
        inputs["input_ids"] = &idT;
        GpuTensor embeddings;
        if (cpuEmbedding) {
            PrepareEmbeddings(tokenIds, T, embeddings);
            inputs["inputs_embeds"] = &embeddings;
        }

        const bool intelQwen2 = arch == "qwen3_5_text" && numLayers == 24 &&
            gpu->adapterName.find("Intel") != std::string::npos;
        GpuTensor positionT;
        std::vector<int64_t> positions;
        if (!intelQwen2) {
            positions.resize(T*positionAxes);
            for (uint32_t i = 0; i < T*positionAxes; i++)
                positions[i] = (int64_t)(pos - T + i%T);
            positionT.shape = positionAxes==1 ? std::vector<int64_t>{1,(int64_t)T} : std::vector<int64_t>{positionAxes,1,(int64_t)T};
            positionT.dtype = TensorDtype::Int64;
            positionT.buffer = positionBuf;
            gpu->writeBuffer(positionBuf, positions.data(), positions.size() * 8);
            positionT.cpuData.resize(positions.size() * 8);
            memcpy(positionT.cpuData.data(), positions.data(), positions.size() * 8);
            inputs["position_ids"] = &positionT;
        }

        std::vector<int64_t> mask(pos, 1);
        GpuTensor maskT;
        maskT.shape = {1, (int64_t)pos};
        maskT.dtype = TensorDtype::Int64;
        maskT.buffer = maskBuf;
        gpu->writeBuffer(maskBuf, mask.data(), pos * 8);
        maskT.cpuData.resize(pos * 8);
        memcpy(maskT.cpuData.data(), mask.data(), pos * 8);
        inputs["attention_mask"] = &maskT;

        int64_t nlk = 1;
        GpuTensor nlkT;
        nlkT.shape = {};
        nlkT.dtype = TensorDtype::Int64;
        nlkT.buffer = nlkBuf;
        if (!nlkWritten) {
            gpu->writeBuffer(nlkBuf, &nlk, 8);
            nlkWritten = true;
        }
        nlkT.cpuData.resize(8);
        memcpy(nlkT.cpuData.data(), &nlk, 8);
        inputs["num_logits_to_keep"] = &nlkT;

        for (auto& [name, t] : convState) inputs[name] = &t;
        for (auto& [name, t] : recurrentState) inputs[name] = &t;
        for (auto& [name, t] : kvState) {
            t.shape = {1, numKvHeads, (int64_t)(pos - T), headDim};
            inputs[name] = &t;
        }

        std::unordered_map<std::string, GpuTensor*> outputs;
        GpuTensor logitsOut;
        logitsOut.shape = {1, 1, vocabSize};
        logitsOut.dtype = TensorDtype::Float32;
        logitsOut.buffer = logitsBuf;
        outputs["logits"] = &logitsOut;

        std::vector<GpuTensor> convOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "present." + std::to_string(convLayerIndices[i]) + ".conv_state";
            convOuts[i].shape = {1, convChannels, convLCache};
            convOuts[i].dtype = TensorDtype::Float32;
            const std::string inputName = "past_key_values." +
                std::to_string(convLayerIndices[i]) + ".conv_state";
            convOuts[i].buffer = fastDecodeEnabled && T == 1
                ? convState[inputName].buffer
                : (convState[inputName].buffer.handle == convOutBufs[i].handle
                    ? convOutAltBufs[i] : convOutBufs[i]);
            outputs[name] = &convOuts[i];
        }

        std::vector<GpuTensor> recurrentOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size() && recurrentHeads > 0; i++) {
            std::string name = "present." + std::to_string(convLayerIndices[i]) +
                               ".recurrent_state";
            recurrentOuts[i].shape = {1, recurrentHeads, recurrentKeyDim, recurrentValueDim};
            recurrentOuts[i].dtype = TensorDtype::Float32;
            const std::string inputName = "past_key_values." +
                std::to_string(convLayerIndices[i]) + ".recurrent_state";
            recurrentOuts[i].buffer = fastDecodeEnabled && T == 1
                ? recurrentState[inputName].buffer
                : (recurrentState[inputName].buffer.handle == recurrentOutBufs[i].handle
                    ? recurrentOutAltBufs[i] : recurrentOutBufs[i]);
            outputs[name] = &recurrentOuts[i];
        }

        std::vector<GpuTensor> kvOuts(attnLayerIndices.size() * 2);
        for (size_t i = 0; i < attnLayerIndices.size(); i++) {
            std::string kName = "present." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vName = "present." + std::to_string(attnLayerIndices[i]) + ".value";
            std::string kIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".value";
            kvOuts[i*2].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2].dtype = TensorDtype::Float32;
            kvOuts[i*2].buffer = kvState[kIn].buffer;
            kvOuts[i*2+1].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2+1].dtype = TensorDtype::Float32;
            kvOuts[i*2+1].buffer = kvState[vIn].buffer;
            outputs[kName] = &kvOuts[i*2];
            outputs[vName] = &kvOuts[i*2+1];
        }

        const uint64_t logitBytes = (uint64_t)vocabSize * sizeof(float);
        auto readbackHandle = gpu->getOrCreateReadbackBuf(logitBytes);
        execCtx.RequestReadback(logitsBuf, {readbackHandle, logitBytes}, logitBytes);
        ExecuteGraph(inputs, outputs);

        int64_t logitNel = logitsOut.ElementCount();
        auto logits = ReadLogits(logitNel);
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string inName = "past_key_values." + std::to_string(convLayerIndices[i]) +
                                 ".conv_state";
            if (!convOuts[i].IsValid()) continue;
            convState[inName] = convOuts[i];
            if (i < recurrentOuts.size() && recurrentOuts[i].IsValid()) {
                std::string recurrentName = "past_key_values." +
                    std::to_string(convLayerIndices[i]) + ".recurrent_state";
                recurrentState[recurrentName] = recurrentOuts[i];
            }
        }

        for (int idx : attnLayerIndices) {
            std::string kIn = "past_key_values." + std::to_string(idx) + ".key";
            std::string vIn = "past_key_values." + std::to_string(idx) + ".value";
            kvState[kIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvState[vIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
        }

        execCtx.SubmitPending();
        return argmax(logits.data(), (int64_t)logits.size());
    }

    int32_t CapturePrefillBatch(const int32_t* tokenIds, uint32_t T) {
        // Flat capture restores CPU writes before commands; it cannot preserve
        // the write/execute interleaving between multiple prompt chunks.
        if (prefillChunkSize && T > prefillChunkSize) return -1;
        ResetCaches(true);
        execCtx.CaptureBegin();
        int32_t result = RunPrefillBatch(tokenIds, T);
        execCtx.CaptureEnd();
        capturedPrefillConvState = convState;
        capturedPrefillRecurrentState = recurrentState;
        prefillCaptureTokens = T;
        prefillCaptureReady = !execCtx.capturedFlushes_.empty();
        return result;
    }

    int32_t ReplayCapturedPrefillBatch(const int32_t* tokenIds, uint32_t T) {
        if (!prefillCaptureReady || T != prefillCaptureTokens) return -1;
        ResetCaches(true);
        pos = T;

        std::vector<int64_t> ids(T), positions(T*positionAxes), mask(T, 1);
        for (uint32_t i = 0; i < T; ++i) {
            ids[i] = tokenIds[i];
            for(uint32_t axis=0;axis<positionAxes;++axis) positions[axis*T+i] = i;
        }
        gpu->writeBuffer(prefillIdsBuf, ids.data(), T * sizeof(int64_t));
        const bool intelQwen2 = arch == "qwen3_5_text" && numLayers == 24 &&
            gpu->adapterName.find("Intel") != std::string::npos;
        if (!intelQwen2)
            gpu->writeBuffer(positionBuf, positions.data(), positions.size() * sizeof(int64_t));
        gpu->writeBuffer(maskBuf, mask.data(), T * sizeof(int64_t));
        const int64_t nlk = 1;
        gpu->writeBuffer(nlkBuf, &nlk, sizeof(nlk));

        // Restore CPU-produced constants frozen by capture. External prompt
        // inputs above deliberately override their corresponding buffers.
        for (const auto& write : execCtx.capturedWrites_) {
            if (!write.handle || write.data.empty()) continue;
            gpu->writeBufferRaw(write.handle, write.offset,
                                write.data.data(), write.data.size());
        }
        gpu->writeBuffer(prefillIdsBuf, ids.data(), T * sizeof(int64_t));
        if (!intelQwen2)
            gpu->writeBuffer(positionBuf, positions.data(), positions.size() * sizeof(int64_t));
        gpu->writeBuffer(maskBuf, mask.data(), T * sizeof(int64_t));
        gpu->writeBuffer(nlkBuf, &nlk, sizeof(nlk));

        if (cpuEmbedding) {
            GpuTensor embeddings;
            PrepareEmbeddings(tokenIds, T, embeddings);
        }
        execCtx.ReplayDispatches();
        auto logits = ReadLogits(vocabSize);

        convState = capturedPrefillConvState;
        recurrentState = capturedPrefillRecurrentState;
        for (int idx : attnLayerIndices) {
            kvState["past_key_values." + std::to_string(idx) + ".key"].shape =
                {1, numKvHeads, (int64_t)T, headDim};
            kvState["past_key_values." + std::to_string(idx) + ".value"].shape =
                {1, numKvHeads, (int64_t)T, headDim};
        }
        execCtx.ReleaseCaptured();
        prefillCaptureReady = false;
        prefillDone = true;
        return argmax(logits.data(), (int64_t)logits.size());
    }

    std::vector<float> RunStep(int64_t tokenId) {
        pos++;

        // Prefill and decode have fundamentally different tensor shapes. A
        // plan learned from the prompt cannot be reused for single-token
        // decode; rebuild it once, then retain the stable decode allocations.
        if (!decodePlanInitialized) {
            execCtx.InvalidateWarmCaches();
            decodePlanInitialized = true;
            // Capture must bind the stable single-token tensor plan. Capturing
            // this first decode would keep one-off allocations alive but also
            // suppress plan finalization, freezing transient bindings.
            if (fastDecodeEnabled) {
                decodeWarmupRemaining = 2;
                return runExecutePath(tokenId);
            }
        }

        if (fastDecodeEnabled && !fastDecodeCaptured &&
            decodeWarmupRemaining > 0) {
            decodeWarmupRemaining--;
            return runExecutePath(tokenId);
        }

        // Fast Decode Replay Path
        if (fastDecodeEnabled && fastDecodeCaptured) {
            if (arch == "qwen3_5_text") {
                ActivateQwenCapture(qwenNextReplayVariant);
            }
            execCtx.replayPosition_ = (uint32_t)(pos - 1);
            execCtx.replayTokenId_ = tokenId;
            // CPU-produced constants and shape results are part of each ONNX
            // execution. Restore the capture variant's writes before applying
            // the token/position-specific overrides below.
            for (const auto& write : execCtx.capturedWrites_) {
                if (!write.replay || !write.handle || write.data.empty()) continue;
                gpu->writeBufferRaw(write.handle, write.offset,
                                    write.data.data(), write.data.size());
            }
            const int64_t replayToken = tokenId;
            const int64_t replayPosition = static_cast<int64_t>(execCtx.replayPosition_);
            const std::vector<int64_t> replayPositions(positionAxes,replayPosition);
            gpu->writeBufferRaw(idsBuf.handle, idsBuf.offset,
                                &replayToken, sizeof(replayToken));
            gpu->writeBufferRaw(positionBuf.handle, positionBuf.offset,
                                replayPositions.data(), replayPositions.size()*sizeof(int64_t));
            const int64_t maskOne = 1;
            gpu->writeBufferRaw(maskBuf.handle,
                maskBuf.offset + static_cast<uint64_t>(execCtx.replayPosition_) * 8,
                &maskOne, sizeof(maskOne));
            if (cpuEmbedding) {
                GpuTensor embeddings; const int32_t token = static_cast<int32_t>(tokenId);
                PrepareEmbeddings(&token, 1, embeddings);
            }
            execCtx.ReplayWrites();

            // The exported Qwen graph selects one 32-float RoPE cache row on
            // the CPU through Where nodes. Capture freezes those six writes;
            // refresh them from the immutable ONNX cache at replay position.
            const auto* sinCache = executor.GetInitData("model.rotary_emb.sin_cache");
            const auto* cosCache = executor.GetInitData("model.rotary_emb.cos_cache");
            for (const auto& write : execCtx.capturedWrites_) {
                if (write.opName.find("Range:/model/attn/synthetic_pos_ids/range/Range") !=
                        std::string::npos && write.data.size() == 4) {
                    const int32_t position = static_cast<int32_t>(execCtx.replayPosition_);
                    gpu->writeBufferRaw(write.handle, write.offset, &position, sizeof(position));
                    continue;
                }
                const OnnxInitData* cache = nullptr;
                if (write.opName.find("/rotary_emb/sin/") != std::string::npos)
                    cache = sinCache;
                else if (write.opName.find("/rotary_emb/cos/") != std::string::npos)
                    cache = cosCache;
                if (!cache || !cache->data || cache->shape.size() != 2 ||
                    cache->shape[0] <= 0 || cache->shape[1] <= 0) continue;
                const uint64_t rowBytes = static_cast<uint64_t>(cache->shape[1]) * 4;
                const uint64_t row = std::min<uint64_t>(execCtx.replayPosition_,
                    static_cast<uint64_t>(cache->shape[0] - 1));
                if (write.data.size() != rowBytes) continue;
                gpu->writeBufferRaw(write.handle, write.offset,
                    cache->data + row * rowBytes, rowBytes);
            }

            const bool gpuGreedy = requestGpuGreedyToken &&
                execCtx.fusedLmHeadArgmaxAvailable_ &&
                execCtx.fusedLmHeadArgmaxResult_.handle;
            uint64_t readbackBytes = gpuGreedy ? 4u : (uint64_t)vocabSize * GpuTensor::DtypeSizeOf(logitsDtype);
            GPUBuffer readbackSrc = gpuGreedy
                ? execCtx.fusedLmHeadArgmaxResult_ : logitsBuf;
            auto rbHandle = gpu->getOrCreateReadbackBuf(readbackBytes);
            execCtx.RequestReadback(readbackSrc,
                {rbHandle, readbackBytes}, readbackBytes);
            execCtx.ReplayDispatches();

            std::vector<float> logits;
            if (gpuGreedy) {
                auto rb = gpu->mapReadbackBuffer(readbackBytes);
                memcpy(&lastGpuGreedyToken, rb.data(), 4);
            } else {
                logits = ReadLogits(vocabSize);
            }

            for (size_t i = 0; i < convLayerIndices.size(); i++) {
                std::string inName = "past_key_values." + std::to_string(convLayerIndices[i]) +
                                     ".conv_state";
                if (arch != "qwen3_5_text") {
                    wgpuBindGroupAddRef(convCastBindGroups[i]);
                    execCtx.QueueDispatch(convCastPipeline->pipeline, convCastBindGroups[i],
                        convCastWorkgroups, 1, 1, "cache_cast_f16");
                    convState[inName].buffer = convCastF16Bufs[i];
                    convState[inName].dtype = TensorDtype::Float16;
                }
                if (arch != "qwen3_5_text" &&
                    i < capturedRecurrentInputBufs.size() &&
                    i < capturedRecurrentOutputBufs.size()) {
                    const uint64_t bytes = static_cast<uint64_t>(recurrentHeads) *
                        recurrentKeyDim * recurrentValueDim * 4;
                    execCtx.QueueCopy(capturedRecurrentOutputBufs[i], 0,
                                      capturedRecurrentInputBufs[i], 0, bytes);
                    std::string recurrentName = "past_key_values." +
                        std::to_string(convLayerIndices[i]) + ".recurrent_state";
                    recurrentState[recurrentName].buffer = capturedRecurrentInputBufs[i];
                }
            }
            execCtx.SubmitPending();
            for (int idx : attnLayerIndices) {
                kvState["past_key_values." + std::to_string(idx) + ".key"].shape
                    = {1, numKvHeads, (int64_t)pos, headDim};
                kvState["past_key_values." + std::to_string(idx) + ".value"].shape
                    = {1, numKvHeads, (int64_t)pos, headDim};
            }
            if (arch == "qwen3_5_text")
                qwenNextReplayVariant =
                    (qwenNextReplayVariant + 1) % qwenCapturedVariants;
            return logits;
        }

        // Capture Stage
        if (fastDecodeEnabled && !fastDecodeCaptured && prefillDone) {
            capturedRecurrentInputBufs.clear();
            capturedRecurrentInputBufs.reserve(convLayerIndices.size());
            for (int idx : convLayerIndices) {
                std::string name = "past_key_values." + std::to_string(idx) +
                                   ".recurrent_state";
                capturedRecurrentInputBufs.push_back(recurrentState[name].buffer);
            }
            execCtx.capturePosition_ = (uint32_t)(pos - 1);
            execCtx.CaptureBegin();
            auto logits = runExecutePath(tokenId);
            execCtx.CaptureEnd();

            if (arch == "qwen3_5_text") {
                const int variant = qwenCapturedVariants;
                qwenCaptureVariants[variant].logits = logitsBuf;
                if (const char* folder = std::getenv("BP_DUMP_ONNX_STATE_DIR")) {
                    qwenCaptureVariants[variant].diagnosticConv = convState["past_key_values.0.conv_state"];
                    qwenCaptureVariants[variant].diagnosticRecurrent = recurrentState["past_key_values.0.recurrent_state"];
                    std::ofstream writes(fs::path(folder) / ("capture-writes-" + std::to_string(variant) + ".txt"));
                    for (const auto& write : execCtx.capturedWrites_) {
                        writes << write.opName << "\t" << write.handle << "\t" << write.offset << "\t" << write.data.size();
                        for (size_t i = 0; i + 4 <= write.data.size() && i < 48; i += 4) {
                            uint32_t value; memcpy(&value, write.data.data() + i, 4);
                            writes << "\t" << value;
                        }
                        writes << "\n";
                    }
                }
                if (std::getenv("BP_EXEC_STATS")) {
                    int nDisp = 0;
                    for (auto& f : execCtx.capturedFlushes_)
                        nDisp += static_cast<int>(f.dispatches.size());
                    fprintf(stderr,
                        "  [fast decode capture %d/2] %zu flushes, %d dispatches, %zu param updates, Q4 quantize=%u reuse=%u\n",
                        variant + 1,
                        execCtx.capturedFlushes_.size(), nDisp,
                        execCtx.replayParamUpdates_.size(),
                        execCtx.q4DecodeQuantizeDispatches_,
                        execCtx.q4DecodeReuseHits_);
                }
                StoreCurrentQwenCapture(variant);
                qwenCapturedVariants++;
                // NVIDIA fast decode freezes invariant CPU-produced constants.
                // Keep capture 1 and capture 2 in disjoint parameter-pool
                // ranges so neither parity overwrites the other's constants.
                if (!IsNvidiaQwenCapture())
                    execCtx.ResetParamPoolCursors();
                // Qwen's recurrent and convolution state buffers ping-pong
                // every token. Preserve both binding parities and alternate
                // them during replay, just as the ordinary graph does.
                const int requiredVariants = 2;
                if (qwenCapturedVariants == requiredVariants) {
                    ClassifyNvidiaQwenCaptureWrites();
                    fastDecodeCaptured = true;
                    // Three ordinary decode steps settle the tensor plan
                    // before capture. Recurrent caches are safely in-place,
                    // leaving one stable command stream to replay.
                    qwenNextReplayVariant = 0;
                }
                return logits;
            }

            fastDecodeCaptured = true;

            capturedConvOutputBufs.resize(convLayerIndices.size());
            capturedRecurrentOutputBufs.resize(convLayerIndices.size());
            for (size_t i = 0; i < convLayerIndices.size(); i++) {
                const std::string prefix = "past_key_values." +
                    std::to_string(convLayerIndices[i]);
                capturedConvOutputBufs[i] = convState[prefix + ".conv_state"].buffer;
                capturedRecurrentOutputBufs[i] =
                    recurrentState[prefix + ".recurrent_state"].buffer;
            }

            if (!execCtx.capturedFlushes_.empty()) {
                auto& lastF = execCtx.capturedFlushes_.back();
                if (!lastF.dispatches.empty() &&
                    lastF.dispatches[0].name.find("cache_cast") != std::string::npos) {
                    execCtx.capturedFlushes_.pop_back();
                }
            }

            if (std::getenv("BP_EXEC_STATS")) {
                int nDisp = 0;
                for (auto& f : execCtx.capturedFlushes_) nDisp += (int)f.dispatches.size();
                fprintf(stderr, "  [fast decode capture] %zu flushes, %d dispatches, %zu param updates, %zu token inputs\n",
                        execCtx.capturedFlushes_.size(), nDisp, execCtx.replayParamUpdates_.size(),
                        execCtx.capturedTokenIdBufs_.size());
            }

            if (!convLayerIndices.empty()) {
                int64_t nel = convChannels * convLCache;
                convCastWorkgroups = (uint32_t)((nel + 255) / 256);
                uint32_t params[4] = {(uint32_t)nel, 0, 0, 0};
                auto paramBuf = gpu->createBuffer("conv_cast_params", 16);
                gpu->writeBuffer(paramBuf, params, 16);
                convCastPipeline = &executor.GetPipelineT("cast_f32_to_f16", 3,
                    []() { return std::string(WGSL_CAST_F32_TO_F16); });
                convCastBindGroups.resize(convLayerIndices.size());
                for (size_t i = 0; i < convLayerIndices.size(); i++) {
                    convCastBindGroups[i] = executor.MakeBindGroup(*convCastPipeline, {
                        {0, capturedConvOutputBufs[i]},
                        {1, convCastF16Bufs[i]},
                        {2, paramBuf}});
                }

                // Feed the capture step's outputs back into the stable input
                // buffers before the first replay. Subsequent replays queue
                // the same feedback after each captured graph execution.
                for (size_t i = 0; i < convLayerIndices.size(); i++) {
                    if (arch == "qwen3_5_text") {
                        execCtx.QueueCopy(capturedConvOutputBufs[i], 0,
                                          convCastF16Bufs[i], 0,
                                          static_cast<uint64_t>(convChannels) * convLCache * 4);
                    } else {
                        wgpuBindGroupAddRef(convCastBindGroups[i]);
                        execCtx.QueueDispatch(convCastPipeline->pipeline, convCastBindGroups[i],
                            convCastWorkgroups, 1, 1, "cache_cast_f16");
                    }
                    if (i < capturedRecurrentInputBufs.size()) {
                        const uint64_t bytes = static_cast<uint64_t>(recurrentHeads) *
                            recurrentKeyDim * recurrentValueDim * 4;
                        execCtx.QueueCopy(capturedRecurrentOutputBufs[i], 0,
                                          capturedRecurrentInputBufs[i], 0, bytes);
                    }
                }
                execCtx.SubmitPending();
            }

            return logits;
        }

        // Normal Path
        return runExecutePath(tokenId);
    }

    int32_t RunStepGreedy(int64_t tokenId) {
        requestGpuGreedyToken = true;
        auto logits = RunStep(tokenId);
        requestGpuGreedyToken = false;
        if (!logits.empty())
            return argmax(logits.data(), (int64_t)logits.size());
        return lastGpuGreedyToken;
    }

    std::string CheckFastDecodeSupport() const {
        auto& graph = executor.GetGraph();
        for (auto& node : graph.nodes) {
            if (node.opType == "If" || node.opType == "Loop" || node.opType == "Scan")
                return "model contains dynamic control flow (" + node.opType + " op: " + node.name + ")";
        }
        if (convLayerIndices.empty() && attnLayerIndices.empty())
            return "model has no conv or attention layers";
        return "";
    }

    void EnableFastDecode() {
        fastDecodeEnabled = true;
        convCastF16Bufs.resize(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            size_t nel = (size_t)(convChannels * convLCache);
            convCastF16Bufs[i] = gpu->createBuffer(
                "conv_replay_input_" + std::to_string(i),
                nel * (arch == "qwen3_5_text" ? 4 : 2));
        }
    }

    void WarmupPipelines() {
        // Qwen 3.5 uses generic ONNX specializations (including recurrent
        // kernels) that are compiled with runtime shapes. The legacy blanket
        // warmup includes unrelated fixed-shape shaders and is both invalid
        // for this graph and slower than lazy compilation.
        if (arch == "qwen3_5_text") return;
        auto t0 = std::chrono::steady_clock::now();
        auto& kernels = getEmbeddedKernels();

        std::vector<std::tuple<std::string, std::string, uint32_t>> specs;
        std::set<std::string> added;

        auto addKernel = [&](const std::string& name) {
            if (added.count(name)) return;
            auto it = kernels.find(name);
            if (it != kernels.end()) {
                specs.emplace_back(name, std::string(it->second.source), it->second.numBindings);
                added.insert(name);
            }
        };

        for (auto& node : executor.GetGraph().nodes) {
            if (node.opType == "MatMul" || node.opType == "Gemm") {
                addKernel("gemm"); addKernel("fp16_gemm");
            } else if (node.opType == "Conv") {
                addKernel("conv2d");
            } else if (node.opType == "ConvTranspose") {
                addKernel("conv_transpose2d");
            } else if (node.opType == "SimplifiedLayerNormalization" ||
                       node.opType == "SkipSimplifiedLayerNormalization") {
                addKernel("rms_norm"); addKernel("rms_norm_batched");
                addKernel("add_rms_norm"); addKernel("add_rms_norm_batched");
            } else if (node.opType == "LayerNormalization") {
                addKernel("layer_norm");
            } else if (node.opType == "Add" || node.opType == "Sub" ||
                       node.opType == "Mul" || node.opType == "Div") {
                addKernel("binary_elementwise");
            } else if (node.opType == "Relu" || node.opType == "Sigmoid" ||
                       node.opType == "Tanh" || node.opType == "Neg" ||
                       node.opType == "Cast" || node.opType == "Exp" ||
                       node.opType == "Sqrt" || node.opType == "Erf") {
                addKernel("unary_elementwise");
            } else if (node.opType == "Softmax") {
                addKernel("softmax");
            } else if (node.opType == "Gather") {
                addKernel("gather");
            } else if (node.opType == "Transpose") {
                addKernel("transpose");
            } else if (node.opType == "Where") {
                addKernel("where_select");
            } else if (node.opType == "Equal") {
                addKernel("equal_op");
            } else if (node.opType == "Expand") {
                addKernel("expand");
            } else if (node.opType == "Slice") {
                addKernel("slice");
            }
        }

        if (!attnLayerIndices.empty()) {
            addKernel("gqa_fused_attn"); addKernel("gqa_prefill");
            addKernel("rotary_embedding");
            addKernel("fused_qknorm_rope"); addKernel("fused_qknorm_rope_batched");
            addKernel("rope_batched_simple");
        }
        if (numExperts > 0) addKernel("silu_mul_fused");
        addKernel("argmax"); addKernel("embed_gather");

        if (!specs.empty()) {
            int compiled = gpu->warmupPipelines(specs);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            fprintf(stderr, "  Warmed %d/%zu GPU pipelines in %.0fms\n",
                   compiled, specs.size(), ms);
        }
    }

private:
    std::vector<float> runExecutePath(int64_t tokenId) {
        std::unordered_map<std::string, GpuTensor*> inputs;

        GpuTensor idT;
        idT.shape = {1, 1};
        idT.dtype = TensorDtype::Int64;
        idT.buffer = idsBuf;
        gpu->writeBuffer(idsBuf, &tokenId, 8);
        idT.cpuData.resize(8);
        memcpy(idT.cpuData.data(), &tokenId, 8);
        inputs["input_ids"] = &idT;
        GpuTensor embeddings;
        if (cpuEmbedding) {
            const int32_t token = static_cast<int32_t>(tokenId);
            PrepareEmbeddings(&token, 1, embeddings);
            inputs["inputs_embeds"] = &embeddings;
        }

        const bool intelQwen2 = arch == "qwen3_5_text" && numLayers == 24 &&
            gpu->adapterName.find("Intel") != std::string::npos;
        int64_t position = (int64_t)(pos - 1);
        std::vector<int64_t> positionValues(positionAxes,position);
        GpuTensor positionT;
        if (!intelQwen2) {
            positionT.shape = positionAxes==1 ? std::vector<int64_t>{1,1} : std::vector<int64_t>{positionAxes,1,1};
            positionT.dtype = TensorDtype::Int64;
            positionT.buffer = positionBuf;
            gpu->writeBuffer(positionBuf, positionValues.data(), positionValues.size()*sizeof(int64_t));
            positionT.cpuData.resize(positionValues.size()*sizeof(int64_t));
            memcpy(positionT.cpuData.data(), positionValues.data(), positionT.cpuData.size());
            inputs["position_ids"] = &positionT;
        }

        std::vector<int64_t> mask(pos, 1);
        GpuTensor maskT;
        maskT.shape = {1, (int64_t)pos};
        maskT.dtype = TensorDtype::Int64;
        maskT.buffer = maskBuf;
        gpu->writeBuffer(maskBuf, mask.data(), pos * 8);
        maskT.cpuData.resize(pos * 8);
        memcpy(maskT.cpuData.data(), mask.data(), pos * 8);
        inputs["attention_mask"] = &maskT;

        int64_t nlk = 1;
        GpuTensor nlkT;
        nlkT.shape = {};
        nlkT.dtype = TensorDtype::Int64;
        nlkT.buffer = nlkBuf;
        if (!nlkWritten) {
            gpu->writeBuffer(nlkBuf, &nlk, 8);
            nlkWritten = true;
        }
        nlkT.cpuData.resize(8);
        memcpy(nlkT.cpuData.data(), &nlk, 8);
        inputs["num_logits_to_keep"] = &nlkT;

        for (auto& [name, t] : convState) inputs[name] = &t;
        for (auto& [name, t] : recurrentState) inputs[name] = &t;
        for (auto& [name, t] : kvState) {
            t.shape = {1, numKvHeads, (int64_t)(pos - 1), headDim};
            inputs[name] = &t;
        }

        std::unordered_map<std::string, GpuTensor*> outputs;

        GpuTensor logitsOut;
        logitsOut.shape = {1, 1, vocabSize};
        logitsOut.dtype = TensorDtype::Float32;
        logitsOut.buffer = logitsBuf;
        outputs["logits"] = &logitsOut;

        std::vector<GpuTensor> convOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "present." + std::to_string(convLayerIndices[i]) + ".conv_state";
            convOuts[i].shape = {1, convChannels, convLCache};
            convOuts[i].dtype = TensorDtype::Float32;
            const std::string inputName = "past_key_values." +
                std::to_string(convLayerIndices[i]) + ".conv_state";
            convOuts[i].buffer = convState[inputName].buffer.handle == convOutBufs[i].handle
                ? convOutAltBufs[i] : convOutBufs[i];
            outputs[name] = &convOuts[i];
        }

        std::vector<GpuTensor> recurrentOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size() && recurrentHeads > 0; i++) {
            std::string name = "present." + std::to_string(convLayerIndices[i]) +
                               ".recurrent_state";
            recurrentOuts[i].shape = {1, recurrentHeads, recurrentKeyDim, recurrentValueDim};
            recurrentOuts[i].dtype = TensorDtype::Float32;
            const std::string inputName = "past_key_values." +
                std::to_string(convLayerIndices[i]) + ".recurrent_state";
            recurrentOuts[i].buffer =
                recurrentState[inputName].buffer.handle == recurrentOutBufs[i].handle
                    ? recurrentOutAltBufs[i] : recurrentOutBufs[i];
            outputs[name] = &recurrentOuts[i];
        }

        std::vector<GpuTensor> kvOuts(attnLayerIndices.size() * 2);
        for (size_t i = 0; i < attnLayerIndices.size(); i++) {
            std::string kName = "present." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vName = "present." + std::to_string(attnLayerIndices[i]) + ".value";
            std::string kIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".value";
            kvOuts[i*2].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2].dtype = TensorDtype::Float32;
            kvOuts[i*2].buffer = kvState[kIn].buffer;
            kvOuts[i*2+1].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2+1].dtype = TensorDtype::Float32;
            kvOuts[i*2+1].buffer = kvState[vIn].buffer;
            outputs[kName] = &kvOuts[i*2];
            outputs[vName] = &kvOuts[i*2+1];
        }

        uint64_t logitBytes = (uint64_t)vocabSize * 4;
        auto rbHandle = gpu->getOrCreateReadbackBuf(logitBytes);
        execCtx.RequestReadback(logitsBuf, {rbHandle, logitBytes}, logitBytes);

        ExecuteGraph(inputs, outputs);

        int64_t logitNel = logitsOut.ElementCount();
        auto logits = ReadLogits(logitNel);

        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string inName = "past_key_values." + std::to_string(convLayerIndices[i]) +
                                 ".conv_state";
            if (!convOuts[i].IsValid()) continue;
            convState[inName] = convOuts[i];
            if (i < recurrentOuts.size() && recurrentOuts[i].IsValid()) {
                std::string recurrentName = "past_key_values." +
                    std::to_string(convLayerIndices[i]) + ".recurrent_state";
                recurrentState[recurrentName] = recurrentOuts[i];
            }
        }

        for (int idx : attnLayerIndices) {
            std::string kIn = "past_key_values." + std::to_string(idx) + ".key";
            std::string vIn = "past_key_values." + std::to_string(idx) + ".value";
            kvState[kIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvState[vIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
        }

        execCtx.SubmitPending();
        return logits;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// StandardState (moved from LlmContext in apps/llm/main.cpp)
// ═══════════════════════════════════════════════════════════════════════════

struct StandardState {
    GPUContext* gpu = nullptr;
    ModelRunner runner;
    Tokenizer ggufTokenizer;
    OnnxTokenizer onnxTokenizer;
    std::string format;
    uint32_t pos = 0;
    bool benchWarmupDone = false;
    int pipelineInFlight = 0;
    uint32_t pipelineNextSubmitPos = 0;

    // A queued Qwen segment can advance recurrent state beyond logical pos.
    // Keep one exact checkpoint at its start, not a history of every prompt.
    struct QwenCheckpointRange { GPUBuffer state; uint64_t offset, bytes; };
    bool qwenCheckpointEnabled = false, qwenCheckpointActive = false;
    GPUBuffer qwenCheckpoint;
    std::vector<QwenCheckpointRange> qwenCheckpointRanges;
    std::vector<uint32_t> qwenCheckpointKvLengths;
    std::vector<int32_t> qwenCommittedInputs;
    uint32_t qwenCheckpointPos = 0;

    void CopyQwenCheckpoint(bool save) {
        const auto start = gpu->diagnosticTimestamp();
        WGPUCommandEncoderDescriptor encoderDesc{};
        auto encoder = wgpuDeviceCreateCommandEncoder(gpu->device, &encoderDesc);
        for (const auto& range : qwenCheckpointRanges) {
            if (save) {
                wgpuCommandEncoderCopyBufferToBuffer(encoder, range.state.handle,
                    range.state.offset, qwenCheckpoint.handle,
                    qwenCheckpoint.offset + range.offset, range.bytes);
            } else {
                wgpuCommandEncoderCopyBufferToBuffer(encoder, qwenCheckpoint.handle,
                    qwenCheckpoint.offset + range.offset, range.state.handle,
                    range.state.offset, range.bytes);
            }
        }
        WGPUCommandBufferDescriptor commandDesc{};
        auto command = wgpuCommandEncoderFinish(encoder, &commandDesc);
        gpu->recordEncode(start);
        if (gpu->diagnosticsEnabled) ++gpu->diagnostics.flushes;
        gpu->submitCommandBuffer(command);
        wgpuCommandBufferRelease(command);
        wgpuCommandEncoderRelease(encoder);
        // The same queue orders this copy before following decode work. Its
        // ordinary readback supplies completion; no extra queue wait is needed.
    }

    void BeginQwenQueuedSegment(int32_t inputToken) {
        if (!qwenCheckpointEnabled || qwenCheckpointActive) return;
        if (!qwenCheckpoint.handle) {
            const auto& c = runner.cfg;
            const uint64_t convBytes = (uint64_t(c.ssmInnerSize) +
                2ull * c.ssmGroupCount * c.ssmStateSize) * c.ssmConvKernel * 4;
            const uint64_t headV = c.ssmInnerSize / c.ssmTimeStepRank;
            const uint64_t recurrentBytes = uint64_t(c.ssmTimeStepRank) * headV * headV * 4;
            uint64_t total = 0;
            for (uint32_t li = 0; li < c.nLayer; ++li) {
                if (c.isAttentionLayer(li)) continue;
                for (const auto& state : {std::pair<GPUBuffer,uint64_t>{runner.ssmConvState.at(li), convBytes},
                                          {runner.ssmHState.at(li), recurrentBytes}}) {
                    if (!state.first.handle || state.first.size < state.second || !state.second)
                        throw std::runtime_error("Invalid Qwen checkpoint state extent");
                    qwenCheckpointRanges.push_back({state.first, total, state.second});
                    total += state.second;
                }
            }
            if (!total || total > 256ull * 1024 * 1024)
                throw std::runtime_error("Qwen checkpoint exceeds the validated state bound");
            qwenCheckpoint = runner.createOwnedBuffer("qwen_committed_state_checkpoint", total);
            if (!qwenCheckpoint.handle) throw std::runtime_error("Qwen checkpoint allocation failed");
            qwenCheckpointKvLengths.resize(runner.kvCache.size());
            qwenCommittedInputs.reserve(runner.maxSeqLen);
        }
        qwenCheckpointPos = pos;
        for (size_t i = 0; i < runner.kvCache.size(); ++i)
            qwenCheckpointKvLengths[i] = runner.kvCache[i].len;
        qwenCommittedInputs.clear();
        CopyQwenCheckpoint(true);
        // A preceding synchronous step writes the shared argmax output but
        // does not advance the slot-local token ring. Seed the new segment
        // explicitly from the token selected by the caller.
        runner.seedDecodeTokenInputs(inputToken);
        qwenCheckpointActive = true;
    }

    void RestoreQwenCommittedBoundary() {
        if (!qwenCheckpointEnabled) return;
        if (!pipelineInFlight) {
            qwenCheckpointActive = false;
            qwenCommittedInputs.clear();
            return;
        }
        const uint32_t boundary = pos;
        if (!qwenCheckpointActive || uint64_t(qwenCheckpointPos) + qwenCommittedInputs.size() != boundary)
            throw std::runtime_error("Qwen committed checkpoint boundary is inconsistent");
        const int depth = std::max(1, runner.decodePoolDepth);
        for (int i = 0; i < pipelineInFlight; ++i)
            (void)runner.readArgmax((pos + i) % depth);
        pipelineInFlight = 0;
        pipelineNextSubmitPos = 0;
        CopyQwenCheckpoint(false);
        for (size_t i = 0; i < runner.kvCache.size(); ++i)
            runner.kvCache[i].len = qwenCheckpointKvLengths[i];
        pos = qwenCheckpointPos;
        qwenCheckpointActive = false;
        int32_t next = -1;
        // Replay only inputs whose predictions were returned to the caller,
        // with the same per-position pooled arithmetic as the original stream.
        for (const int32_t token : qwenCommittedInputs) {
            const int slot = pos % depth;
            runner.seedDecodeTokenInputs(token);
            runner.submitDecode(pos++, slot);
            next = runner.readArgmax(slot);
        }
        qwenCommittedInputs.clear();
        if (next >= 0) runner.seedDecodeTokenInputs(next);
        if (pos != boundary || gpu->executionError || gpu->deviceLost)
            throw std::runtime_error("Qwen committed-state recovery failed");
    }

    std::string arch, gpuName, backendName;
    uint32_t nLayer=0, nHead=0, nKvHeads=0, nEmbd=0, headDim=0, nVocab=0;

    bool Load(GPUContext& gpuCtx, const std::string& path, int64_t maxSeqOverride) {
        gpu = &gpuCtx;
        if (maxSeqOverride < 0 || maxSeqOverride > UINT32_MAX) {
            fprintf(stderr, "Invalid context length\n");
            return false;
        }
        if (maxSeqOverride > 0) runner.maxSeqLen = (uint32_t)maxSeqOverride;
        std::string resolved = resolvePath(path, format);

        bool ok = (format == "onnx")
            ? runner.loadOnnx(gpuCtx, resolved)
            : runner.load(gpuCtx, resolved);
        if (!ok) return false;

        if (format == "onnx") {
            // Split GenAI models resolve to their decoder subdirectory while
            // tokenizer.json remains at the package root.  Select the existing
            // tokenizer directory before loading so a valid package does not
            // emit a misleading "Failed to open" diagnostic.
            fs::path tokenizerDir(resolved);
            if (!fs::exists(tokenizerDir / "tokenizer.json"))
                tokenizerDir = tokenizerDir.parent_path();
            if (!onnxTokenizer.load(tokenizerDir.string())) return false;
        } else {
            if (!ggufTokenizer.load(runner.gguf)) return false;
        }

        // Generic startup replay is unsafe for Qwen 3.5's hybrid recurrent
        // path: it advances SSM state before the real prompt and can corrupt
        // pooled decode ownership. Pipelines are already created by load().
        if (runner.cfg.arch != "qwen35") {
            if (runner.embeddingCPU.empty() || runner.pleGpuPreprocess) {
                runner.seedDecodeTokenInputs(0);
                runner.submitDecode(0, 0);
                (void)runner.readArgmax(0);
            } else {
                runner.decode(0, 0);
            }
            runner.resetKVCache();
        }
        if (runner.cfg.arch != "qwen35") {
            if (!runner.loadDecodeAutotuneCache()) {
                runner.autotuneDecodeDepth();
                runner.autotuneDecodeKernels();
                runner.saveDecodeAutotuneCache();
            }
        }

        auto& c = runner.cfg;
        arch = c.arch; nLayer = c.nLayer; nHead = c.nHead;
        nKvHeads = c.nKvHeads; nEmbd = c.nEmbd; headDim = c.headDim;
        nVocab = c.nVocab;
        gpuName = gpuCtx.adapterName;
        qwenCheckpointEnabled = format == "gguf" && c.arch == "qwen35" &&
            c.ssmInnerSize > 0 && c.ssmTimeStepRank > 0 &&
            ((c.nLayer == 24 && c.nEmbd == 2048) ||
             (c.nLayer == 32 && c.nEmbd == 2560) ||
             (c.nLayer == 64 && c.nEmbd == 5120)) &&
            gpuCtx.backendType == WGPUBackendType_D3D12 &&
            gpuCtx.adapterName == "NVIDIA GeForce RTX 5080";
        return true;
    }

    std::vector<int32_t> Tokenize(const std::string& text) {
        if (format == "onnx") {
            auto ids = onnxTokenizer.encode(text);
            // Gemma's shipped ONNX chat template begins with bos_token. The
            // tokenizer.json encoder does not add it automatically, unlike
            // the GGUF tokenizer's add_bos_token path below.
            if (runner.cfg.arch.rfind("gemma", 0) == 0 &&
                onnxTokenizer.bos_token_id >= 0 &&
                (ids.empty() || ids.front() != onnxTokenizer.bos_token_id)) {
                ids.insert(ids.begin(), onnxTokenizer.bos_token_id);
            }
            return ids;
        }
        auto ids = ggufTokenizer.encode(text);
        // Prepend BOS when the tokenizer flags add_bos_token (e.g. Gemma).
        // Tokens already starting with BOS (e.g. <bos> literal in the input)
        // would have hit the special-token matcher above, so we only add when
        // the first id is not the BOS id.
        if (ggufTokenizer.add_bos_token &&
            ggufTokenizer.bos_token_id >= 0 &&
            (ids.empty() || ids.front() != ggufTokenizer.bos_token_id)) {
            ids.insert(ids.begin(), ggufTokenizer.bos_token_id);
        }
        return ids;
    }

    std::string DetokenizeOne(int32_t tok) {
        return (format == "onnx") ? onnxTokenizer.decode_token(tok) : ggufTokenizer.decode_token(tok);
    }

    void DumpTopLogits(const char* stage, const std::vector<float>& logits) {
        const char* env = std::getenv("BP_DUMP_TOP_LOGITS");
        if (!env || !*env || logits.empty()) return;

        int k = std::atoi(env);
        if (k <= 0) k = 10;
        k = std::min<int>(k, (int)logits.size());

        std::vector<int32_t> ids(logits.size());
        for (int32_t i = 0; i < (int32_t)ids.size(); i++) ids[i] = i;
        std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
            [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });

        fprintf(stderr, "\n[debug] top %d logits after %s:\n", k, stage);
        for (int i = 0; i < k; i++) {
            int32_t id = ids[i];
            std::string text = DetokenizeOne(id);
            for (char& ch : text) {
                if (ch == '\n') ch = ' ';
                if (ch == '\r') ch = ' ';
                if (ch == '\t') ch = ' ';
            }
            fprintf(stderr, "  %2d: id=%d logit=% .6f text=\"%s\"\n",
                    i + 1, id, logits[id], text.c_str());
        }
    }

    int32_t Eos() {
        return (format == "onnx") ? onnxTokenizer.eos_token_id : ggufTokenizer.eos_token_id;
    }

    void RestoreGemmaCommittedBoundary() {
        if (!pipelineInFlight || runner.cfg.arch != "gemma4" ||
            runner.cfg.nLayer != 35 || runner.cfg.nEmbd != 1536 ||
            runner.cfg.nVocab != 262144 || runner.cfg.ssmInnerSize != 0 ||
            gpu->backendType != WGPUBackendType_D3D12 ||
            gpu->adapterName != "NVIDIA GeForce RTX 5080") return;
        // These Gemma caches are chronological, including sliding attention:
        // queued work writes only positions at or beyond the committed prefix.
        // Complete and unmap every pending slot before overwriting that suffix.
        // Recurrent models cannot recover by truncating KV lengths alone.
        const int depth = std::max(1, runner.decodePoolDepth);
        for (int i = 0; i < pipelineInFlight; ++i)
            (void)runner.readArgmax((pos + i) % depth);
        pipelineInFlight = 0;
        pipelineNextSubmitPos = 0;
        for (auto& cache : runner.kvCache) cache.len = pos;
    }

    int32_t Prefill(const int32_t* tokens, uint32_t n) {
        if (n > runner.maxSeqLen || pos > runner.maxSeqLen - n) {
            fprintf(stderr, "Prompt exceeds the configured context length (%u)\n", runner.maxSeqLen);
            return -1;
        }
        RestoreGemmaCommittedBoundary();
        RestoreQwenCommittedBoundary();
        if (std::getenv("BP_DUMP_TOKENS")) {
            fprintf(stderr, "[debug] prompt tokens:");
            for (uint32_t i = 0; i < n; i++) fprintf(stderr, " %d", tokens[i]);
            fprintf(stderr, "\n");
        }
        int32_t next;
        const uint32_t startPos = pos;
        bool pooledPrefill = runner.pleGpuPreprocess || runner.cfg.arch == "qwen35";
        bool qwenBatched = runner.cfg.arch == "qwen35" && n > 16 &&
                           runner.qwen35FastPrefill;
        // Validated Gemma E2B routes use batching on this device. The serial
        // override retains the previous route; GGUF additionally requires the
        // native-Q4 layout validated by the shared application checks.
        const bool gemmaOnnxBatched = format == "onnx" && runner.cfg.arch == "gemma4" &&
            runner.cfg.nLayer == 35 && runner.cfg.nEmbd == 1536 && runner.cfg.nVocab == 262144 &&
            runner.hasBatchedPrefill() && n > 16 && gpu->backendType == WGPUBackendType_D3D12 &&
            gpu->adapterName.find("RTX 5080") != std::string::npos &&
            !std::getenv("BP_GEMMA_SERIAL_PREFILL");
        const char* ggufProbe = std::getenv("BP_GEMMA_GGUF_BATCHED_PROBE");
        const bool gemmaGgufBatched = (!ggufProbe || std::strcmp(ggufProbe, "0") != 0) &&
            format == "gguf" && runner.cfg.arch == "gemma4" && runner.cfg.nLayer == 35 &&
            runner.cfg.nEmbd == 1536 && runner.cfg.nVocab == 262144 && runner.weightsAreNativeQ4 &&
            runner.hasBatchedPrefill() && n > 16 &&
            gpu->backendType == WGPUBackendType_D3D12 && gpu->adapterName == "NVIDIA GeForce RTX 5080" &&
            !std::getenv("BP_GEMMA_SERIAL_PREFILL");
        if ((qwenBatched || gemmaOnnxBatched || gemmaGgufBatched) && !std::getenv("BP_SYNC_PREFILL")) {
            next = runner.prefillBatched(tokens, n, startPos);
        } else if (pooledPrefill && !std::getenv("BP_SYNC_PREFILL")) {
            next = runner.prefillPooledKnown(tokens, n, startPos);
        } else {
            std::vector<float> logits;
            for (uint32_t i = 0; i < n; i++) logits = runner.decode(tokens[i], startPos + i);
            DumpTopLogits("prefill", logits);
            next = ModelRunner::argmax(logits);
        }
        if (const char* path = std::getenv("BP_DUMP_PREFILL_LOGITS")) {
            auto bytes = gpu->readBuffer(runner.logitsBuf, uint64_t(nVocab) * sizeof(float));
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        runner.seedDecodeTokenInputs(next);
        pos += n;
        return next;
    }

    int32_t DecodePipelined(int32_t inputToken) {
        if (pos >= runner.maxSeqLen) return -1;
        BeginQwenQueuedSegment(inputToken);
        int depth = runner.decodePoolDepth;

        if (pipelineInFlight == 0) {
            const int available = std::min<uint32_t>(depth, runner.maxSeqLen - pos);
            for (int i = 0; i < available; i++) {
                int slot = (pos + i) % depth;
                runner.submitDecode(pos + i, slot);
            }
            pipelineInFlight = available;
            pipelineNextSubmitPos = pos + available;
        }

        int readSlot = pos % depth;
        int32_t tok = runner.readArgmax(readSlot);
        pipelineInFlight--;

        if (pipelineNextSubmitPos < runner.maxSeqLen) {
            runner.submitDecode(pipelineNextSubmitPos, readSlot);
            pipelineNextSubmitPos++;
            pipelineInFlight++;
        }

        pos++;
        if (qwenCheckpointEnabled) qwenCommittedInputs.push_back(inputToken);
        return tok;
    }

    std::vector<float> DecodeSynchronous(int32_t token) {
        if (pos >= runner.maxSeqLen) return {};
        RestoreGemmaCommittedBoundary();
        RestoreQwenCommittedBoundary();
        auto logits = runner.decode(token, pos);
        DumpTopLogits("decode", logits);
        pos++;
        return logits;
    }

    int32_t DecodeArgmaxSynchronous(int32_t token) {
        if (pos >= runner.maxSeqLen) return -1;
        if (std::getenv("BP_DUMP_TOP_LOGITS")) {
            auto logits = DecodeSynchronous(token);
            return ModelRunner::argmax(logits);
        }
        RestoreGemmaCommittedBoundary();
        RestoreQwenCommittedBoundary();
        int32_t next = runner.decodeArgmax(token, pos);
        pos++;
        return next;
    }

    void Reset() {
        // Streaming decode submits ahead of the last token returned to the
        // caller. Complete those maps before prefill reuses the staging ring.
        // Queue completion alone does not unmap the buffers.
        const int depth = std::max(1, runner.decodePoolDepth);
        for (int i = 0; i < pipelineInFlight; ++i)
            (void)runner.readArgmax((pos + i) % depth);
        pipelineInFlight = 0;
        pipelineNextSubmitPos = 0;
        runner.resetKVCache();
        pos = 0;
        qwenCheckpointActive = false;
        qwenCommittedInputs.clear();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// LmSession::Impl
// ═══════════════════════════════════════════════════════════════════════════

struct LmSession::Impl {
    Device* device = nullptr;
    GPUContext* gpu = nullptr;
    LmConfig config;
    LmOptions options;

    enum class Backend { Standard, GenericOnnx } backend;

    std::unique_ptr<StandardState> std_;
    std::unique_ptr<GenericOnnxState> gen_;

    // Last decode token (for pipelined decode on standard path)
    int32_t lastToken = -1;
};

// ═══════════════════════════════════════════════════════════════════════════
// Factory
// ═══════════════════════════════════════════════════════════════════════════

LmSession LmSession::Create(Device& device, const std::string& modelPath,
                             const LmOptions& options) {
    return Create(device, modelPath, "", options);
}

LmSession LmSession::Create(Device& device, const std::string& modelPath,
                             const std::string& format,
                             const LmOptions& options) {
    if (!device.IsValid()) return {};

    auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
    if (!gpuCtx) return {};

    std::string modelFormat;
    std::string resolved = resolvePath(modelPath, modelFormat, format);

    LmSession session;
    session.impl_ = std::make_unique<Impl>();
    session.impl_->device = &device;
    session.impl_->gpu = gpuCtx;
    session.impl_->options = options;

    if (modelFormat == "onnx_generic") {
        // Generic ONNX backend (LFM2, conv+MoE, etc.)
        session.impl_->backend = Impl::Backend::GenericOnnx;
        session.impl_->gen_ = std::make_unique<GenericOnnxState>();

        if (!session.impl_->gen_->Load(*gpuCtx, resolved, options.maxSeqLen)) {
            fprintf(stderr, "bp::LmSession: failed to load generic ONNX model\n");
            return {};
        }

        auto* gen = session.impl_->gen_.get();
        gen->prefillChunkSize = options.prefillChunkSize;

        // Auto-enable fast decode
        if (options.fastDecode) {
            std::string reason = gen->CheckFastDecodeSupport();
            if (reason.empty()) {
                gen->EnableFastDecode();
                fprintf(stderr, "  Fast decode: enabled\n");
            } else {
                fprintf(stderr, "  Fast decode: disabled — %s\n", reason.c_str());
            }
        }

        // Pre-compile GPU pipelines
        if (options.warmupPipelines)
            gen->WarmupPipelines();

        // Populate config
        auto& cfg = session.impl_->config;
        cfg.arch = gen->arch;
        cfg.format = "onnx_generic";
        cfg.layers = (int)gen->numLayers;
        cfg.hiddenSize = (int)gen->hiddenSize;
        cfg.vocabSize = (int)gen->vocabSize;
        cfg.numHeads = (int)gen->numHeads;
        cfg.numKvHeads = (int)gen->numKvHeads;
        cfg.headDim = (int)gen->headDim;
        cfg.maxSeqLen = gen->maxSeqLen;

    } else {
        // Standard transformer backend (GGUF or standard ONNX)
        session.impl_->backend = Impl::Backend::Standard;
        session.impl_->std_ = std::make_unique<StandardState>();

        if (!session.impl_->std_->Load(*gpuCtx, modelPath, options.maxSeqLen)) {
            fprintf(stderr, "bp::LmSession: failed to load model\n");
            return {};
        }

        auto* std = session.impl_->std_.get();

        auto& cfg = session.impl_->config;
        cfg.arch = std->arch;
        cfg.format = std->format;
        cfg.layers = (int)std->nLayer;
        cfg.hiddenSize = (int)std->nEmbd;
        cfg.vocabSize = (int)std->nVocab;
        cfg.numHeads = (int)std->nHead;
        cfg.numKvHeads = (int)std->nKvHeads;
        cfg.headDim = (int)std->headDim;
        cfg.maxSeqLen = std->runner.maxSeqLen;
    }

    return session;
}

// ═══════════════════════════════════════════════════════════════════════════
// Metadata
// ═══════════════════════════════════════════════════════════════════════════

LmConfig LmSession::GetConfig() const {
    return impl_ ? impl_->config : LmConfig{};
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int32_t> LmSession::Tokenize(const std::string& text) const {
    if (!impl_) return {};
    if (impl_->backend == Impl::Backend::GenericOnnx)
        return impl_->gen_->tokenizer.encode(text);
    return impl_->std_->Tokenize(text);
}

std::vector<int32_t> LmSession::TokenizeRaw(const std::string& text) const {
    if (!impl_) return {};
    if (impl_->backend == Impl::Backend::GenericOnnx)
        return impl_->gen_->tokenizer.encode(text);
    auto* state = impl_->std_.get();
    return state->format == "onnx" ? state->onnxTokenizer.encode(text) : state->ggufTokenizer.encode(text);
}

std::string LmSession::Detokenize(int32_t tokenId) const {
    if (!impl_) return {};
    if (impl_->backend == Impl::Backend::GenericOnnx)
        return impl_->gen_->tokenizer.decode_token(tokenId);
    return impl_->std_->DetokenizeOne(tokenId);
}

std::string LmSession::Detokenize(const std::vector<int32_t>& tokenIds) const {
    if (!impl_) return {};
    std::string result;
    for (auto id : tokenIds) result += Detokenize(id);
    return result;
}

int32_t LmSession::GetEosTokenId() const {
    if (!impl_) return -1;
    if (impl_->backend == Impl::Backend::GenericOnnx)
        return impl_->gen_->tokenizer.eos_token_id;
    return impl_->std_->Eos();
}

// ═══════════════════════════════════════════════════════════════════════════
// High-level Generation
// ═══════════════════════════════════════════════════════════════════════════

std::string LmSession::Generate(const std::string& prompt, int maxTokens,
                                 const SamplingParams& sampling,
                                 StreamCallback onToken,
                                 bool resetSession) {
    if (!impl_ || maxTokens <= 0) return {};

    if (resetSession) Reset();
    auto tokens = Tokenize(prompt);
    if (tokens.empty()) return {};

    int32_t next = Prefill(tokens.data(), (uint32_t)tokens.size());
    if (impl_->gpu->deviceLost) return {};
    auto isEnd = [&](int32_t token) {
        if (impl_->backend == Impl::Backend::GenericOnnx) return impl_->gen_->tokenizer.is_end_token(token);
        if (impl_->std_->format == "onnx") return impl_->std_->onnxTokenizer.is_end_token(token);
        return impl_->std_->ggufTokenizer.is_end_token(token);
    };

    bool useSampling = (sampling.temperature > 0.0f);
    std::mt19937 rng(sampling.seed ? sampling.seed : std::random_device{}());

    // Prefill already produced the first prediction. Reading its logits must
    // not consume that greedy prediction before the sampled token is selected.
    if (useSampling) {
        std::vector<float> logits;
        if (impl_->backend == Impl::Backend::GenericOnnx) {
            logits = impl_->gen_->CurrentLogits();
        } else {
            auto* state = impl_->std_.get();
            const auto bytes = impl_->gpu->readBuffer(state->runner.logitsBuf,
                                                     uint64_t(state->nVocab) * sizeof(float));
            logits.resize(state->nVocab);
            memcpy(logits.data(), bytes.data(), bytes.size());
        }
        if (logits.empty()) return {};
        next = sampleToken(logits.data(), (uint32_t)logits.size(),
                           sampling.temperature, sampling.topK, rng);
        impl_->lastToken = next;
    }

    std::string result;
    for (int i = 0; i < maxTokens; i++) {
        if (impl_->gpu->deviceLost || next < 0 || isEnd(next)) break;
        std::string text = Detokenize(next);
        // Skip special tokens (enclosed in < >)
        if (!(text.size() >= 2 && text[0] == '<' && text.back() == '>')) {
            result += text;
            if (onToken && !onToken(text)) break;
        }

        if (useSampling) {
            auto logits = DecodeLogits();
            if (logits.empty()) break;
            next = sampleToken(logits.data(), (uint32_t)logits.size(),
                               sampling.temperature, sampling.topK, rng);
            impl_->lastToken = next;
        } else {
            next = Decode();
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Low-level Stepping
// ═══════════════════════════════════════════════════════════════════════════

int32_t LmSession::Prefill(const int32_t* tokens, uint32_t count) {
    if (!impl_ || count == 0) return -1;

    if (impl_->backend == Impl::Backend::GenericOnnx) {
        auto* gen = impl_->gen_.get();
        int32_t next = gen->RunPrefillBatch(tokens, count);
        gen->prefillDone = true;
        impl_->lastToken = next;
        gen->DumpReferenceState();
        return next;
    }

    auto* std = impl_->std_.get();
    int32_t next = std->Prefill(tokens, count);
    impl_->lastToken = next;
    return next;
}

int32_t LmSession::Decode() {
    if (!impl_) return -1;

    if (impl_->backend == Impl::Backend::GenericOnnx) {
        auto* gen = impl_->gen_.get();
        if (gen->pos >= gen->maxSeqLen) return -1;
        int32_t next = gen->RunStepGreedy(impl_->lastToken);
        impl_->lastToken = next;
        gen->DumpReferenceState();
        return next;
    }

    // Qwen3.5 uses the queued decode path after fixing slot-local dynamic
    // params and GPU embedding gather. Keep a synchronous fallback for
    // validation or bisecting correctness regressions.
    auto* std = impl_->std_.get();
    if (std->pos >= std->runner.maxSeqLen) return -1;
    int32_t next;
    bool q35Sync = std->runner.cfg.arch == "qwen35" &&
                   std::getenv("BP_Q35_SYNC") != nullptr;
    // Gemma models (sandwich norms, per-layer head dims, shared-KV) don't fit
    // the pre-recorded pooled decode path; use synchronous decode.
    bool gemma4Sync = std->runner.cfg.arch.rfind("gemma", 0) == 0 &&
                      !std->runner.pleGpuPreprocess;
    if (q35Sync || gemma4Sync) {
        next = gemma4Sync && std::getenv("BP_GEMMA_SYNC") == nullptr
            ? std->runner.decodeArgmaxPooled(impl_->lastToken, std->pos++)
            : std->DecodeArgmaxSynchronous(impl_->lastToken);
    } else {
        next = std->DecodePipelined(impl_->lastToken);
    }
    impl_->lastToken = next;
    return next;
}

std::vector<float> LmSession::DecodeLogits() {
    if (!impl_) return {};

    if (impl_->backend == Impl::Backend::GenericOnnx) {
        if (impl_->gen_->pos >= impl_->gen_->maxSeqLen) return {};
        auto* gen = impl_->gen_.get();
        auto logits = gen->RunStep(impl_->lastToken);
        return logits;
    }

    // Standard path: synchronous decode (returns logits for sampling)
    auto* std = impl_->std_.get();
    return std->DecodeSynchronous(impl_->lastToken);
}

int32_t LmSession::DecodeWithMTP(std::vector<int32_t>& acceptedTokens, int maxDraftTokens) {
    if (!impl_) return 0;

    // If MTP is not available, fall back to regular decode
    if (!HasMTP()) {
        int32_t next = Decode();
        if (next >= 0) acceptedTokens.push_back(next);
        return next >= 0 ? 1 : 0;
    }

    // MTP speculative decoding:
    // 1. Draft N tokens using MTP head
    // 2. Verify all drafts in a single batched forward pass
    // 3. Accept matching tokens

    auto* std = impl_->std_.get();
    if (!std) {
        int32_t next = Decode();
        if (next >= 0) acceptedTokens.push_back(next);
        return next >= 0 ? 1 : 0;
    }

    // Step 1: Get the base token via normal decode
    int32_t baseToken = std->DecodePipelined(impl_->lastToken);
    if (baseToken < 0) return 0;

    // Step 2: Draft tokens using MTP
    std::vector<int32_t> drafts;
    std->runner.mtpDraft(baseToken, std->pos, drafts);

    if (drafts.empty()) {
        acceptedTokens.push_back(baseToken);
        impl_->lastToken = baseToken;
        return 1;
    }

    // Step 3: Verify drafts
    uint32_t accepted = 0;
    int32_t correction = std->runner.mtpVerifyAndAccept(drafts, std->pos, accepted);

    // Always accept the base token
    acceptedTokens.push_back(baseToken);

    // Accept verified draft tokens
    for (uint32_t i = 0; i < accepted; i++) {
        acceptedTokens.push_back(drafts[i]);
        std->pos++;
    }

    // If we got a correction token (from the verification step), accept it too
    if (correction >= 0 && accepted < (uint32_t)drafts.size()) {
        acceptedTokens.push_back(correction);
        std->pos++;
    }

    impl_->lastToken = acceptedTokens.back();
    return (int32_t)acceptedTokens.size();
}

bool LmSession::HasMTP() const {
    if (!impl_) return false;
    if (impl_->backend == Impl::Backend::Standard && impl_->std_) {
        return impl_->std_->runner.mtpCfg.type != ModelRunner::MTPType::None;
    }
    return false;
}

void LmSession::Reset() {
    if (!impl_) return;
    impl_->lastToken = -1;
    if (impl_->backend == Impl::Backend::GenericOnnx)
        impl_->gen_->ResetCaches();
    else
        impl_->std_->Reset();
}

uint32_t LmSession::GetPosition() const {
    if (!impl_) return 0;
    if (impl_->backend == Impl::Backend::GenericOnnx)
        return impl_->gen_->pos;
    return impl_->std_->pos;
}

uint64_t LmSession::TrimMemory() {
    if (!impl_) return 0;
    impl_->gpu->waitForQueue();
    const auto bytes = impl_->gpu->pooledBufferBytes();
    impl_->gpu->flushBufferPool();
    return bytes;
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmarking + Profiling
// ═══════════════════════════════════════════════════════════════════════════

BenchmarkResult LmSession::Benchmark(int promptLen, int genTokens) {
    if (!impl_ || promptLen <= 0) return {};
    if (std::getenv("BP_BENCH_PATTERN_TOKENS")) {
        std::vector<int32_t> tokens(promptLen);
        for (int i=0;i<promptLen;++i)
            tokens[i]=42+(i*7919)%std::max(1,impl_->config.vocabSize-42);
        return BenchmarkTokens(tokens,genTokens,1);
    }
    std::string text = "A";
    for (int i = 1; i < promptLen; ++i) text += " A";
    auto tokens = TokenizeRaw(text);
    if (tokens.size() != static_cast<size_t>(promptLen))
        throw std::runtime_error("Default benchmark text has a different token count; supply exact prompt tokens");
    return BenchmarkTokens(tokens, genTokens, 1);
}

BenchmarkResult LmSession::BenchmarkTokens(const std::vector<int32_t>& promptTokens,
                                         int genTokens, int warmupRuns) {
    if (!impl_) return {};
    const int promptLen = static_cast<int>(promptTokens.size());
    if (promptLen <= 0 || genTokens <= 0 || warmupRuns < 0 ||
        int64_t(promptLen) + genTokens > impl_->config.maxSeqLen)
        return {};
    using Clock = std::chrono::steady_clock;
    auto run = [&]() {
        Reset();
        BenchmarkResult result;
        result.promptLen = promptLen;
        result.tokenIds.reserve(genTokens);
        const auto prefillStart = Clock::now();
        // Use the application's selected prefill implementation and options.
        // Calling an unrelated batched runner path here can produce timings for
        // a path that the application never uses (and may not be conformant).
        int32_t token = Prefill(promptTokens.data(), static_cast<uint32_t>(promptLen));
        const auto prefillEnd = Clock::now();
        if (token < 0 || impl_->gpu->deviceLost || impl_->gpu->executionError)
            throw std::runtime_error("Benchmark prefill failed");
        result.tokenIds.push_back(token);
        result.prefillMs = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();
        result.prefillTokPerSec = promptLen * 1000.0 / result.prefillMs;
        result.ttftMs = result.prefillMs;
        const int steps = genTokens - 1;
        impl_->gpu->timing.wait_ns = 0;
        const auto decodeStart = Clock::now();
        if (impl_->backend == Impl::Backend::GenericOnnx) {
            auto* gen = impl_->gen_.get();
            // Capture setup belongs to the requested continuation. Do not
            // consume unreported warmup tokens in the measured conversation.
            for (int i = 0; i < steps; ++i) {
                token = gen->RunStepGreedy(token);
                if (token < 0 || impl_->gpu->deviceLost || impl_->gpu->executionError)
                    throw std::runtime_error("Benchmark decode failed");
                result.tokenIds.push_back(token);
                ++result.decodeSampleTokens;
            }
        } else if (steps > 0) {
            auto* state = impl_->std_.get();
            const int depth = std::max(1, state->runner.decodePoolDepth);
            int submitted = 0, completed = 0;
            auto submit = [&]() {
                const uint32_t position = static_cast<uint32_t>(promptLen + submitted);
                state->runner.submitDecode(position, position % depth);
                ++submitted;
            };
            for (int i = 0; i < std::min(depth, steps); ++i) submit();
            while (completed < submitted) {
                token = state->runner.readArgmax((promptLen + completed) % depth);
                if (token < 0 || impl_->gpu->deviceLost || impl_->gpu->executionError)
                    throw std::runtime_error("Benchmark decode failed");
                result.tokenIds.push_back(token);
                ++completed;
                ++state->pos;
                ++result.decodeSampleTokens;
                if (submitted < steps) submit();
            }
        }
        const auto decodeEnd = Clock::now();
        result.decodeMs = steps ? std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count() : 0;
        result.decodeTokPerSec = steps ? steps * 1000.0 / result.decodeMs : 0;
        result.fenceWaitMs = impl_->gpu->timing.wait_ns / 1e6;
        result.generatedTokens = static_cast<int>(result.tokenIds.size());
        result.finalPosition = GetPosition();
        if (result.generatedTokens != genTokens || result.decodeSampleTokens != steps ||
            result.finalPosition != static_cast<uint32_t>(promptLen + steps))
            throw std::runtime_error("Benchmark did not execute the exact workload");
        return result;
    };
    try {
        std::vector<int32_t> warmupTokens;
        for (int i = 0; i < warmupRuns; ++i) {
            auto warm = run();
            if (i && warm.tokenIds != warmupTokens)
                throw std::runtime_error("Benchmark warmup continuation is not deterministic");
            warmupTokens = std::move(warm.tokenIds);
        }
        auto result = run();
        if (warmupRuns && result.tokenIds != warmupTokens)
            throw std::runtime_error("Benchmark continuation differs after warmup/reset");
        Reset();
        return result;
    } catch (...) {
        Reset();
        throw;
    }
}

void LmSession::EnableProfiling() {
    if (!impl_) return;
    if (impl_->backend == Impl::Backend::GenericOnnx)
        impl_->gen_->execCtx.enableGpuProfiling();
    else
        impl_->std_->runner.enableProfiling();
}

void LmSession::FinishProfiling(const std::string& htmlPath, int measuredTokens, double elapsedMs,
                               bool prefill) {
    if (!impl_ || measuredTokens <= 0) return;
    if (impl_->backend == Impl::Backend::GenericOnnx)
        impl_->gen_->execCtx.printGpuProfileReport(measuredTokens, elapsedMs, htmlPath);
    else
        impl_->std_->runner.printProfileReport(prefill ? 0 : measuredTokens,
            prefill ? measuredTokens : 0, prefill ? elapsedMs : 0,
            prefill ? 0 : elapsedMs, htmlPath);
}

void LmSession::PrintProfileReport(const std::string& htmlPath) {
    if (!impl_) return;
    auto prefillProfileTokens=[](){
        constexpr uint32_t fallback=64;
        const char* value=std::getenv("BP_PROFILE_PREFILL_TOKENS");
        if(!value||!*value)return fallback;
        char* end=nullptr;unsigned long parsed=std::strtoul(value,&end,10);
        return end!=value&&*end=='\0'&&parsed>=1&&parsed<=4096
            ?static_cast<uint32_t>(parsed):fallback;
    };
    if (impl_->backend == Impl::Backend::GenericOnnx) {
        auto* gen = impl_->gen_.get();
        if (std::getenv("BP_PROFILE_PREFILL")) {
            const uint32_t kProfileTokens = prefillProfileTokens();
            std::vector<int32_t> tokens(kProfileTokens, 1);
            gen->ResetCaches();
            gen->execCtx.enableGpuProfiling();
            auto pt0 = std::chrono::steady_clock::now();
            gen->RunPrefillBatch(tokens.data(), kProfileTokens);
            auto pt1 = std::chrono::steady_clock::now();
            double profMs = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
            gen->execCtx.printGpuProfileReport(kProfileTokens, profMs, htmlPath);
            return;
        }

        // Run a profiled decode step.
        gen->ResetCaches();
        int32_t tok = 1;
        auto logits = gen->RunPrefillStep(tok);
        tok = argmax(logits.data(), (int64_t)logits.size());
        gen->prefillDone = true;
        for (int i = 0; i < 3; i++) {
            auto lg = gen->RunStep(tok);
            tok = argmax(lg.data(), (int64_t)lg.size());
        }
        gen->execCtx.enableGpuProfiling();
        auto pt0 = std::chrono::steady_clock::now();
        logits = gen->RunStep(tok);
        tok = argmax(logits.data(), (int64_t)logits.size());
        auto pt1 = std::chrono::steady_clock::now();
        double profMs = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
        gen->execCtx.printGpuProfileReport(1, profMs, htmlPath);
    } else {
        auto* st = impl_->std_.get();
        if (!st->runner.profiler)
            st->runner.enableProfiling();
        if (!st->runner.profiler)
            return;

        if (std::getenv("BP_PROFILE_PREFILL")) {
            const uint32_t kProfileTokens = prefillProfileTokens();
            st->Reset();
            st->runner.profiler->nextIndex = 0;
            st->runner.profiler->entries.clear();
            std::vector<int32_t> tokens(kProfileTokens, 1);
            auto pt0 = std::chrono::steady_clock::now();
            (void)st->runner.prefillBatched(tokens.data(), kProfileTokens, 0);
            auto pt1 = std::chrono::steady_clock::now();
            double profMs = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
            st->runner.printProfileReport(0, kProfileTokens, profMs, 0.0, htmlPath);
            st->Reset();
            return;
        }

        st->Reset();
        std::vector<float> logits;
        int32_t tok = 0;
        uint32_t warmupTokens = 5;
        for (uint32_t i = 0; i < warmupTokens; i++) {
            logits = st->runner.decode(tok, i);
            tok = ModelRunner::argmax(logits);
        }
        if (st->runner.profiler) {
            st->runner.profiler->nextIndex = 0;
            st->runner.profiler->entries.clear();
        }
        auto pt0 = std::chrono::steady_clock::now();
        logits = st->runner.decode(tok, warmupTokens);
        tok = ModelRunner::argmax(logits);
        (void)tok;
        auto pt1 = std::chrono::steady_clock::now();
        double profMs = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
        st->runner.printProfileReport(1, (int)warmupTokens, 0.0, profMs, htmlPath);
        st->Reset();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════════════

void LmSession::Release() { impl_.reset(); }
LmSession::LmSession() = default;
LmSession::~LmSession() = default;
LmSession::LmSession(LmSession&& o) noexcept = default;
LmSession& LmSession::operator=(LmSession&& o) noexcept = default;

} // namespace bp
