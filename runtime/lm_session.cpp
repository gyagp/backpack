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
#include "wgsl_shaders.h"
#include "json_parser.h"

#include <algorithm>
#include <chrono>
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
    return s.find("\"layer_types\"") != std::string::npos ||
           s.find("\"conv_L_cache\"") != std::string::npos;
}

static std::string findOnnxFile(const std::string& dir) {
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

    int64_t hiddenSize = 0, numLayers = 0, vocabSize = 0;
    int64_t numHeads = 0, numKvHeads = 0, headDim = 0;
    int64_t convLCache = 3;
    int64_t maxSeqLen = 0;
    int64_t intermediateSize = 0;
    int64_t convChannels = 0;
    int64_t recurrentHeads = 0, recurrentKeyDim = 0, recurrentValueDim = 0;
    int64_t moeIntermediateSize = 0;
    int64_t numExperts = 0, numExpertsPerTok = 0;
    float normEps = 1e-5f;
    float ropeTheta = 1000000.0f;
    std::vector<std::string> layerTypes;
    std::string arch;

    uint32_t pos = 0;
    std::unordered_map<std::string, GpuTensor> convState;
    std::unordered_map<std::string, GpuTensor> recurrentState;
    std::unordered_map<std::string, GpuTensor> kvState;
    std::vector<int> convLayerIndices, attnLayerIndices;

    GPUBuffer idsBuf, positionBuf, maskBuf, nlkBuf, logitsBuf;
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
        std::vector<ExecutionContext::CapturedWrite> writes;
        std::vector<ExecutionContext::ReplayParamUpdate> params;
        std::vector<ExecutionContext::ReplayScalarUpdate> scalars;
        std::vector<ExecutionContext::CapturedTokenIdBuf> tokenIds;
        std::vector<WGPUBuffer> skipBuffers;
        GPUBuffer logits;
    } qwenCaptureVariants[4];
    int qwenCapturedVariants = 0;
    int qwenActiveVariant = -1;
    int qwenNextReplayVariant = 0;

    void SwapCaptureVariant(int index) {
        auto& variant = qwenCaptureVariants[index];
        execCtx.capturedFlushes_.swap(variant.flushes);
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
        headDim = hiddenSize / numHeads;
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

    void ResetCaches() {
        pos = 0;
        prefillDone = false;
        decodePlanInitialized = false;
        decodeWarmupRemaining = 0;
        nlkWritten = false;
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
        if (fastDecodeCaptured) {
            execCtx.ReleaseCaptured();
            fastDecodeCaptured = false;
        }
        execCtx.InvalidateWarmCaches();

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
                t.buffer = gpu->createBuffer(name, kvBytes);
                kvState[name] = t;
            }
        }

        idsBuf = gpu->createBuffer("ids", 8);
        positionBuf = gpu->createBuffer("position_ids", maxSeqLen * 8);
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
    }

    std::vector<float> RunPrefillStep(int64_t tokenId) {
        pos++;
        return runExecutePath(tokenId);
    }

    int32_t RunPrefillBatch(const int32_t* tokenIds, uint32_t T) {
        if (T == 0) return -1;
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

        // NVIDIA and AMD use the sequence-causal Conv/LinearAttention graph.
        // AMD's GQA dispatch is selected separately in attention.cpp and uses
        // the same subgroup-plus-workgroup reduction as conformant decode;
        // this does not require subgroup matrices (unavailable on Windows).
        const bool deviceQwenBatch = arch == "qwen3_5_text" &&
            (gpu->adapterName.find("NVIDIA") != std::string::npos ||
             gpu->adapterName.find("AMD") != std::string::npos) &&
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
        idT.buffer = gpu->createBuffer("pf_ids", T * 8);
        gpu->writeBuffer(idT.buffer, ids64.data(), T * 8);
        idT.cpuData.resize(T * 8);
        memcpy(idT.cpuData.data(), ids64.data(), T * 8);
        inputs["input_ids"] = &idT;

        const bool intelQwen2 = arch == "qwen3_5_text" && numLayers == 24 &&
            gpu->adapterName.find("Intel") != std::string::npos;
        GpuTensor positionT;
        std::vector<int64_t> positions;
        if (!intelQwen2) {
            positions.resize(T);
            for (uint32_t i = 0; i < T; i++)
                positions[i] = (int64_t)(pos - T + i);
            positionT.shape = {1, (int64_t)T};
            positionT.dtype = TensorDtype::Int64;
            positionT.buffer = positionBuf;
            gpu->writeBuffer(positionBuf, positions.data(), T * 8);
            positionT.cpuData.resize(T * 8);
            memcpy(positionT.cpuData.data(), positions.data(), T * 8);
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
        executor.Execute(execCtx, inputs, outputs);

        int64_t logitNel = logitsOut.ElementCount();
        std::vector<float> logits(logitNel);
        auto rb = gpu->mapReadbackBuffer(logitNel * 4);
        memcpy(logits.data(), rb.data(), logitNel * 4);
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
        gpu->releaseBuffer(idT.buffer);

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
                if (!write.handle || write.data.empty()) continue;
                gpu->writeBufferRaw(write.handle, write.offset,
                                    write.data.data(), write.data.size());
            }
            const int64_t replayToken = tokenId;
            const int64_t replayPosition = static_cast<int64_t>(execCtx.replayPosition_);
            gpu->writeBufferRaw(idsBuf.handle, idsBuf.offset,
                                &replayToken, sizeof(replayToken));
            gpu->writeBufferRaw(positionBuf.handle, positionBuf.offset,
                                &replayPosition, sizeof(replayPosition));
            const int64_t maskOne = 1;
            gpu->writeBufferRaw(maskBuf.handle,
                maskBuf.offset + static_cast<uint64_t>(execCtx.replayPosition_) * 8,
                &maskOne, sizeof(maskOne));
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

            uint64_t logitBytes = (uint64_t)vocabSize * 4;
            auto rbHandle = gpu->getOrCreateReadbackBuf(logitBytes);
            execCtx.RequestReadback(logitsBuf, {rbHandle, logitBytes}, logitBytes);
            execCtx.ReplayDispatches();

            std::vector<float> logits(vocabSize);
            auto rb = gpu->mapReadbackBuffer(vocabSize * 4);
            memcpy(logits.data(), rb.data(), vocabSize * 4);

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
                int nDisp = 0;
                for (auto& f : execCtx.capturedFlushes_)
                    nDisp += static_cast<int>(f.dispatches.size());
                fprintf(stderr,
                    "  [fast decode capture %d/2] %zu flushes, %d dispatches, %zu param updates\n",
                    variant + 1,
                    execCtx.capturedFlushes_.size(), nDisp,
                    execCtx.replayParamUpdates_.size());
                StoreCurrentQwenCapture(variant);
                qwenCapturedVariants++;
                execCtx.ResetParamPoolCursors();
                // Qwen's recurrent and convolution state buffers ping-pong
                // every token. Preserve both binding parities and alternate
                // them during replay, just as the ordinary graph does.
                const int requiredVariants = 2;
                if (qwenCapturedVariants == requiredVariants) {
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

            {
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

        const bool intelQwen2 = arch == "qwen3_5_text" && numLayers == 24 &&
            gpu->adapterName.find("Intel") != std::string::npos;
        int64_t position = (int64_t)(pos - 1);
        GpuTensor positionT;
        if (!intelQwen2) {
            positionT.shape = {1, 1};
            positionT.dtype = TensorDtype::Int64;
            positionT.buffer = positionBuf;
            gpu->writeBuffer(positionBuf, &position, sizeof(position));
            positionT.cpuData.resize(sizeof(position));
            memcpy(positionT.cpuData.data(), &position, sizeof(position));
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

        executor.Execute(execCtx, inputs, outputs);

        int64_t logitNel = logitsOut.ElementCount();
        std::vector<float> logits(logitNel);
        auto rb = gpu->mapReadbackBuffer(logitNel * 4);
        memcpy(logits.data(), rb.data(), logitNel * 4);

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

    std::string arch, gpuName, backendName;
    uint32_t nLayer=0, nHead=0, nKvHeads=0, nEmbd=0, headDim=0, nVocab=0;

    bool Load(GPUContext& gpuCtx, const std::string& path) {
        gpu = &gpuCtx;
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
        return true;
    }

    std::vector<int32_t> Tokenize(const std::string& text) {
        if (format == "onnx") return onnxTokenizer.encode(text);
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

    int32_t Prefill(const int32_t* tokens, uint32_t n) {
        if (std::getenv("BP_DUMP_TOKENS")) {
            fprintf(stderr, "[debug] prompt tokens:");
            for (uint32_t i = 0; i < n; i++) fprintf(stderr, " %d", tokens[i]);
            fprintf(stderr, "\n");
        }
        int32_t next;
        bool pooledPrefill = runner.pleGpuPreprocess || runner.cfg.arch == "qwen35";
        bool qwenBatched = runner.cfg.arch == "qwen35" && n > 16 &&
                           !std::getenv("BP_QWEN35_SERIAL_PREFILL");
        if (qwenBatched && !std::getenv("BP_SYNC_PREFILL")) {
            next = runner.prefillBatched(tokens, n, 0);
        } else if (pooledPrefill && !std::getenv("BP_SYNC_PREFILL")) {
            next = runner.prefillPooledKnown(tokens, n, 0);
        } else {
            std::vector<float> logits;
            for (uint32_t i = 0; i < n; i++) logits = runner.decode(tokens[i], i);
            DumpTopLogits("prefill", logits);
            next = ModelRunner::argmax(logits);
        }
        runner.seedDecodeTokenInputs(next);
        pos = n;
        return next;
    }

    int32_t DecodePipelined() {
        int depth = runner.decodePoolDepth;

        if (pipelineInFlight == 0) {
            for (int i = 0; i < depth; i++) {
                int slot = (pos + i) % depth;
                runner.submitDecode(pos + i, slot);
            }
            pipelineInFlight = depth;
            pipelineNextSubmitPos = pos + depth;
        }

        int readSlot = pos % depth;
        int32_t tok = runner.readArgmax(readSlot);
        pipelineInFlight--;

        runner.submitDecode(pipelineNextSubmitPos, readSlot);
        pipelineNextSubmitPos++;
        pipelineInFlight++;

        pos++;
        return tok;
    }

    std::vector<float> DecodeSynchronous(int32_t token) {
        auto logits = runner.decode(token, pos);
        DumpTopLogits("decode", logits);
        pos++;
        return logits;
    }

    int32_t DecodeArgmaxSynchronous(int32_t token) {
        if (std::getenv("BP_DUMP_TOP_LOGITS")) {
            auto logits = DecodeSynchronous(token);
            return ModelRunner::argmax(logits);
        }
        int32_t next = runner.decodeArgmax(token, pos);
        pos++;
        return next;
    }

    void Reset() { runner.resetKVCache(); pos = 0; pipelineInFlight = 0; pipelineNextSubmitPos = 0; }
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

        if (!session.impl_->std_->Load(*gpuCtx, modelPath)) {
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
        cfg.maxSeqLen = 0;  // determined by ModelRunner internally
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
    if (!impl_) return {};

    if (resetSession) Reset();
    auto tokens = Tokenize(prompt);
    if (tokens.empty()) return {};

    int32_t next = Prefill(tokens.data(), (uint32_t)tokens.size());
    int32_t eos = GetEosTokenId();

    bool useSampling = (sampling.temperature > 0.0f);
    std::mt19937 rng(sampling.seed ? sampling.seed : std::random_device{}());

    // Re-sample from prefill logits if sampling enabled
    if (useSampling) {
        auto logits = DecodeLogits();
        if (!logits.empty())
            next = sampleToken(logits.data(), (uint32_t)logits.size(),
                               sampling.temperature, sampling.topK, rng);
    }

    std::string result;
    for (int i = 0; i < maxTokens; i++) {
        if (next == eos) break;
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
        auto logits = gen->RunStep(impl_->lastToken);
        int32_t next = argmax(logits.data(), (int64_t)logits.size());
        impl_->lastToken = next;
        return next;
    }

    // Qwen3.5 uses the queued decode path after fixing slot-local dynamic
    // params and GPU embedding gather. Keep a synchronous fallback for
    // validation or bisecting correctness regressions.
    auto* std = impl_->std_.get();
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
        next = std->DecodePipelined();
    }
    impl_->lastToken = next;
    return next;
}

std::vector<float> LmSession::DecodeLogits() {
    if (!impl_) return {};

    if (impl_->backend == Impl::Backend::GenericOnnx) {
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
    int32_t baseToken = std->DecodePipelined();
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

// ═══════════════════════════════════════════════════════════════════════════
// Benchmarking + Profiling
// ═══════════════════════════════════════════════════════════════════════════

BenchmarkResult LmSession::Benchmark(int promptLen, int genTokens) {
    if (!impl_) return {};

    BenchmarkResult result;
    result.promptLen = promptLen;

    if (impl_->backend == Impl::Backend::GenericOnnx) {
        auto* gen = impl_->gen_.get();

        // Check capacity
        int totalNeeded = promptLen + 3 + genTokens;
        if (totalNeeded > (int)gen->maxSeqLen) return result;

        std::vector<int32_t> prefillTokens(promptLen, 1);
        // Match ORT GenAI's reused-generator benchmark semantics: compile the
        // shape-specific pipelines and materialize the tensor plan before the
        // timed sample. Otherwise the first prompt reports shader compilation
        // as inference time and understates steady-state prefill by several x.
        if (!gen->benchWarmupDone) {
            gen->ResetCaches();
            (void)gen->RunPrefillBatch(prefillTokens.data(), (uint32_t)promptLen);
            gen->ResetCaches();
            gen->benchWarmupDone = true;
        } else {
            gen->ResetCaches();
        }
        int32_t tok = 1;

        // Prefill
        auto pfStart = std::chrono::steady_clock::now();
        tok = gen->RunPrefillBatch(prefillTokens.data(), (uint32_t)promptLen);
        auto pfEnd = std::chrono::steady_clock::now();
        result.prefillMs = std::chrono::duration<double, std::milli>(pfEnd - pfStart).count();
        result.prefillTokPerSec = promptLen * 1000.0 / result.prefillMs;
        gen->prefillDone = true;

        // TTFT (first decode step)
        auto ttftStart = std::chrono::steady_clock::now();
        {
            auto logits = gen->RunStep(tok);
            tok = argmax(logits.data(), (int64_t)logits.size());
        }
        auto ttftEnd = std::chrono::steady_clock::now();
        result.ttftMs = std::chrono::duration<double, std::milli>(ttftEnd - ttftStart).count();

        // Warm up the ordinary decode tensor plan and both Qwen ping-pong
        // capture variants before timing steady-state decode.  Batched prefill
        // deliberately invalidates its prompt-shaped tensor plan, so a fixed
        // two steps left capture work in the timed region while serial prefill
        // happened to complete it during the prompt loop.
        for (int i = 0; i < 2; i++) {
            auto logits = gen->RunStep(tok);
            tok = argmax(logits.data(), (int64_t)logits.size());
        }
        for (int i = 0; gen->fastDecodeEnabled && !gen->fastDecodeCaptured && i < 4; i++) {
            auto logits = gen->RunStep(tok);
            tok = argmax(logits.data(), (int64_t)logits.size());
        }

        // Timed decode
        auto dcStart = std::chrono::steady_clock::now();
        for (int i = 0; i < genTokens; i++) {
            auto logits = gen->RunStep(tok);
            tok = argmax(logits.data(), (int64_t)logits.size());
        }
        auto dcEnd = std::chrono::steady_clock::now();
        result.decodeMs = std::chrono::duration<double, std::milli>(dcEnd - dcStart).count();
        result.decodeTokPerSec = genTokens * 1000.0 / result.decodeMs;

        return result;
    }

    // Standard path
    auto* st = impl_->std_.get();
    int DEPTH = st->runner.decodePoolDepth;

    if (!st->benchWarmupDone) {
        st->Reset();
        // K-quant serial prefill can crash with large HD on some models,
        // so use a minimal warmup (single token) when batched prefill is unavailable
        int warmupT = st->runner.hasBatchedPrefill() ? std::min(promptLen, 32) : 1;
        std::vector<int32_t> w(warmupT, 0);
        int32_t t = st->runner.prefillBatched(w.data(), (uint32_t)w.size(), 0);
        st->runner.seedDecodeTokenInputs(t);
        for (int i = 0; i < DEPTH; i++) {
            int slot = ((int)w.size() + i) % DEPTH;
            st->runner.submitDecode((uint32_t)(w.size() + i), slot);
        }
        for (int i = 0; i < DEPTH; i++)
            st->runner.readArgmax(((int)w.size() + i) % DEPTH);
        st->Reset();
        st->benchWarmupDone = true;
    }

    st->Reset();
    std::vector<int32_t> dummy(promptLen, 0);
    auto t0 = std::chrono::steady_clock::now();
    int32_t first = st->runner.prefillBatched(dummy.data(), (uint32_t)promptLen, 0);
    auto t1 = std::chrono::steady_clock::now();
    result.prefillMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.prefillTokPerSec = promptLen * 1000.0 / result.prefillMs;

    st->runner.seedDecodeTokenInputs(first);
    st->gpu->timing.wait_ns = 0;
    auto t2 = std::chrono::steady_clock::now();
    int sub = 0, comp = 0;
    for (int i = 0; i < std::min(DEPTH, genTokens); i++) {
        int slot = (promptLen + i) % DEPTH;
        st->runner.submitDecode((uint32_t)(promptLen + i), slot);
        sub++;
    }
    while (comp < sub) {
        st->runner.readArgmax((promptLen + comp) % DEPTH);
        comp++;
        if (sub < genTokens) {
            st->runner.submitDecode((uint32_t)(promptLen + sub), (promptLen + sub) % DEPTH);
            sub++;
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    result.decodeMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    result.decodeTokPerSec = comp * 1000.0 / result.decodeMs;
    result.fenceWaitMs = st->gpu->timing.wait_ns / 1e6;

    st->Reset();
    return result;
}

void LmSession::EnableProfiling() {
    if (!impl_) return;
    if (impl_->backend == Impl::Backend::GenericOnnx)
        impl_->gen_->execCtx.enableGpuProfiling();
    else
        impl_->std_->runner.enableProfiling();
}

void LmSession::PrintProfileReport(const std::string& htmlPath) {
    if (!impl_) return;
    if (impl_->backend == Impl::Backend::GenericOnnx) {
        auto* gen = impl_->gen_.get();
        if (std::getenv("BP_PROFILE_PREFILL")) {
            constexpr uint32_t kProfileTokens = 64;
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
