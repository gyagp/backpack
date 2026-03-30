/**
 * Backpack LLM — End-to-end LLM text generation.
 *
 * Uses bp::Device from the runtime for GPU init, then drives
 * ModelRunner / Tokenizer internals directly for LLM inference.
 *
 * Usage:
 *   backpack_llm --model path/to/model  --chat "What is 2+2?"
 *   backpack_llm --model path/to/model  --prompt "Hello"
 *   backpack_llm --model path/to/model  --benchmark
 */

#include "backpack.h"

// LLM internals (not part of the public API)
#include "gpu_context.h"
#include "model_runner.h"
#include "graph_executor.h"
#include "tokenizer.h"
#include "onnx_tokenizer.h"
#include "wgsl_shaders.h"
#include "json_parser.h"

// Shared app utilities
#include "../common/app_common.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Path resolution ─────────────────────────────────────────────────────────

/// Check if a directory is a standard ONNX model (transformer architecture).
/// Requires config.json + tokenizer + .onnx file.
static bool isStandardOnnxDir(const std::string& path) {
    if (!fs::is_directory(path)) return false;
    // Must have some config
    bool hasConfig = fs::exists(fs::path(path) / "config.json");
    if (!hasConfig) return false;
    // Must have tokenizer
    if (!fs::exists(fs::path(path) / "tokenizer.json")) return false;
    // Must have an .onnx model file
    for (auto& e : fs::directory_iterator(path))
        if (e.is_regular_file() && e.path().extension() == ".onnx") return true;
    return false;
}

/// Check if a model has non-standard architecture needing GraphExecutor.
/// (conv+MoE hybrid like LFM2 — detected by layer_types in config.json)
static bool isNonStandardArch(const std::string& path) {
    std::string cfgPath = (fs::path(path) / "config.json").string();
    if (!fs::exists(cfgPath)) return false;
    std::ifstream f(cfgPath);
    std::string s{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    // Quick check: if config has "layer_types" or "conv_L_cache", it's non-standard
    return s.find("\"layer_types\"") != std::string::npos ||
           s.find("\"conv_L_cache\"") != std::string::npos;
}

/// Find the best ONNX model file in a directory (prefer q4f16)
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

static std::string resolvePath(const std::string& path, std::string& format,
                               const std::string& formatOverride = "") {
    // Format override: force gguf or onnx
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
                format = isNonStandardArch(path) ? "onnx_generic" : "onnx";
                if (format == "onnx_generic") return findOnnxFile(path);
                return path;
            }
            std::string dir = fs::path(path).parent_path().string();
            format = isNonStandardArch(dir) ? "onnx_generic" : "onnx";
            return format == "onnx" ? dir : path;
        }
    }

    // Direct GGUF file
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf") {
        if (fs::exists(path)) { format = "gguf"; return path; }
    }
    // Direct .onnx file
    if (path.size() > 5 && path.substr(path.size() - 5) == ".onnx") {
        if (fs::exists(path)) {
            std::string dir = fs::path(path).parent_path().string();
            format = isNonStandardArch(dir) ? "onnx_generic" : "onnx";
            return format == "onnx" ? dir : path;
        }
    }
    // Directory
    if (fs::is_directory(path)) {
        if (isStandardOnnxDir(path)) {
            format = isNonStandardArch(path) ? "onnx_generic" : "onnx";
            if (format == "onnx_generic") {
                return findOnnxFile(path);
            }
            return path;
        }
        // Check subdirectories
        for (auto& e : fs::directory_iterator(path)) {
            if (!e.is_directory()) continue;
            if (isStandardOnnxDir(e.path().string())) {
                format = isNonStandardArch(e.path().string()) ? "onnx_generic" : "onnx";
                if (format == "onnx_generic") return findOnnxFile(e.path().string());
                return e.path().string();
            }
        }
        // GGUF search
        std::string best;
        for (auto& e : fs::recursive_directory_iterator(path))
            if (e.is_regular_file() && e.path().extension() == ".gguf")
                { best = e.path().string(); if (best.find("Q8_0") != std::string::npos) break; }
        if (!best.empty()) { format = "gguf"; return best; }
    }
    format = "gguf";
    return path;
}

// Chat template: use app::applyChatTemplate from app_common.h

// ─── LLM Context (all LLM logic lives here, in the app) ─────────────────────

/// Generic ONNX LLM context using GraphExecutor (for non-standard architectures like LFM2)
struct OnnxLlmContext {
    GPUContext* gpu = nullptr;
    GraphExecutor executor;
    OnnxTokenizer tokenizer;
    std::string modelPath;  // path to .onnx file
    std::string modelDir;   // parent directory

    // Model config (parsed from config.json)
    int64_t hiddenSize = 0, numLayers = 0, vocabSize = 0;
    int64_t numHeads = 0, numKvHeads = 0, headDim = 0;
    int64_t convLCache = 3;
    int64_t maxSeqLen = 0;  // static KV cache max sequence length (0 = auto)
    int64_t intermediateSize = 0;     // dense MLP hidden dim (layers 0,1)
    int64_t moeIntermediateSize = 0;  // MoE expert hidden dim
    int64_t numExperts = 0, numExpertsPerTok = 0;
    float normEps = 1e-5f;
    float ropeTheta = 1000000.0f;
    std::vector<std::string> layerTypes;
    std::string arch;

    // KV cache state
    uint32_t pos = 0;
    std::unordered_map<std::string, GpuTensor> convState;
    std::unordered_map<std::string, GpuTensor> kvState;  // static f32 KV buffers
    std::vector<int> convLayerIndices, attnLayerIndices;

    // Reusable buffers for fast decode
    GPUBuffer idsBuf, maskBuf, nlkBuf, logitsBuf;
    std::vector<GPUBuffer> convOutBufs;
    uint32_t maskBufCapacity = 0;

    // Fast decode state (capture + replay)
    bool fastDecodeEnabled = false;
    bool fastDecodeCaptured = false;
    bool prefillDone = false;  // Capture only after prefill completes
    bool nlkWritten = false;   // Tracks whether nlkBuf has been initialized
    // Pre-allocated f16 conv cast output buffers (reused during replay stage)
    std::vector<GPUBuffer> convCastF16Bufs;
    // Pre-built conv cast bind groups for replay stage (avoid per-step MakeBindGroup)
    std::vector<WGPUBindGroup> convCastBindGroups;
    const CompiledPipeline* convCastPipeline = nullptr;
    uint32_t convCastWorkgroups = 0;

    bool Load(GPUContext& gpuCtx, const std::string& onnxPath) {
        gpu = &gpuCtx;
        modelPath = onnxPath;
        modelDir = fs::path(onnxPath).parent_path().string();

        // Load ONNX model
        if (!executor.Load(gpuCtx, onnxPath)) return false;

        // Load tokenizer
        if (!tokenizer.load(modelDir)) return false;

        // Parse config.json for model metadata
        std::string cfgPath = (fs::path(modelDir) / "config.json").string();
        if (!fs::exists(cfgPath)) {
            fprintf(stderr, "OnnxLlm: missing config.json\n");
            return false;
        }
        std::ifstream cfgFile(cfgPath);
        std::string cfgStr{std::istreambuf_iterator<char>(cfgFile),
                           std::istreambuf_iterator<char>()};
        cfgFile.close();
        auto cfg = json_parse(cfgStr);

        hiddenSize = cfg.has("hidden_size") ? cfg["hidden_size"].as_int() : 2048;
        numLayers = cfg.has("num_hidden_layers") ? cfg["num_hidden_layers"].as_int() : 24;
        vocabSize = cfg.has("vocab_size") ? cfg["vocab_size"].as_int() : 65536;
        numKvHeads = cfg.has("num_key_value_heads") ? cfg["num_key_value_heads"].as_int() : 8;
        numHeads = cfg.has("num_attention_heads") ? cfg["num_attention_heads"].as_int() : 32;
        headDim = hiddenSize / numHeads;
        convLCache = cfg.has("conv_L_cache") ? cfg["conv_L_cache"].as_int() : 3;
        intermediateSize = cfg.has("intermediate_size") ? cfg["intermediate_size"].as_int() : 7168;
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
                if (lt == "conv") convLayerIndices.push_back((int)i);
                else if (lt == "full_attention") attnLayerIndices.push_back((int)i);
            }
        }

        // Compute maxSeqLen based on GPU memory if not set by caller (--max-seq-len)
        if (maxSeqLen <= 0) {
            // Each KV buffer: [1, numKvHeads, seqLen, headDim] in f32
            // Budget: use at most 25% of maxBufferSize for total KV cache, leaving
            // room for model weights and intermediate buffers.
            int64_t nAttn = std::max((int64_t)1, (int64_t)attnLayerIndices.size());
            uint64_t maxBuf = gpuCtx.adapterLimits.maxBufferSize;
            // Per-buffer limit: each KV buffer must fit in maxBufferSize
            int64_t perBufLimit = (int64_t)(maxBuf / (numKvHeads * headDim * 4));
            // Total budget limit: nAttn * 2 (key+value) buffers, 25% of maxBufferSize
            int64_t budgetBytes = (int64_t)(maxBuf / 4);
            int64_t perBufFromBudget = budgetBytes / (nAttn * 2 * numKvHeads * headDim * 4);
            int64_t computed = std::min(perBufLimit, perBufFromBudget);
            // Round down to power of 2 for cleanliness
            int64_t rounded = 1;
            while (rounded * 2 <= computed) rounded *= 2;
            maxSeqLen = std::min(modelMaxSeq, std::max(rounded, (int64_t)4096));
        } else {
            maxSeqLen = std::min(modelMaxSeq, maxSeqLen);
        }

        printf("Model: %s (%lldL, H=%lld, V=%lld, conv=%zu, attn=%zu)\n",
               arch.c_str(), (long long)numLayers, (long long)hiddenSize,
               (long long)vocabSize, convLayerIndices.size(), attnLayerIndices.size());

        // Init caches
        ResetCaches();

        // Warmup (skip for now — just init)
        printf("  Init complete.\n");

        return true;
    }

    void ResetCaches() {
        pos = 0;
        prefillDone = false;
        nlkWritten = false;
        convState.clear();
        kvState.clear();
        // Reset fast decode state (captured bind groups reference old buffers)
        if (fastDecodeCaptured) {
            executor.ReleaseCaptured();
            fastDecodeCaptured = false;
        }
        // Invalidate warm execute caches (tensor plan buffers reference old state)
        executor.InvalidateWarmCaches();
        // Conv caches: [1, hiddenSize, convLCache] fp16, zero-initialized
        for (size_t ci = 0; ci < convLayerIndices.size(); ci++) {
            int idx = convLayerIndices[ci];
            std::string name = "past_conv." + std::to_string(idx);
            GpuTensor t;
            t.shape = {1, hiddenSize, convLCache};
            t.dtype = TensorDtype::Float16;
            size_t bytes = (size_t)(hiddenSize * convLCache * 2);
            // For fast decode: use pre-allocated f16 buffers so the captured
            // bind groups reference stable handles across steps.
            if (fastDecodeEnabled && ci < convCastF16Bufs.size()) {
                t.buffer = convCastF16Bufs[ci];
                // Zero-init the pre-allocated buffer
                std::vector<uint16_t> zeros((size_t)(hiddenSize * convLCache), 0);
                gpu->writeBuffer(t.buffer, zeros.data(), bytes);
            } else {
                t.buffer = gpu->createBuffer(name, bytes);
                std::vector<uint16_t> zeros((size_t)(hiddenSize * convLCache), 0);
                gpu->writeBuffer(t.buffer, zeros.data(), bytes);
            }
            convState[name] = t;
        }
        // Static KV caches: [1, numKvHeads, maxSeqLen, headDim] f32, pre-allocated
        size_t kvBytes = (size_t)(numKvHeads * maxSeqLen * headDim * 4);
        for (int idx : attnLayerIndices) {
            for (const char* suffix : {".key", ".value"}) {
                std::string name = "past_key_values." + std::to_string(idx) + suffix;
                GpuTensor t;
                t.shape = {1, numKvHeads, 0, headDim};  // seq_len=0 initially
                t.dtype = TensorDtype::Float32;          // f32 for static mode
                t.buffer = gpu->createBuffer(name, kvBytes);
                kvState[name] = t;
            }
        }
        // Pre-allocate reusable decode buffers
        idsBuf = gpu->createBuffer("ids", 8);
        nlkBuf = gpu->createBuffer("nlk", 8);
        logitsBuf = gpu->createBuffer("logits_out", vocabSize * 4);
        maskBuf = gpu->createBuffer("mask", maxSeqLen * 8);
        maskBufCapacity = (uint32_t)maxSeqLen;
        convOutBufs.resize(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "conv_out_" + std::to_string(convLayerIndices[i]);
            convOutBufs[i] = gpu->createBuffer(name, hiddenSize * convLCache * 4);
        }
    }

    // Run a single prefill step (always uses normal Execute, never captures).
    std::vector<float> RunPrefillStep(int64_t tokenId) {
        pos++;
        return runExecutePath(tokenId);
    }

    // Batched prefill: process T tokens in a single Execute call.
    // Returns argmax of the last token's logits.
    int32_t RunPrefillBatch(const int32_t* tokenIds, uint32_t T) {
        if (T == 0) return -1;
        if (T == 1) {
            auto logits = RunPrefillStep(tokenIds[0]);
            return Argmax(logits);
        }

        pos += T;

        std::unordered_map<std::string, GpuTensor*> inputs;

        // input_ids: [1, T] as int64
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

        // attention_mask: [1, T] — all ones (no past context for initial prefill)
        std::vector<int64_t> mask(pos, 1);
        GpuTensor maskT;
        maskT.shape = {1, (int64_t)pos};
        maskT.dtype = TensorDtype::Int64;
        maskT.buffer = maskBuf;
        gpu->writeBuffer(maskBuf, mask.data(), pos * 8);
        maskT.cpuData.resize(pos * 8);
        memcpy(maskT.cpuData.data(), mask.data(), pos * 8);
        inputs["attention_mask"] = &maskT;

        // num_logits_to_keep = 1 (only last token logits)
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

        // Caches — pass static KV buffers with 0 past sequence length
        for (auto& [name, t] : convState) inputs[name] = &t;
        for (auto& [name, t] : kvState) {
            t.shape = {1, numKvHeads, (int64_t)(pos - T), headDim};
            inputs[name] = &t;
        }

        // Outputs
        std::unordered_map<std::string, GpuTensor*> outputs;

        GpuTensor logitsOut;
        logitsOut.shape = {1, 1, vocabSize};
        logitsOut.dtype = TensorDtype::Float32;
        logitsOut.buffer = logitsBuf;
        outputs["logits"] = &logitsOut;

        std::vector<GpuTensor> convOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "present_conv." + std::to_string(convLayerIndices[i]);
            convOuts[i].shape = {1, hiddenSize, convLCache};
            convOuts[i].dtype = TensorDtype::Float32;
            convOuts[i].buffer = convOutBufs[i];
            outputs[name] = &convOuts[i];
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

        // Execute the full graph with T-wide inputs
        executor.Execute(inputs, outputs);

        // Map readback buffer
        int64_t logitNel = logitsOut.ElementCount();
        std::vector<float> logits(logitNel);
        auto rb = gpu->mapReadbackBuffer(logitNel * 4);
        memcpy(logits.data(), rb.data(), logitNel * 4);

        // Update conv state (f32→f16 cast for conv layers)
        auto gpuCastF32ToF16 = [&](GpuTensor& src, const std::string& name, size_t idx) {
            if (src.dtype != TensorDtype::Float32) return;
            int64_t nel = src.ElementCount();
            GpuTensor dst;
            dst.shape = src.shape;
            dst.dtype = TensorDtype::Float16;
            if (fastDecodeEnabled && idx < convCastF16Bufs.size()) {
                dst.buffer = convCastF16Bufs[idx];
            } else {
                dst.buffer = gpu->createBuffer(name, nel * 2);
                if (fastDecodeEnabled && convCastF16Bufs.size() <= idx) {
                    convCastF16Bufs.resize(idx + 1);
                    convCastF16Bufs[idx] = dst.buffer;
                }
            }
            uint32_t params[4] = {(uint32_t)nel, 0, 0, 0};
            auto paramBuf = executor.getParamBuffer(16);
            gpu->writeBuffer(paramBuf, params, 16);
            auto& cPl = executor.GetPipelineT("cast_f32_to_f16", 3,
                []() { return std::string(WGSL_CAST_F32_TO_F16); });
            auto bg = executor.MakeBindGroup(cPl, {
                {0, src.buffer}, {1, dst.buffer}, {2, paramBuf}});
            executor.QueueDispatch(cPl.pipeline, bg,
                (uint32_t)((nel + 255) / 256), 1, 1, "cache_cast_f16");
            src = dst;
        };

        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string inName = "past_conv." + std::to_string(convLayerIndices[i]);
            if (!convOuts[i].IsValid()) continue;
            gpuCastF32ToF16(convOuts[i], inName, i);
            convState[inName] = convOuts[i];
        }

        for (int idx : attnLayerIndices) {
            std::string kIn = "past_key_values." + std::to_string(idx) + ".key";
            std::string vIn = "past_key_values." + std::to_string(idx) + ".value";
            kvState[kIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvState[vIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
        }

        executor.SubmitPending();

        gpu->releaseBuffer(idT.buffer);

        return Argmax(logits);
    }

    // Run a single decode step and return logits.
    // Fast decode uses capture + replay pattern:
    //   warmup steps (normal) → capture step (save+execute) → replay steps
    std::vector<float> RunStep(int64_t tokenId) {
        pos++;

        // ── Fast Decode Replay Path ──────────────────────────────────────
        if (fastDecodeEnabled && fastDecodeCaptured) {
            // Set position for GQA param updates
            executor.replayPosition_ = (uint32_t)(pos - 1);

            // Set token ID for embedding re-upload
            executor.replayTokenId_ = tokenId;

            // Phase 1: replay param writes (GQA position params + token ID)
            executor.ReplayWrites();

            // Phase 2: encode+submit captured dispatches + readback
            uint64_t logitBytes = (uint64_t)vocabSize * 4;
            auto rbHandle = gpu->getOrCreateReadbackBuf(logitBytes);
            executor.RequestReadback(logitsBuf, {rbHandle, logitBytes}, logitBytes);
            executor.ReplayDispatches();

            // Map readback buffer
            std::vector<float> logits(vocabSize);
            auto rb = gpu->mapReadbackBuffer(vocabSize * 4);
            memcpy(logits.data(), rb.data(), vocabSize * 4);

            // Conv cast with pre-built bind groups
            for (size_t i = 0; i < convLayerIndices.size(); i++) {
                std::string inName = "past_conv." + std::to_string(convLayerIndices[i]);
                // AddRef because flushToEncoder releases bind groups after submit
                wgpuBindGroupAddRef(convCastBindGroups[i]);
                executor.QueueDispatch(convCastPipeline->pipeline, convCastBindGroups[i],
                    convCastWorkgroups, 1, 1, "cache_cast_f16");
                convState[inName].buffer = convCastF16Bufs[i];
                convState[inName].dtype = TensorDtype::Float16;
            }
            executor.SubmitPending();

            // Update KV shapes
            for (int idx : attnLayerIndices) {
                kvState["past_key_values." + std::to_string(idx) + ".key"].shape
                    = {1, numKvHeads, (int64_t)pos, headDim};
                kvState["past_key_values." + std::to_string(idx) + ".value"].shape
                    = {1, numKvHeads, (int64_t)pos, headDim};
            }
            return logits;
        }

        // ── Capture Stage (save AND execute) ────────────────────────
        // Bind groups are AddRef'd for replay. The capture stage produces
        // valid output since GPU work IS submitted.
        if (fastDecodeEnabled && !fastDecodeCaptured && prefillDone) {
            executor.capturePosition_ = (uint32_t)(pos - 1);
            executor.CaptureBegin();
            auto logits = runExecutePath(tokenId);
            executor.CaptureEnd();
            fastDecodeCaptured = true;

            // Remove conv cast flush if present
            if (!executor.capturedFlushes_.empty()) {
                auto& lastF = executor.capturedFlushes_.back();
                if (!lastF.dispatches.empty() &&
                    lastF.dispatches[0].name.find("cache_cast") != std::string::npos) {
                    executor.capturedFlushes_.pop_back();
                }
            }

            {
                int nDisp = 0;
                for (auto& f : executor.capturedFlushes_) nDisp += (int)f.dispatches.size();
                fprintf(stderr, "  [fast decode capture] %zu flushes, %d dispatches, %zu param updates\n",
                        executor.capturedFlushes_.size(), nDisp, executor.replayParamUpdates_.size());
            }

            // Pre-build conv cast bind groups for replay stage
            if (!convLayerIndices.empty()) {
                int64_t nel = hiddenSize * convLCache;
                convCastWorkgroups = (uint32_t)((nel + 255) / 256);
                uint32_t params[4] = {(uint32_t)nel, 0, 0, 0};
                auto paramBuf = gpu->createBuffer("conv_cast_params", 16);
                gpu->writeBuffer(paramBuf, params, 16);
                convCastPipeline = &executor.GetPipelineT("cast_f32_to_f16", 3,
                    []() { return std::string(WGSL_CAST_F32_TO_F16); });
                convCastBindGroups.resize(convLayerIndices.size());
                for (size_t i = 0; i < convLayerIndices.size(); i++) {
                    convCastBindGroups[i] = executor.MakeBindGroup(*convCastPipeline, {
                        {0, {convOutBufs[i].handle, convOutBufs[i].size}},
                        {1, convCastF16Bufs[i]},
                        {2, paramBuf}});
                }
            }

            // Logits are valid — capture stage executes GPU work
            return logits;
        }

        // ── Normal Path (fast decode disabled) ────────────────────────
        return runExecutePath(tokenId);
    }

private:
    // Shared implementation for the full Execute path (used by normal and capture stages)
    std::vector<float> runExecutePath(int64_t tokenId) {

        std::unordered_map<std::string, GpuTensor*> inputs;

        // input_ids: [1, 1] — reuse buffer
        GpuTensor idT;
        idT.shape = {1, 1};
        idT.dtype = TensorDtype::Int64;
        idT.buffer = idsBuf;
        gpu->writeBuffer(idsBuf, &tokenId, 8);
        idT.cpuData.resize(8);
        memcpy(idT.cpuData.data(), &tokenId, 8);
        inputs["input_ids"] = &idT;

        // attention_mask: [1, pos] — reuse buffer (grows, but pre-allocated to maxSeqLen)
        std::vector<int64_t> mask(pos, 1);
        GpuTensor maskT;
        maskT.shape = {1, (int64_t)pos};
        maskT.dtype = TensorDtype::Int64;
        maskT.buffer = maskBuf;
        gpu->writeBuffer(maskBuf, mask.data(), pos * 8);
        maskT.cpuData.resize(pos * 8);
        memcpy(maskT.cpuData.data(), mask.data(), pos * 8);
        inputs["attention_mask"] = &maskT;

        // num_logits_to_keep — reuse buffer, write once per cache reset
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

        // Caches — pass static KV buffers with current seq length
        for (auto& [name, t] : convState) inputs[name] = &t;
        for (auto& [name, t] : kvState) {
            // Update shape to reflect current sequence length
            t.shape = {1, numKvHeads, (int64_t)(pos - 1), headDim};
            inputs[name] = &t;
        }

        // Outputs
        std::unordered_map<std::string, GpuTensor*> outputs;

        GpuTensor logitsOut;
        logitsOut.shape = {1, 1, vocabSize};
        logitsOut.dtype = TensorDtype::Float32;
        logitsOut.buffer = logitsBuf;
        outputs["logits"] = &logitsOut;

        std::vector<GpuTensor> convOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "present_conv." + std::to_string(convLayerIndices[i]);
            convOuts[i].shape = {1, hiddenSize, convLCache};
            convOuts[i].dtype = TensorDtype::Float32;
            convOuts[i].buffer = convOutBufs[i];
            outputs[name] = &convOuts[i];
        }

        // KV outputs: point to the same static buffers (attention op detects static mode)
        std::vector<GpuTensor> kvOuts(attnLayerIndices.size() * 2);
        for (size_t i = 0; i < attnLayerIndices.size(); i++) {
            std::string kName = "present." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vName = "present." + std::to_string(attnLayerIndices[i]) + ".value";
            std::string kIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".value";
            // Output uses same static buffer — op will write in-place
            kvOuts[i*2].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2].dtype = TensorDtype::Float32;
            kvOuts[i*2].buffer = kvState[kIn].buffer;
            kvOuts[i*2+1].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2+1].dtype = TensorDtype::Float32;
            kvOuts[i*2+1].buffer = kvState[vIn].buffer;
            outputs[kName] = &kvOuts[i*2];
            outputs[vName] = &kvOuts[i*2+1];
        }

        // Request logits readback to be folded into Execute's final CB.
        uint64_t logitBytes = (uint64_t)vocabSize * 4;
        auto rbHandle = gpu->getOrCreateReadbackBuf(logitBytes);
        executor.RequestReadback(logitsBuf, {rbHandle, logitBytes}, logitBytes);

        executor.Execute(inputs, outputs);

        // Map readback buffer — the copy was already in the CB, no second submit
        int64_t logitNel = logitsOut.ElementCount();
        std::vector<float> logits(logitNel);
        auto rb = gpu->mapReadbackBuffer(logitNel * 4);
        memcpy(logits.data(), rb.data(), logitNel * 4);

        // Update conv state — cast f32→f16 (conv layers expect fp16 input)
        auto gpuCastF32ToF16 = [&](GpuTensor& src, const std::string& name, size_t idx) {
            if (src.dtype != TensorDtype::Float32) return;
            int64_t nel = src.ElementCount();
            GpuTensor dst;
            dst.shape = src.shape;
            dst.dtype = TensorDtype::Float16;
            // Use pre-allocated f16 buffer for fast decode
            if (fastDecodeEnabled && idx < convCastF16Bufs.size()) {
                dst.buffer = convCastF16Bufs[idx];
            } else {
                dst.buffer = gpu->createBuffer(name, nel * 2);
                // Save for future replay
                if (fastDecodeEnabled && convCastF16Bufs.size() <= idx) {
                    convCastF16Bufs.resize(idx + 1);
                    convCastF16Bufs[idx] = dst.buffer;
                }
            }
            uint32_t params[4] = {(uint32_t)nel, 0, 0, 0};
            auto paramBuf = executor.getParamBuffer(16);
            gpu->writeBuffer(paramBuf, params, 16);
            auto& cPl = executor.GetPipelineT("cast_f32_to_f16", 3,
                []() { return std::string(WGSL_CAST_F32_TO_F16); });
            auto bg = executor.MakeBindGroup(cPl, {
                {0, src.buffer}, {1, dst.buffer}, {2, paramBuf}});
            executor.QueueDispatch(cPl.pipeline, bg,
                (uint32_t)((nel + 255) / 256), 1, 1, "cache_cast_f16");
            src = dst;
        };

        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string inName = "past_conv." + std::to_string(convLayerIndices[i]);
            if (!convOuts[i].IsValid()) continue;
            gpuCastF32ToF16(convOuts[i], inName, i);
            convState[inName] = convOuts[i];
        }

        // KV cache: static buffers already updated in-place by the attention op.
        for (int idx : attnLayerIndices) {
            std::string kIn = "past_key_values." + std::to_string(idx) + ".key";
            std::string vIn = "past_key_values." + std::to_string(idx) + ".value";
            kvState[kIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvState[vIn].shape = {1, numKvHeads, (int64_t)pos, headDim};
        }

        // Submit conv cast dispatches async
        executor.SubmitPending();

        return logits;
    }

public:
    int32_t Argmax(const std::vector<float>& logits) {
        float mx = -1e30f;
        int32_t idx = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            if (logits[i] > mx) { mx = logits[i]; idx = (int32_t)i; }
        }
        return idx;
    }

    /// Check if the model supports fast decode.
    /// Returns empty string if supported, or a reason string if not.
    std::string CheckFastDecodeSupport() const {
        auto& graph = executor.GetGraph();

        // Check for dynamic control flow ops
        for (auto& node : graph.nodes) {
            if (node.opType == "If" || node.opType == "Loop" || node.opType == "Scan")
                return "model contains dynamic control flow (" + node.opType + " op: " + node.name + ")";
        }

        // Must have conv or attention layers (otherwise it's a simple model
        // that doesn't benefit from fast decode)
        if (convLayerIndices.empty() && attnLayerIndices.empty())
            return "model has no conv or attention layers";

        return "";
    }

    /// Enable fast decode and pre-allocate required buffers.
    void EnableFastDecode() {
        fastDecodeEnabled = true;
        convCastF16Bufs.resize(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            size_t nel = (size_t)(hiddenSize * convLCache);
            convCastF16Bufs[i] = gpu->createBuffer(
                "conv_cast_f16_" + std::to_string(i), nel * 2);
        }
    }

    /// Pre-compile GPU pipelines in parallel to reduce time-to-first-token.
    /// Walks the ONNX graph to identify needed kernels and compiles them ahead of time.
    void WarmupPipelines() {
        auto t0 = std::chrono::steady_clock::now();
        auto& kernels = getEmbeddedKernels();

        // Collect kernel specs needed by this model's ops
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

        // Walk graph ops and map to kernel names
        for (auto& node : executor.GetGraph().nodes) {
            if (node.opType == "MatMul" || node.opType == "Gemm") {
                addKernel("gemm");
                addKernel("fp16_gemm");
            } else if (node.opType == "Conv") {
                addKernel("conv2d");
            } else if (node.opType == "ConvTranspose") {
                addKernel("conv_transpose2d");
            } else if (node.opType == "SimplifiedLayerNormalization" ||
                       node.opType == "SkipSimplifiedLayerNormalization") {
                addKernel("rms_norm");
                addKernel("rms_norm_batched");
                addKernel("add_rms_norm");
                addKernel("add_rms_norm_batched");
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

        // Add attention + rope kernels if model has attention layers
        if (!attnLayerIndices.empty()) {
            addKernel("gqa_fused_attn");
            addKernel("gqa_prefill");
            addKernel("rotary_embedding");
            addKernel("fused_qknorm_rope");
            addKernel("fused_qknorm_rope_batched");
            addKernel("rope_batched_simple");
        }

        // Add MoE-related kernels if model has MoE
        if (numExperts > 0) {
            addKernel("silu_mul_fused");
        }

        // Add argmax + embed gather
        addKernel("argmax");
        addKernel("embed_gather");

        if (!specs.empty()) {
            int compiled = gpu->warmupPipelines(specs);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("  Warmed %d/%zu GPU pipelines in %.0fms\n",
                   compiled, specs.size(), ms);
        }
    }

    // ─── Batched Prefill ──────────────────────────────────────────────
    // Processes all T prompt tokens in a single Execute() call with
    // T-wide tensor shapes. Most ops (norms, matmuls, conv) natively
    // handle T>1. MoE and attention use per-token loops internally.

    /// Prefill all T tokens at once. Returns argmax of last token's logits.
    int32_t prefillBatched(const int32_t* tokenIds, uint32_t T) {
        return RunPrefillBatch(tokenIds, T);
    }
};

/// Standard transformer LLM context using ModelRunner
struct LlmContext {
    GPUContext* gpu = nullptr;
    ModelRunner runner;
    Tokenizer ggufTokenizer;
    OnnxTokenizer onnxTokenizer;
    std::string format;
    uint32_t pos = 0;
    bool benchWarmupDone = false;

    // Model info
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
            if (!onnxTokenizer.load(resolved)) return false;
        } else {
            if (!ggufTokenizer.load(runner.gguf)) return false;
        }

        // Warmup + autotune
        runner.decode(0, 0);
        runner.resetKVCache();
        if (!runner.loadDecodeAutotuneCache()) {
            runner.autotuneDecodeDepth();
            runner.autotuneDecodeKernels();
            runner.saveDecodeAutotuneCache();
        }

        auto& c = runner.cfg;
        arch = c.arch; nLayer = c.nLayer; nHead = c.nHead;
        nKvHeads = c.nKvHeads; nEmbd = c.nEmbd; headDim = c.headDim;
        nVocab = c.nVocab;
        gpuName = gpuCtx.adapterName;
        return true;
    }

    // Tokenizer
    std::vector<int32_t> Tokenize(const std::string& text) {
        return (format == "onnx") ? onnxTokenizer.encode(text) : ggufTokenizer.encode(text);
    }
    std::string Detokenize(int32_t tok) {
        return (format == "onnx") ? onnxTokenizer.decode_token(tok) : ggufTokenizer.decode_token(tok);
    }
    int32_t Eos() {
        return (format == "onnx") ? onnxTokenizer.eos_token_id : ggufTokenizer.eos_token_id;
    }

    // Prefill (sequential)
    int32_t Prefill(const int32_t* tokens, uint32_t n) {
        std::vector<float> logits;
        for (uint32_t i = 0; i < n; i++) logits = runner.decode(tokens[i], i);
        int32_t next = ModelRunner::argmax(logits);
        gpu->writeBuffer(runner.argmaxResultBuf, &next, 4);
        pos = n;
        return next;
    }

    // Decode (pipelined)
    int32_t Decode() {
        int slot = pos % runner.decodePoolDepth;
        runner.submitDecode(pos, slot);
        int32_t tok = runner.readArgmax(slot);
        pos++;
        return tok;
    }

    void Reset() { runner.resetKVCache(); pos = 0; }

    // Generate with streaming
    using StreamCB = std::function<bool(const std::string&)>;
    std::string Generate(const std::string& prompt, int maxTokens, StreamCB onToken) {
        Reset();
        auto tokens = Tokenize(prompt);
        if (tokens.empty()) return "";
        int32_t next = Prefill(tokens.data(), (uint32_t)tokens.size());
        int32_t eos = Eos();
        std::string result;
        for (int i = 0; i < maxTokens; i++) {
            if (next == eos) break;
            std::string text = Detokenize(next);
            if (!(text.size() >= 2 && text[0] == '<' && text.back() == '>')) {
                result += text;
                if (onToken && !onToken(text)) break;
            }
            next = Decode();
        }
        return result;
    }

    // Generate with temperature/top-k sampling (synchronous decode for logits access)
    std::string GenerateSampled(const std::string& prompt, int maxTokens,
                                const app::SamplingParams& sp, std::mt19937& rng,
                                StreamCB onToken) {
        Reset();
        auto tokens = Tokenize(prompt);
        if (tokens.empty()) return "";
        int32_t next = Prefill(tokens.data(), (uint32_t)tokens.size());
        // Re-sample from prefill logits if sampling enabled
        if (sp.temperature > 0.0f) {
            auto logits = runner.decode(next, pos);
            next = app::sampleToken(logits.data(), (uint32_t)logits.size(), sp, rng);
            pos++;
        }
        int32_t eos = Eos();
        std::string result;
        for (int i = 0; i < maxTokens; i++) {
            if (next == eos) break;
            std::string text = Detokenize(next);
            if (!(text.size() >= 2 && text[0] == '<' && text.back() == '>')) {
                result += text;
                if (onToken && !onToken(text)) break;
            }
            auto logits = runner.decode(next, pos);
            next = app::sampleToken(logits.data(), (uint32_t)logits.size(), sp, rng);
            pos++;
        }
        return result;
    }

    // Benchmark
    struct BenchResult { double pfMs, pfTps, dcMs, dcTps; int nPf, nDc; };
    BenchResult Benchmark(int promptLen, int genTokens) {
        BenchResult r{};
        int DEPTH = runner.decodePoolDepth;
        if (!benchWarmupDone) {
            Reset();
            std::vector<int32_t> w(std::min(promptLen, 32), 0);
            int32_t t = runner.prefillBatched(w.data(), (uint32_t)w.size(), 0);
            gpu->writeBuffer(runner.argmaxResultBuf, &t, 4);
            int n = std::min(DEPTH, 4);
            for (int i = 0; i < n; i++) runner.submitDecode((uint32_t)(w.size()+i), i);
            for (int i = 0; i < n; i++) runner.readArgmax(i);
            Reset();
            benchWarmupDone = true;
        }
        Reset();
        std::vector<int32_t> dummy(promptLen, 0);
        auto t0 = std::chrono::steady_clock::now();
        int32_t first = runner.prefillBatched(dummy.data(), (uint32_t)promptLen, 0);
        auto t1 = std::chrono::steady_clock::now();
        r.pfMs = std::chrono::duration<double,std::milli>(t1-t0).count();
        r.pfTps = promptLen * 1000.0 / r.pfMs;
        r.nPf = promptLen;
        gpu->writeBuffer(runner.argmaxResultBuf, &first, 4);
        auto t2 = std::chrono::steady_clock::now();
        int sub=0, comp=0;
        for (int i = 0; i < std::min(DEPTH, genTokens); i++) { runner.submitDecode((uint32_t)(promptLen+i), i); sub++; }
        while (comp < sub) { runner.readArgmax(comp%DEPTH); comp++;
            if (sub < genTokens) { runner.submitDecode((uint32_t)(promptLen+sub), comp%DEPTH); sub++; } }
        auto t3 = std::chrono::steady_clock::now();
        r.dcMs = std::chrono::duration<double,std::milli>(t3-t2).count();
        r.dcTps = comp * 1000.0 / r.dcMs;
        r.nDc = comp;
        Reset();
        return r;
    }
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string modelPath, prompt, chatMessage, backendStr;
    std::string formatOverride;  // --format gguf|onnx
    int maxTokens = 100;
    bool benchmark = false, profile = false, noFastDecode = false;
    bool saveBaseline = false;
    std::string baselinePath;
    int benchPromptLen = 0, benchGenTokens = 128;
    int maxSeqLenOverride = 0;  // --max-seq-len, 0 = auto
    float temperature = 0.0f;
    int topK = 0;
    uint64_t samplerSeed = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc)           modelPath = argv[++i];
        else if (arg == "--prompt" && i+1 < argc)     prompt = argv[++i];
        else if (arg == "--chat" && i+1 < argc)       chatMessage = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) maxTokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc)    backendStr = argv[++i];
        else if (arg == "--format" && i+1 < argc)     formatOverride = argv[++i];
        else if (arg == "--benchmark")                benchmark = true;
        else if (arg == "--profile")                  profile = true;
        else if (arg == "--no-fast-decode")            noFastDecode = true;
        else if (arg == "--fast-decode")              {} // accepted for compat, now default
        else if (arg == "--bench-prompt-len" && i+1 < argc) benchPromptLen = atoi(argv[++i]);
        else if (arg == "--bench-gen-tokens" && i+1 < argc) benchGenTokens = atoi(argv[++i]);
        else if (arg == "--max-seq-len" && i+1 < argc)    maxSeqLenOverride = atoi(argv[++i]);
        else if (arg == "--temperature" && i+1 < argc)   temperature = (float)atof(argv[++i]);
        else if (arg == "--top-k" && i+1 < argc)         topK = atoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc)           samplerSeed = (uint64_t)atoll(argv[++i]);
        else if (arg == "--save-baseline") {
            saveBaseline = true;
            if (i+1 < argc && argv[i+1][0] != '-') baselinePath = argv[++i];
        }
    }

    if (saveBaseline) benchmark = true;

    if (modelPath.empty()) {
        fprintf(stderr,
            "Backpack LLM — WebGPU text generation\n\n"
            "Usage: %s --model <path> [options]\n\n"
            "  --prompt <text>    Raw text prompt\n"
            "  --chat <message>   Chat message (auto-template)\n"
            "  --max-tokens <n>   Max tokens (default: 100)\n"
            "  --backend <name>   vulkan / d3d12 / metal\n"
            "  --format <fmt>     Force model format: gguf / onnx\n"
            "  --no-fast-decode   Disable fast decode\n"
            "  --temperature <f>  Sampling temperature (0 = greedy)\n"
            "  --top-k <n>       Top-k sampling (0 = disabled)\n"
            "  --seed <n>        Random seed for sampling\n"
            "  --benchmark        Prefill+decode sweep\n"
            "  --profile          GPU timestamp profiling + HTML timeline\n"
            "  --save-baseline [path]  Save benchmark results to JSON\n"
            "  --max-seq-len <n>  Max context length (default: auto from GPU memory)\n", argv[0]);
        return 1;
    }

    // Model auto-discovery (BACKPACK_MODELS env var support)
    modelPath = app::discoverModelPath(modelPath);

    if (prompt.empty() && chatMessage.empty() && !benchmark)
        prompt = "Hello";

    // 1. Create device via runtime API
    auto device = app::createDevice(backendStr);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }

    // 2. Load LLM
    std::string modelFormat;
    std::string resolvedPath = resolvePath(modelPath, modelFormat, formatOverride);
    printf("Loading model: %s (%s)\n", resolvedPath.c_str(), modelFormat.c_str()); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    app::LoadingInfo loadInfo;

    // Standard transformer models (GGUF or ONNX)
    LlmContext llm;
    // Generic ONNX models (LFM2, etc.)
    OnnxLlmContext onnxLlm;
    bool isGenericOnnx = (modelFormat == "onnx_generic");

    if (isGenericOnnx) {
        if (maxSeqLenOverride > 0) onnxLlm.maxSeqLen = maxSeqLenOverride;
        if (!onnxLlm.Load(*static_cast<GPUContext*>(device.GetGPUContext()), resolvedPath)) {
            fprintf(stderr, "Failed to load model\n"); return 1;
        }
        auto tWeightsDone = std::chrono::steady_clock::now();
        loadInfo.weightMs = std::chrono::duration<double, std::milli>(tWeightsDone - t0).count();

        // Auto-enable fast decode unless disabled
        if (!noFastDecode) {
            std::string reason = onnxLlm.CheckFastDecodeSupport();
            if (reason.empty()) {
                onnxLlm.EnableFastDecode();
                printf("  Fast decode: enabled\n");
            } else {
                fprintf(stderr, "  Fast decode: disabled — %s\n", reason.c_str());
            }
        }
        // Pre-compile GPU pipelines in parallel
        auto tShaderStart = std::chrono::steady_clock::now();
        onnxLlm.WarmupPipelines();
        auto tShaderDone = std::chrono::steady_clock::now();
        loadInfo.shaderMs = std::chrono::duration<double, std::milli>(tShaderDone - tShaderStart).count();
    } else {
        if (!llm.Load(*static_cast<GPUContext*>(device.GetGPUContext()), modelPath)) {
            fprintf(stderr, "Failed to load model\n"); return 1;
        }
        auto tWeightsDone = std::chrono::steady_clock::now();
        loadInfo.weightMs = std::chrono::duration<double, std::milli>(tWeightsDone - t0).count();
    }

    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    loadInfo.totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // (Micro-test removed — fast decode validated)

    if (isGenericOnnx) {
        printf("\nModel: %s (generic ONNX, %lldL, H=%lld, V=%lld, ctx=%lld)\n",
               onnxLlm.arch.c_str(), (long long)onnxLlm.numLayers,
               (long long)onnxLlm.hiddenSize, (long long)onnxLlm.vocabSize,
               (long long)onnxLlm.maxSeqLen);
    } else {
        printf("\nModel: %s (%s, %uL, E=%u, HD=%u, V=%u)\n",
               llm.arch.c_str(), llm.format.c_str(),
               llm.nLayer, llm.nEmbd, llm.headDim, llm.nVocab);
    }
    printf("GPU:   %s (%s)\n", device.GetName().c_str(), device.GetBackendName().c_str());
    printf("Ready: %lldms\n\n", (long long)loadMs);

    // Resolve baseline output path (default: apps/baseline/<model_name>.json)
    if (saveBaseline && baselinePath.empty()) {
        std::string arch = isGenericOnnx ? onnxLlm.arch : llm.arch;
        auto baselineDir = fs::path(__FILE__).parent_path().parent_path() / "baseline";
        fs::create_directories(baselineDir);
        baselinePath = (baselineDir / (arch + ".json")).string();
    }

    // 3. Benchmark
    if (benchmark) {
        auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
        std::vector<app::BenchResultEntry> benchResults;
        std::vector<int> lens;
        if (benchPromptLen > 0) lens.push_back(benchPromptLen);
        else lens = {128, 256, 512, 1024, 2048, 4096};

        if (isGenericOnnx) {
            printf("=== Benchmark: %s (generic ONNX) ===\n", onnxLlm.arch.c_str());
            printf("%-12s %10s %10s %10s %10s\n",
                   "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s");
            printf("%-12s %10s %10s %10s %10s\n",
                   "----------", "----------", "--------", "---------", "--------");

            int genTokens = benchGenTokens > 0 ? benchGenTokens : 128;

            for (int pl : lens) {
                // Check if total tokens fit in maxSeqLen
                int totalNeeded = pl + 3 + genTokens;  // prefill + warmup + decode
                if (totalNeeded > (int)onnxLlm.maxSeqLen) {
                    printf("%-12d   (skipped — %d tokens exceeds maxSeqLen %lld)\n",
                           pl, totalNeeded, (long long)onnxLlm.maxSeqLen);
                    continue;
                }

                onnxLlm.ResetCaches();
                int32_t tok = 1;  // BOS

                // Prefill: batched capture+replay
                std::vector<int32_t> prefillTokens(pl, 1);  // BOS fill
                auto pfStart = std::chrono::steady_clock::now();
                tok = onnxLlm.prefillBatched(prefillTokens.data(), (uint32_t)pl);
                auto pfEnd = std::chrono::steady_clock::now();
                double pfMs = std::chrono::duration<double, std::milli>(pfEnd - pfStart).count();
                onnxLlm.prefillDone = true;

                // TTFT: time the first decode step
                auto ttftStart = std::chrono::steady_clock::now();
                {
                    auto logits = onnxLlm.RunStep(tok);
                    tok = onnxLlm.Argmax(logits);
                }
                auto ttftEnd = std::chrono::steady_clock::now();
                double ttftMs = std::chrono::duration<double, std::milli>(ttftEnd - ttftStart).count();

                // Warmup 2 more decode steps (capture/replay stabilization)
                for (int i = 0; i < 2; i++) {
                    auto logits = onnxLlm.RunStep(tok);
                    tok = onnxLlm.Argmax(logits);
                }

                // Timed decode
                auto dcStart = std::chrono::steady_clock::now();
                for (int i = 0; i < genTokens; i++) {
                    auto logits = onnxLlm.RunStep(tok);
                    tok = onnxLlm.Argmax(logits);
                }
                auto dcEnd = std::chrono::steady_clock::now();
                double dcMs = std::chrono::duration<double, std::milli>(dcEnd - dcStart).count();

                double pfTps = pl * 1000.0 / pfMs;
                double dcTps = genTokens * 1000.0 / dcMs;

                printf("%-12d %10.1f %10.1f %10.1f %10.1f\n",
                       pl, pfMs, pfTps, dcMs, dcTps);
                fflush(stdout);

                benchResults.push_back({pl, pfMs, pfTps, dcMs, dcTps, ttftMs});
            }

            // CPU profile: single replay step (skip for now — hangs)
            if (false && !benchResults.empty()) {
                onnxLlm.ResetCaches();
                int32_t tok = 1;
                auto logits = onnxLlm.RunPrefillStep(tok);
                tok = onnxLlm.Argmax(logits);
                onnxLlm.prefillDone = true;
                for (int i = 0; i < 3; i++) {
                    logits = onnxLlm.RunStep(tok);
                    tok = onnxLlm.Argmax(logits);
                }
                printf("\n=== Profile (single replay step) ===\n");
                onnxLlm.executor.profilingEnabled = true;
                logits = onnxLlm.RunStep(tok);
                onnxLlm.executor.profilingEnabled = false;
            }

            // GPU hardware timestamp profiling (if --profile)
            if (profile) {
                printf("\n=== GPU Hardware Timestamp Profile (single replay step) ===\n");
                onnxLlm.ResetCaches();
                int32_t tok = 1;
                auto logits = onnxLlm.RunPrefillStep(tok);
                tok = onnxLlm.Argmax(logits);
                onnxLlm.prefillDone = true;
                for (int i = 0; i < 3; i++) {
                    auto lg = onnxLlm.RunStep(tok);
                    tok = onnxLlm.Argmax(lg);
                }
                onnxLlm.executor.enableGpuProfiling();
                auto pt0 = std::chrono::steady_clock::now();
                logits = onnxLlm.RunStep(tok);
                tok = onnxLlm.Argmax(logits);
                auto pt1 = std::chrono::steady_clock::now();
                double profMs = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
                std::string htmlPath = (fs::path(onnxLlm.modelDir) / "profile.html").string();
                onnxLlm.executor.printGpuProfileReport(1, profMs, htmlPath);
            }

            // Save baseline JSON
            if (saveBaseline) {
                auto sysInfo = app::getSystemInfo();
                auto memStats = gpuCtx->getMemoryStats();
                app::MemoryInfo memInfo{memStats.peakBytes, memStats.currentBytes, memStats.allocCount};
                app::writeBaselineJson(baselinePath, sysInfo,
                    device.GetName(), device.GetBackendName(),
                    gpuCtx->adapterDescription,
                    onnxLlm.arch, onnxLlm.modelPath, "onnx_generic",
                    (int)onnxLlm.numLayers, (int)onnxLlm.hiddenSize,
                    (int)onnxLlm.vocabSize,
                    benchGenTokens, benchResults,
                    &loadInfo, &memInfo);
            }

            printf("\n");
            return 0;
        }

        // Standard model benchmark
        printf("=== Benchmark: %s ===\n", llm.arch.c_str());
        printf("%-12s %10s %10s %10s %10s\n",
               "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s");
        printf("%-12s %10s %10s %10s %10s\n",
               "----------", "----------", "--------", "---------", "--------");
        for (int pl : lens) {
            auto r = llm.Benchmark(pl, benchGenTokens);
            printf("%-12d %10.1f %10.1f %10.1f %10.1f\n",
                   pl, r.pfMs, r.pfTps, r.dcMs, r.dcTps);
            fflush(stdout);
            benchResults.push_back({pl, r.pfMs, r.pfTps, r.dcMs, r.dcTps});
        }

        // Save baseline JSON
        if (saveBaseline) {
            auto sysInfo = app::getSystemInfo();
            auto memStats = gpuCtx->getMemoryStats();
            app::MemoryInfo memInfo{memStats.peakBytes, memStats.currentBytes, memStats.allocCount};
            std::string mDir = fs::path(modelPath).parent_path().string();
            app::writeBaselineJson(baselinePath, sysInfo,
                device.GetName(), device.GetBackendName(),
                gpuCtx->adapterDescription,
                llm.arch, modelPath, llm.format,
                (int)llm.nLayer, (int)llm.nEmbd, (int)llm.nVocab,
                benchGenTokens, benchResults,
                &loadInfo, &memInfo);
        }

        printf("\n");
        return 0;
    }

    // 4. Generate
    std::string finalPrompt;
    bool chat = !chatMessage.empty();
    if (chat) {
        std::string arch_str = isGenericOnnx ? onnxLlm.arch : llm.arch;
        finalPrompt = app::applyChatTemplate(chatMessage, arch_str);
        printf("Chat: %s\n", chatMessage.c_str());
    } else {
        finalPrompt = prompt;
    }

    if (isGenericOnnx) {
        auto promptTokens = onnxLlm.tokenizer.encode(finalPrompt);
        printf("Prompt: %zu tokens\n", promptTokens.size());
        if (temperature > 0)
            printf("Sampling: temperature=%.2f, top_k=%d, seed=%llu\n",
                   temperature, topK, (unsigned long long)samplerSeed);
        printf("\n--- Output ---\n");
        if (!chat) printf("%s", finalPrompt.c_str());
        fflush(stdout);

        // Set up sampler
        app::SamplingParams sp{temperature, topK, samplerSeed};
        std::mt19937 rng(samplerSeed ? samplerSeed : std::random_device{}());

        auto genStart = std::chrono::steady_clock::now();
        int tokenCount = 0;
        int32_t eos = onnxLlm.tokenizer.eos_token_id;

        onnxLlm.ResetCaches();
        // Prefill: batched capture+replay
        int32_t next = onnxLlm.prefillBatched(promptTokens.data(), (uint32_t)promptTokens.size());
        onnxLlm.prefillDone = true;

        // When sampling, re-sample the last prefill logits too
        if (temperature > 0.0f) {
            auto logits = onnxLlm.RunStep(next);
            next = app::sampleToken(logits.data(), (uint32_t)onnxLlm.vocabSize, sp, rng);
        }

        // Decode loop
        for (int i = 0; i < maxTokens; i++) {
            if (next == eos) break;
            std::string text = onnxLlm.tokenizer.decode_token(next);
            if (!(text.size() >= 2 && text[0] == '<' && text.back() == '>')) {
                printf("%s", text.c_str()); fflush(stdout);
                tokenCount++;
            }
            auto logits = onnxLlm.RunStep(next);
            if (temperature > 0.0f) {
                next = app::sampleToken(logits.data(), (uint32_t)onnxLlm.vocabSize, sp, rng);
            } else {
                next = onnxLlm.Argmax(logits);
            }
        }

        auto genMs = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - genStart).count();
        double tps = tokenCount > 0 ? tokenCount * 1000.0 / genMs : 0;

        printf("\n\n--- Performance ---\n");
        printf("  Prompt:   %zu tokens\n", promptTokens.size());
        printf("  Generate: %d tokens in %.0fms (%.1f tok/s)\n", tokenCount, genMs, tps);
        return 0;
    }

    // Standard model path
    auto promptTokens = llm.Tokenize(finalPrompt);
    printf("Prompt: %zu tokens\n", promptTokens.size());
    if (temperature > 0)
        printf("Sampling: temperature=%.2f, top_k=%d, seed=%llu\n",
               temperature, topK, (unsigned long long)samplerSeed);
    printf("\n--- Output ---\n");
    if (!chat) printf("%s", finalPrompt.c_str());
    fflush(stdout);

    auto genStart = std::chrono::steady_clock::now();
    int tokenCount = 0;

    if (temperature > 0.0f) {
        app::SamplingParams sp{temperature, topK, samplerSeed};
        std::mt19937 rng(samplerSeed ? samplerSeed : std::random_device{}());
        llm.GenerateSampled(finalPrompt, maxTokens, sp, rng,
            [&](const std::string& text) {
                printf("%s", text.c_str()); fflush(stdout);
                tokenCount++;
                return true;
            });
    } else {
        llm.Generate(finalPrompt, maxTokens, [&](const std::string& text) {
            printf("%s", text.c_str()); fflush(stdout);
            tokenCount++;
            return true;
        });
    }

    auto genMs = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - genStart).count();
    double tps = tokenCount > 0 ? tokenCount * 1000.0 / genMs : 0;

    printf("\n\n--- Performance ---\n");
    printf("  Prompt:   %zu tokens\n", promptTokens.size());
    printf("  Generate: %d tokens in %.0fms (%.1f tok/s)\n", tokenCount, genMs, tps);
    return 0;
}
