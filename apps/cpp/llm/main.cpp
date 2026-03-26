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
#include "json_parser.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Path resolution ─────────────────────────────────────────────────────────

/// Check if a directory is a standard ONNX model (transformer architecture).
/// Requires config (genai_config.json or config.json) + tokenizer + .onnx file.
static bool isStandardOnnxDir(const std::string& path) {
    if (!fs::is_directory(path)) return false;
    // Must have some config
    bool hasConfig = fs::exists(fs::path(path) / "genai_config.json") ||
                     fs::exists(fs::path(path) / "config.json");
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

static std::string resolvePath(const std::string& path, std::string& format) {
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

// ─── Chat template ───────────────────────────────────────────────────────────

static std::string applyChatTemplate(const std::string& message,
                                      const std::string& arch) {
    if (arch.find("qwen3") != std::string::npos)
        return "<|im_start|>user\n" + message + "<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n</think>\n";
    return "<|im_start|>user\n" + message + "<|im_end|>\n<|im_start|>assistant\n";
}

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
    int64_t numKvHeads = 0, headDim = 0;
    int64_t convLCache = 3;
    std::vector<std::string> layerTypes;
    std::string arch;

    // KV cache state
    uint32_t pos = 0;
    std::unordered_map<std::string, GpuTensor> convState;
    std::unordered_map<std::string, GpuTensor> kvState;
    std::vector<int> convLayerIndices, attnLayerIndices;

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
        int64_t numHeads = cfg.has("num_attention_heads") ? cfg["num_attention_heads"].as_int() : 32;
        headDim = hiddenSize / numHeads;
        convLCache = cfg.has("conv_L_cache") ? cfg["conv_L_cache"].as_int() : 3;
        arch = cfg.has("model_type") ? cfg["model_type"].as_string() : "onnx";

        if (cfg.has("layer_types") && cfg["layer_types"].is_array()) {
            for (int64_t i = 0; i < cfg["layer_types"].size(); i++) {
                std::string lt = cfg["layer_types"][i].as_string();
                layerTypes.push_back(lt);
                if (lt == "conv") convLayerIndices.push_back((int)i);
                else if (lt == "full_attention") attnLayerIndices.push_back((int)i);
            }
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
        convState.clear();
        kvState.clear();
        for (int idx : convLayerIndices) {
            std::string name = "past_conv." + std::to_string(idx);
            GpuTensor t;
            t.shape = {1, hiddenSize, convLCache};
            t.dtype = TensorDtype::Float16;
            size_t bytes = (size_t)(hiddenSize * convLCache * 2);
            t.buffer = gpu->createBuffer(name, bytes);
            std::vector<uint16_t> zeros((size_t)(hiddenSize * convLCache), 0);
            gpu->writeBuffer(t.buffer, zeros.data(), bytes);
            convState[name] = t;
        }
        for (int idx : attnLayerIndices) {
            for (const char* suffix : {".key", ".value"}) {
                std::string name = "past_key_values." + std::to_string(idx) + suffix;
                GpuTensor t;
                t.shape = {1, numKvHeads, 0, headDim};
                t.dtype = TensorDtype::Float16;
                t.buffer = gpu->createBuffer(name, 4);
                kvState[name] = t;
            }
        }
    }

    // Run a single decode step and return logits
    std::vector<float> RunStep(int64_t tokenId) {
        pos++;

        std::unordered_map<std::string, GpuTensor*> inputs;

        // input_ids: [1, 1]
        GpuTensor idT;
        idT.shape = {1, 1};
        idT.dtype = TensorDtype::Int64;
        idT.buffer = gpu->createBuffer("ids", 8);
        gpu->writeBuffer(idT.buffer, &tokenId, 8);
        idT.cpuData.resize(8);
        memcpy(idT.cpuData.data(), &tokenId, 8);
        inputs["input_ids"] = &idT;

        // attention_mask: [1, pos]
        std::vector<int64_t> mask(pos, 1);
        GpuTensor maskT;
        maskT.shape = {1, (int64_t)pos};
        maskT.dtype = TensorDtype::Int64;
        maskT.buffer = gpu->createBuffer("mask", pos * 8);
        gpu->writeBuffer(maskT.buffer, mask.data(), pos * 8);
        maskT.cpuData.resize(pos * 8);
        memcpy(maskT.cpuData.data(), mask.data(), pos * 8);
        inputs["attention_mask"] = &maskT;

        // num_logits_to_keep
        int64_t nlk = 1;
        GpuTensor nlkT;
        nlkT.shape = {};
        nlkT.dtype = TensorDtype::Int64;
        nlkT.buffer = gpu->createBuffer("nlk", 8);
        gpu->writeBuffer(nlkT.buffer, &nlk, 8);
        nlkT.cpuData.resize(8);
        memcpy(nlkT.cpuData.data(), &nlk, 8);
        inputs["num_logits_to_keep"] = &nlkT;

        // Caches
        for (auto& [name, t] : convState) inputs[name] = &t;
        for (auto& [name, t] : kvState) inputs[name] = &t;

        // Outputs
        std::unordered_map<std::string, GpuTensor*> outputs;

        GpuTensor logitsOut;
        logitsOut.shape = {1, 1, vocabSize};
        logitsOut.dtype = TensorDtype::Float32;
        logitsOut.buffer = gpu->createBuffer("logits_out", vocabSize * 4);
        outputs["logits"] = &logitsOut;

        std::vector<GpuTensor> convOuts(convLayerIndices.size());
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string name = "present_conv." + std::to_string(convLayerIndices[i]);
            convOuts[i].shape = {1, hiddenSize, convLCache};
            convOuts[i].dtype = TensorDtype::Float16;
            convOuts[i].buffer = gpu->createBuffer(name + "_o", hiddenSize * convLCache * 2);
            outputs[name] = &convOuts[i];
        }

        std::vector<GpuTensor> kvOuts(attnLayerIndices.size() * 2);
        for (size_t i = 0; i < attnLayerIndices.size(); i++) {
            std::string kName = "present." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vName = "present." + std::to_string(attnLayerIndices[i]) + ".value";
            size_t kvBytes = (size_t)(numKvHeads * pos * headDim * 4);
            if (kvBytes == 0) kvBytes = 4;
            kvOuts[i*2].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2].dtype = TensorDtype::Float32;
            kvOuts[i*2].buffer = gpu->createBuffer(kName + "_o", kvBytes);
            kvOuts[i*2+1].shape = {1, numKvHeads, (int64_t)pos, headDim};
            kvOuts[i*2+1].dtype = TensorDtype::Float32;
            kvOuts[i*2+1].buffer = gpu->createBuffer(vName + "_o", kvBytes);
            outputs[kName] = &kvOuts[i*2];
            outputs[vName] = &kvOuts[i*2+1];
        }

        executor.Execute(inputs, outputs);

        // Update caches
        for (size_t i = 0; i < convLayerIndices.size(); i++) {
            std::string inName = "past_conv." + std::to_string(convLayerIndices[i]);
            if (convOuts[i].IsValid()) convState[inName] = convOuts[i];
        }
        for (size_t i = 0; i < attnLayerIndices.size(); i++) {
            std::string kIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".key";
            std::string vIn = "past_key_values." + std::to_string(attnLayerIndices[i]) + ".value";
            if (kvOuts[i*2].IsValid()) kvState[kIn] = kvOuts[i*2];
            if (kvOuts[i*2+1].IsValid()) kvState[vIn] = kvOuts[i*2+1];
        }

        // Read logits
        executor.FlushPendingWork();
        gpu->waitForQueue();

        int64_t logitNel = logitsOut.ElementCount();
        std::vector<float> logits(logitNel);
        if (logitsOut.dtype == TensorDtype::Float32) {
            auto rb = gpu->readBuffer(logitsOut.buffer, logitNel * 4);
            memcpy(logits.data(), rb.data(), logitNel * 4);
        } else {
            auto rb = gpu->readBuffer(logitsOut.buffer, logitNel * 2);
            auto fp16 = reinterpret_cast<const uint16_t*>(rb.data());
            for (int64_t i = 0; i < logitNel; i++) {
                uint32_t h = fp16[i], s = (h>>15)&1, e = (h>>10)&0x1F, m = h&0x3FF;
                uint32_t f;
                if (e == 0) f = (s<<31)|(m<<13);
                else if (e == 31) f = (s<<31)|0x7F800000|(m<<13);
                else f = (s<<31)|((e+112)<<23)|(m<<13);
                memcpy(&logits[i], &f, 4);
            }
        }
        return logits;
    }

    int32_t Argmax(const std::vector<float>& logits) {
        float mx = -1e30f;
        int32_t idx = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            if (logits[i] > mx) { mx = logits[i]; idx = (int32_t)i; }
        }
        return idx;
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
    int maxTokens = 100;
    bool benchmark = false;
    int benchPromptLen = 0, benchGenTokens = 128;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc)           modelPath = argv[++i];
        else if (arg == "--prompt" && i+1 < argc)     prompt = argv[++i];
        else if (arg == "--chat" && i+1 < argc)       chatMessage = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) maxTokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc)    backendStr = argv[++i];
        else if (arg == "--benchmark")                benchmark = true;
        else if (arg == "--bench-prompt-len" && i+1 < argc) benchPromptLen = atoi(argv[++i]);
        else if (arg == "--bench-gen-tokens" && i+1 < argc) benchGenTokens = atoi(argv[++i]);
    }

    if (modelPath.empty()) {
        fprintf(stderr,
            "Backpack LLM — WebGPU text generation\n\n"
            "Usage: %s --model <path> [options]\n\n"
            "  --prompt <text>    Raw text prompt\n"
            "  --chat <message>   Chat message (auto-template)\n"
            "  --max-tokens <n>   Max tokens (default: 100)\n"
            "  --backend <name>   vulkan / d3d12 / metal\n"
            "  --benchmark        Prefill+decode sweep\n", argv[0]);
        return 1;
    }

    if (prompt.empty() && chatMessage.empty() && !benchmark)
        prompt = "Hello";

    // 1. Create device via runtime API
    bp::Backend backend = bp::Backend::Default;
    if (backendStr == "d3d12")       backend = bp::Backend::D3D12;
    else if (backendStr == "vulkan") backend = bp::Backend::Vulkan;
    else if (backendStr == "metal")  backend = bp::Backend::Metal;

    auto device = bp::Device::Create(backend);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }

    // 2. Load LLM
    std::string modelFormat;
    std::string resolvedPath = resolvePath(modelPath, modelFormat);
    printf("Loading model: %s (%s)\n", resolvedPath.c_str(), modelFormat.c_str()); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();

    // Standard transformer models (GGUF or ONNX)
    LlmContext llm;
    // Generic ONNX models (LFM2, etc.)
    OnnxLlmContext onnxLlm;
    bool isGenericOnnx = (modelFormat == "onnx_generic");

    if (isGenericOnnx) {
        if (!onnxLlm.Load(*static_cast<GPUContext*>(device.GetGPUContext()), resolvedPath)) {
            fprintf(stderr, "Failed to load model\n"); return 1;
        }
    } else {
        if (!llm.Load(*static_cast<GPUContext*>(device.GetGPUContext()), modelPath)) {
            fprintf(stderr, "Failed to load model\n"); return 1;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    if (isGenericOnnx) {
        printf("\nModel: %s (generic ONNX, %lldL, H=%lld, V=%lld)\n",
               onnxLlm.arch.c_str(), (long long)onnxLlm.numLayers,
               (long long)onnxLlm.hiddenSize, (long long)onnxLlm.vocabSize);
    } else {
        printf("\nModel: %s (%s, %uL, E=%u, HD=%u, V=%u)\n",
               llm.arch.c_str(), llm.format.c_str(),
               llm.nLayer, llm.nEmbd, llm.headDim, llm.nVocab);
    }
    printf("GPU:   %s (%s)\n", device.GetName().c_str(), device.GetBackendName().c_str());
    printf("Ready: %lldms\n\n", (long long)loadMs);

    // 3. Benchmark
    if (benchmark) {
        if (isGenericOnnx) {
            fprintf(stderr, "Benchmark not yet supported for generic ONNX models.\n");
            return 1;
        }
        printf("=== Benchmark: %s ===\n", llm.arch.c_str());
        printf("%-12s %10s %10s %10s %10s\n",
               "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s");
        printf("%-12s %10s %10s %10s %10s\n",
               "----------", "----------", "--------", "---------", "--------");
        std::vector<int> lens;
        if (benchPromptLen > 0) lens.push_back(benchPromptLen);
        else lens = {128, 256, 512, 1024, 2048, 4096};
        for (int pl : lens) {
            auto r = llm.Benchmark(pl, benchGenTokens);
            printf("%-12d %10.1f %10.1f %10.1f %10.1f\n",
                   pl, r.pfMs, r.pfTps, r.dcMs, r.dcTps);
            fflush(stdout);
        }
        printf("\n");
        return 0;
    }

    // 4. Generate
    std::string finalPrompt;
    bool chat = !chatMessage.empty();
    if (chat) {
        std::string arch_str = isGenericOnnx ? onnxLlm.arch : llm.arch;
        finalPrompt = applyChatTemplate(chatMessage, arch_str);
        printf("Chat: %s\n", chatMessage.c_str());
    } else {
        finalPrompt = prompt;
    }

    if (isGenericOnnx) {
        auto promptTokens = onnxLlm.tokenizer.encode(finalPrompt);
        printf("Prompt: %zu tokens\n\n--- Output ---\n", promptTokens.size());
        if (!chat) printf("%s", finalPrompt.c_str());
        fflush(stdout);

        auto genStart = std::chrono::steady_clock::now();
        int tokenCount = 0;
        int32_t eos = onnxLlm.tokenizer.eos_token_id;

        onnxLlm.ResetCaches();
        // Prefill: feed prompt tokens one at a time
        int32_t next = 0;
        for (size_t i = 0; i < promptTokens.size(); i++) {
            auto logits = onnxLlm.RunStep(promptTokens[i]);
            if (i == promptTokens.size() - 1) {
                next = onnxLlm.Argmax(logits);
            }
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
            next = onnxLlm.Argmax(logits);
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
    printf("Prompt: %zu tokens\n\n--- Output ---\n", promptTokens.size());
    if (!chat) printf("%s", finalPrompt.c_str());
    fflush(stdout);

    auto genStart = std::chrono::steady_clock::now();
    int tokenCount = 0;

    llm.Generate(finalPrompt, maxTokens, [&](const std::string& text) {
        printf("%s", text.c_str()); fflush(stdout);
        tokenCount++;
        return true;
    });

    auto genMs = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - genStart).count();
    double tps = tokenCount > 0 ? tokenCount * 1000.0 / genMs : 0;

    printf("\n\n--- Performance ---\n");
    printf("  Prompt:   %zu tokens\n", promptTokens.size());
    printf("  Generate: %d tokens in %.0fms (%.1f tok/s)\n", tokenCount, genMs, tps);
    return 0;
}
