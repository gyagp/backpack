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
#include "tokenizer.h"
#include "onnx_tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Path resolution ─────────────────────────────────────────────────────────

static bool isOnnxDir(const std::string& path) {
    return fs::is_directory(path) &&
           fs::exists(fs::path(path) / "model.onnx") &&
           fs::exists(fs::path(path) / "genai_config.json");
}

static std::string resolvePath(const std::string& path, std::string& format) {
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf") {
        if (fs::exists(path)) { format = "gguf"; return path; }
    }
    if (isOnnxDir(path)) { format = "onnx"; return path; }
    if (fs::is_directory(path)) {
        for (auto& e : fs::directory_iterator(path))
            if (e.is_directory() && isOnnxDir(e.path().string()))
                { format = "onnx"; return e.path().string(); }
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

    // 2. Load LLM (uses internals, not Layer 1 Model/Session)
    printf("Loading model: %s\n", modelPath.c_str()); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();

    LlmContext llm;
    if (!llm.Load(*static_cast<GPUContext*>(device.GetGPUContext()), modelPath)) {
        fprintf(stderr, "Failed to load model\n"); return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    printf("\nModel: %s (%s, %uL, E=%u, HD=%u, V=%u)\n",
           llm.arch.c_str(), llm.format.c_str(),
           llm.nLayer, llm.nEmbd, llm.headDim, llm.nVocab);
    printf("GPU:   %s (%s)\n", device.GetName().c_str(), device.GetBackendName().c_str());
    printf("Ready: %lldms\n\n", (long long)loadMs);

    // 3. Benchmark
    if (benchmark) {
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
        finalPrompt = applyChatTemplate(chatMessage, llm.arch);
        printf("Chat: %s\n", chatMessage.c_str());
    } else {
        finalPrompt = prompt;
    }

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
