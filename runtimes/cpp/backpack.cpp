/**
 * backpack.cpp — Implementation of the Backpack public API.
 *
 * Wraps ModelRunner, GPUContext, and Tokenizer/OnnxTokenizer behind
 * opaque BpModel/BpContext/BpTokenizer handles.
 */

#include "backpack.h"

#include "gpu_context.h"
#include "model_runner.h"
#include "tokenizer.h"
#include "onnx_tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ─── Internal structs (opaque to callers) ────────────────────────────────────

struct BpTokenizer {
    BpModel* model = nullptr;
};

struct BpModel {
    GPUContext gpu;
    ModelRunner runner;
    Tokenizer ggufTokenizer;
    OnnxTokenizer onnxTokenizer;
    std::string format;         // "gguf" or "onnx"
    std::string path;
    BpModelInfo info;
    BpTokenizer tokenizer;      // inline tokenizer (back-pointer set in load)
};

struct BpContext {
    BpModel* model = nullptr;
    uint32_t pos = 0;           // current sequence position
    bool warmupDone = false;
    bool autotuneDone = false;
    bool benchWarmupDone = false;
};

// ─── Path resolution ─────────────────────────────────────────────────────────

static bool isOnnxDir(const std::string& path) {
    return fs::is_directory(path) &&
           fs::exists(fs::path(path) / "model.onnx") &&
           fs::exists(fs::path(path) / "genai_config.json");
}

static std::string resolveModelPath(const std::string& path, std::string& format) {
    // Direct GGUF file
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf") {
        if (fs::exists(path)) { format = "gguf"; return path; }
    }

    // ONNX directory
    if (isOnnxDir(path)) { format = "onnx"; return path; }

    // Directory search
    if (fs::is_directory(path)) {
        // Check for ONNX subdirs
        for (auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory() && isOnnxDir(entry.path().string())) {
                format = "onnx";
                return entry.path().string();
            }
        }
        // Search for GGUF
        std::string bestQ8, bestQ4, first;
        for (auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                auto name = entry.path().filename().string();
                if (first.empty()) first = entry.path().string();
                if (name.find("Q8_0") != std::string::npos) bestQ8 = entry.path().string();
                else if (name.find("Q4_K") != std::string::npos) bestQ4 = entry.path().string();
            }
        }
        if (!bestQ8.empty()) { format = "gguf"; return bestQ8; }
        if (!bestQ4.empty()) { format = "gguf"; return bestQ4; }
        if (!first.empty())  { format = "gguf"; return first; }
    }

    format = "gguf";
    return path;
}

static WGPUBackendType toWGPU(BpBackend b) {
    switch (b) {
        case BpBackend::D3D12:  return WGPUBackendType_D3D12;
        case BpBackend::Metal:  return WGPUBackendType_Metal;
        case BpBackend::Vulkan: return WGPUBackendType_Vulkan;
        default: {
#if defined(_WIN32)
            return WGPUBackendType_D3D12;
#elif defined(__APPLE__)
            return WGPUBackendType_Metal;
#else
            return WGPUBackendType_Vulkan;
#endif
        }
    }
}

static std::string backendStr(WGPUBackendType bt) {
    switch (bt) {
        case WGPUBackendType_D3D12: return "d3d12";
        case WGPUBackendType_Metal: return "metal";
        case WGPUBackendType_Vulkan: return "vulkan";
        default: return "unknown";
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

BpModel* bp_model_load(const std::string& path, const BpModelParams& params) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered stdout for diagnostics
    setvbuf(stderr, nullptr, _IONBF, 0);

    auto* m = new BpModel();
    m->path = path;
    m->tokenizer.model = m;

    // Resolve path + detect format
    fprintf(stderr, "[bp] resolving path: %s\n", path.c_str());
    std::string resolvedPath = resolveModelPath(path, m->format);
    fprintf(stderr, "[bp] format=%s resolved=%s\n", m->format.c_str(), resolvedPath.c_str());

    // Init GPU
    WGPUBackendType backend = toWGPU(params.backend);
    if (!m->gpu.init(backend)) {
        fprintf(stderr, "bp_model_load: failed to initialize GPU\n");
        delete m;
        return nullptr;
    }

    m->runner.maxSeqLen = params.maxSeqLen;

    // Load model
    bool ok = false;
    if (m->format == "onnx") {
        ok = m->runner.loadOnnx(m->gpu, resolvedPath);
    } else {
        ok = m->runner.load(m->gpu, resolvedPath);
    }
    if (!ok) {
        fprintf(stderr, "bp_model_load: failed to load model from %s\n",
                resolvedPath.c_str());
        delete m;
        return nullptr;
    }

    // Load tokenizer
    fprintf(stderr, "[bp] loading tokenizer...\n"); fflush(stderr);
    if (m->format == "onnx") {
        if (!m->onnxTokenizer.load(resolvedPath)) {
            fprintf(stderr, "bp_model_load: failed to load ONNX tokenizer\n");
            delete m;
            return nullptr;
        }
    } else {
        if (!m->ggufTokenizer.load(m->runner.gguf)) {
            fprintf(stderr, "bp_model_load: failed to load GGUF tokenizer\n");
            delete m;
            return nullptr;
        }
    }

    // Populate info
    auto& cfg = m->runner.cfg;
    m->info.arch = cfg.arch;
    m->info.format = m->format;
    m->info.nLayer = cfg.nLayer;
    m->info.nHead = cfg.nHead;
    m->info.nKvHeads = cfg.nKvHeads;
    m->info.nEmbd = cfg.nEmbd;
    m->info.headDim = cfg.headDim;
    m->info.nVocab = cfg.nVocab;
    m->info.intermediateSize = cfg.intermediateSize;
    m->info.ropeTheta = cfg.ropeTheta;
    m->info.gpuName = m->gpu.adapterName;
    m->info.backendName = backendStr(m->gpu.backendType);

    return m;
}

void bp_model_free(BpModel* model) {
    if (!model) return;
#if !defined(_WIN32)
    model->runner.destroy();
    model->gpu.destroy();
#endif
    delete model;
}

BpModelInfo bp_model_info(const BpModel* model) {
    return model ? model->info : BpModelInfo{};
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

BpTokenizer* bp_tokenizer(const BpModel* model) {
    return model ? const_cast<BpTokenizer*>(&model->tokenizer) : nullptr;
}

std::vector<int32_t> bp_tokenize(const BpTokenizer* tok, const std::string& text) {
    if (!tok || !tok->model) return {};
    auto* m = tok->model;
    return (m->format == "onnx")
        ? m->onnxTokenizer.encode(text)
        : m->ggufTokenizer.encode(text);
}

std::string bp_token_to_text(const BpTokenizer* tok, int32_t token) {
    if (!tok || !tok->model) return "";
    auto* m = tok->model;
    return (m->format == "onnx")
        ? m->onnxTokenizer.decode_token(token)
        : m->ggufTokenizer.decode_token(token);
}

std::string bp_detokenize(const BpTokenizer* tok,
                           const std::vector<int32_t>& tokens) {
    if (!tok || !tok->model) return "";
    auto* m = tok->model;
    return (m->format == "onnx")
        ? m->onnxTokenizer.decode(tokens)
        : m->ggufTokenizer.decode(tokens);
}

int32_t bp_token_eos(const BpTokenizer* tok) {
    if (!tok || !tok->model) return -1;
    auto* m = tok->model;
    return (m->format == "onnx")
        ? m->onnxTokenizer.eos_token_id
        : m->ggufTokenizer.eos_token_id;
}

int32_t bp_token_bos(const BpTokenizer* tok) {
    if (!tok || !tok->model) return -1;
    auto* m = tok->model;
    return (m->format == "onnx")
        ? m->onnxTokenizer.bos_token_id
        : m->ggufTokenizer.bos_token_id;
}

// ─── Context ─────────────────────────────────────────────────────────────────

BpContext* bp_context_create(BpModel* model, const BpContextParams& params) {
    if (!model) return nullptr;

    auto* ctx = new BpContext();
    ctx->model = model;
    ctx->pos = 0;

    // Warmup: trigger shader compilation
    if (params.autoWarmup) {
        auto& runner = model->runner;
        fprintf(stderr, "[bp] warmup: decode(0,0)...\n"); fflush(stderr);
        auto logits = runner.decode(0, 0);
        fprintf(stderr, "[bp] warmup: decode done (%zu logits), resetting KV...\n", logits.size()); fflush(stderr);
        runner.resetKVCache();
        // Skip batched prefill warmup for now (may hang on some models)
        ctx->warmupDone = true;
    }

    // Autotune
    if (params.autoAutotune) {
        auto& runner = model->runner;
        if (!runner.loadDecodeAutotuneCache()) {
            runner.autotuneDecodeDepth();
            runner.autotuneDecodeKernels();
            runner.saveDecodeAutotuneCache();
        }
        ctx->autotuneDone = true;
    }

    return ctx;
}

void bp_context_free(BpContext* ctx) {
    delete ctx;
}

void bp_context_reset(BpContext* ctx) {
    if (!ctx) return;
    ctx->model->runner.resetKVCache();
    ctx->pos = 0;
}

uint32_t bp_context_pos(const BpContext* ctx) {
    return ctx ? ctx->pos : 0;
}

// ─── Low-level inference ─────────────────────────────────────────────────────

int32_t bp_prefill(BpContext* ctx, const int32_t* tokens, uint32_t nTokens) {
    if (!ctx || !tokens || nTokens == 0) return -1;

    auto& runner = ctx->model->runner;
    auto& gpu = ctx->model->gpu;

    // Sequential prefill: decode each prompt token one by one.
    // This is simpler and works correctly for all model formats.
    // (Batched prefill path has compatibility issues with some ONNX models.)
    std::vector<float> logits;
    for (uint32_t i = 0; i < nTokens; i++)
        logits = runner.decode(tokens[i], i);

    int32_t nextToken = ModelRunner::argmax(logits);

    // Seed argmax buffer for autoregressive decode
    gpu.writeBuffer(runner.argmaxResultBuf, &nextToken, 4);

    ctx->pos = nTokens;
    return nextToken;
}

int32_t bp_decode(BpContext* ctx) {
    if (!ctx) return -1;

    auto& runner = ctx->model->runner;
    const int DEPTH = runner.decodePoolDepth;

    // Simple single-step pipelined decode
    int slot = ctx->pos % DEPTH;
    runner.submitDecode(ctx->pos, slot);
    int32_t token = runner.readArgmax(slot);

    ctx->pos++;
    return token;
}

// ─── High-level generation ───────────────────────────────────────────────────

std::string bp_generate(BpContext* ctx, const std::string& prompt,
                         const BpGenerateParams& params,
                         BpStreamCallback onToken) {
    if (!ctx) return "";

    auto* tok = bp_tokenizer(ctx->model);
    int32_t eos = bp_token_eos(tok);

    // Reset state
    bp_context_reset(ctx);

    // Tokenize + prefill
    auto promptTokens = bp_tokenize(tok, prompt);
    if (promptTokens.empty()) return "";

    int32_t nextToken = bp_prefill(ctx, promptTokens.data(),
                                    (uint32_t)promptTokens.size());

    // Decode loop
    std::string result;
    auto& runner = ctx->model->runner;
    const int DEPTH = runner.decodePoolDepth;

    for (int step = 0; step < params.maxTokens; step++) {
        if (nextToken == eos) break;

        std::string text = bp_token_to_text(tok, nextToken);
        // Skip special tokens
        if (!(text.size() >= 2 && text[0] == '<' && text.back() == '>')) {
            result += text;
            if (onToken && !onToken(text)) break;
        }

        // Pipelined decode
        int slot = step % DEPTH;
        runner.submitDecode(ctx->pos, slot);
        nextToken = runner.readArgmax(slot);
        ctx->pos++;
    }

    return result;
}

// ─── Profiling ───────────────────────────────────────────────────────────────

void bp_enable_profiling(BpContext* ctx) {
    if (ctx) ctx->model->runner.enableProfiling();
}

void bp_print_profile(BpContext* ctx, const std::string& outputPath) {
    if (!ctx) return;
    ctx->model->runner.printProfileReport(0, 0, 0, 0, outputPath);
}

BpBenchResult bp_benchmark(BpContext* ctx, int promptLen, int genTokens) {
    BpBenchResult r{};
    if (!ctx) return r;

    auto& runner = ctx->model->runner;
    auto& gpu = ctx->model->gpu;
    const int DEPTH = runner.decodePoolDepth;

    // Warmup: run one full cycle to prime GPU caches and finish shader compilation
    if (!ctx->benchWarmupDone) {
        runner.resetKVCache();
        std::vector<int32_t> warmupTokens(std::min(promptLen, 32), 0);
        int32_t tok = runner.prefillBatched(warmupTokens.data(),
                                             (uint32_t)warmupTokens.size(), 0);
        gpu.writeBuffer(runner.argmaxResultBuf, &tok, 4);
        int warmupDecode = std::min(DEPTH, 4);
        for (int i = 0; i < warmupDecode; i++)
            runner.submitDecode((uint32_t)(warmupTokens.size() + i), i);
        for (int i = 0; i < warmupDecode; i++)
            runner.readArgmax(i);
        runner.resetKVCache();
        ctx->benchWarmupDone = true;
    }

    // Prefill (timed)
    runner.resetKVCache();
    std::vector<int32_t> dummyTokens(promptLen, 0);
    auto pf_t0 = std::chrono::steady_clock::now();
    int32_t firstTok = runner.prefillBatched(dummyTokens.data(),
                                              (uint32_t)promptLen, 0);
    auto pf_t1 = std::chrono::steady_clock::now();
    r.prefillMs = std::chrono::duration<double, std::milli>(pf_t1 - pf_t0).count();
    r.prefillTps = promptLen * 1000.0 / r.prefillMs;
    r.nPrefillTokens = promptLen;

    gpu.writeBuffer(runner.argmaxResultBuf, &firstTok, 4);

    // Decode
    auto dc_t0 = std::chrono::steady_clock::now();
    int submitted = 0, completed = 0;
    int primeCount = std::min(DEPTH, genTokens);
    for (int i = 0; i < primeCount; i++) {
        runner.submitDecode((uint32_t)(promptLen + i), i);
        submitted++;
    }
    while (completed < submitted) {
        int slot = completed % DEPTH;
        (void)runner.readArgmax(slot);
        completed++;
        if (submitted < genTokens) {
            runner.submitDecode((uint32_t)(promptLen + submitted), slot);
            submitted++;
        }
    }
    auto dc_t1 = std::chrono::steady_clock::now();
    r.decodeMs = std::chrono::duration<double, std::milli>(dc_t1 - dc_t0).count();
    r.decodeTps = completed * 1000.0 / r.decodeMs;
    r.nDecodeTokens = completed;

    runner.resetKVCache();
    ctx->pos = 0;

    return r;
}
