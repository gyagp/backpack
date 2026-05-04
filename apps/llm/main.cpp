/**
 * Backpack LLM — End-to-end LLM text generation.
 *
 * Thin CLI that uses bp::LmSession (Layer 2 API) for all LLM operations.
 *
 * Usage:
 *   backpack_llm --model path/to/model  --chat "What is 2+2?"
 *   backpack_llm --model path/to/model  --prompt "Hello"
 *   backpack_llm --model path/to/model  --benchmark
 */

#include "backpack.h"
#include "lm_session.h"

// Internal access for baseline memory stats (not part of public API)
#include "gpu_context.h"

// Shared app utilities
#include "../common/app_common.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string modelPath, prompt, chatMessage, backendStr;
    std::string formatOverride;
    int maxTokens = 100;
    bool benchmark = false, profile = false, noFastDecode = false;
    bool saveBaseline = false;
    std::string baselinePath;
    int benchPromptLen = 0, benchGenTokens = 128;
    int maxSeqLenOverride = 0;
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
        else if (arg == "--fast-decode")              {}
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

    modelPath = app::discoverModelPath(modelPath);

    if (prompt.empty() && chatMessage.empty() && !benchmark)
        prompt = "Hello";

    // 1. Create device
    auto device = app::createDevice(backendStr);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }

    // 2. Load model via LmSession
    bp::LmOptions opts;
    opts.fastDecode = !noFastDecode;
    opts.maxSeqLen = maxSeqLenOverride;
    opts.warmupPipelines = true;

    auto t0 = std::chrono::steady_clock::now();

    auto session = formatOverride.empty()
        ? bp::LmSession::Create(device, modelPath, opts)
        : bp::LmSession::Create(device, modelPath, formatOverride, opts);

    if (!session.IsValid()) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    fprintf(stderr, "  [main] getting config...\n"); fflush(stderr);
    auto cfg = session.GetConfig();
    fprintf(stderr, "  [main] config: arch=%s format=%s layers=%d\n",
            cfg.arch.c_str(), cfg.format.c_str(), cfg.layers); fflush(stderr);
    fprintf(stderr, "\nModel: %s (%s, %dL, H=%d, V=%d)\n",
           cfg.arch.c_str(), cfg.format.c_str(),
           cfg.layers, cfg.hiddenSize, cfg.vocabSize);
    fprintf(stderr, "GPU:   %s (%s)\n", device.GetName().c_str(), device.GetBackendName().c_str());
    fprintf(stderr, "Ready: %lldms\n\n", (long long)loadMs);

    // Resolve baseline output path
    if (saveBaseline && baselinePath.empty()) {
        auto baselineDir = fs::path(__FILE__).parent_path().parent_path() / "baseline";
        fs::create_directories(baselineDir);
        baselinePath = (baselineDir / (cfg.arch + ".json")).string();
    }

    // 3. Benchmark
    if (benchmark) {
        std::vector<app::BenchResultEntry> benchResults;
        std::vector<int> lens;
        if (benchPromptLen > 0) lens.push_back(benchPromptLen);
        else lens = {128, 256, 512, 1024, 2048, 4096};

        fprintf(stderr, "=== Benchmark: %s ===\n", cfg.arch.c_str());
        fprintf(stderr, "%-12s %10s %10s %10s %10s %12s %8s\n",
               "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s", "fence_ms", "fence%");
        fprintf(stderr, "%-12s %10s %10s %10s %10s %12s %8s\n",
               "----------", "----------", "--------", "---------", "--------", "--------", "------");

        int genTokens = benchGenTokens > 0 ? benchGenTokens : 128;

        for (int pl : lens) {
            auto r = session.Benchmark(pl, genTokens);
            if (r.prefillMs == 0 && r.decodeTokPerSec == 0) {
                fprintf(stderr, "%-12d   (skipped — exceeds maxSeqLen)\n", pl);
                continue;
            }
            double fencePct = r.decodeMs > 0 ? 100.0 * r.fenceWaitMs / r.decodeMs : 0;
            fprintf(stderr, "%-12d %10.1f %10.1f %10.1f %10.1f %12.1f %7.1f%%\n",
                   pl, r.prefillMs, r.prefillTokPerSec, r.decodeMs, r.decodeTokPerSec,
                   r.fenceWaitMs, fencePct);
            fflush(stderr);
            benchResults.push_back({pl, r.prefillMs, r.prefillTokPerSec,
                                    r.decodeMs, r.decodeTokPerSec, r.ttftMs});
        }

        if (profile) {
            fprintf(stderr, "\n=== GPU Hardware Timestamp Profile ===\n");
            std::string htmlPath = "profile.html";
            session.PrintProfileReport(htmlPath);
        }

        // Save baseline JSON
        if (saveBaseline) {
            auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
            auto sysInfo = app::getSystemInfo();
            auto memStats = gpuCtx->getMemoryStats();
            app::MemoryInfo memInfo{memStats.peakBytes, memStats.currentBytes, memStats.allocCount};
            app::LoadingInfo loadInfo;
            loadInfo.totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            app::writeBaselineJson(baselinePath, sysInfo,
                device.GetName(), device.GetBackendName(),
                gpuCtx->adapterDescription,
                cfg.arch, modelPath, cfg.format,
                cfg.layers, cfg.hiddenSize, cfg.vocabSize,
                genTokens, benchResults,
                &loadInfo, &memInfo);
        }

        fprintf(stderr, "\n");
        return 0;
    }

    // 4. Generate
    std::string finalPrompt;
    bool chat = !chatMessage.empty();
    if (chat) {
        finalPrompt = app::applyChatTemplate(chatMessage, cfg.arch);
        fprintf(stderr, "Chat: %s\n", chatMessage.c_str());
    } else {
        finalPrompt = prompt;
    }

    auto promptTokens = session.Tokenize(finalPrompt);
    fprintf(stderr, "Prompt: %zu tokens\n", promptTokens.size());
    if (temperature > 0)
        fprintf(stderr, "Sampling: temperature=%.2f, top_k=%d, seed=%llu\n",
               temperature, topK, (unsigned long long)samplerSeed);
    fprintf(stderr, "\n--- Output ---\n");
    if (!chat) fprintf(stderr, "%s", finalPrompt.c_str());
    fflush(stderr);

    bp::SamplingParams sp{temperature, topK, samplerSeed};
    auto genStart = std::chrono::steady_clock::now();
    int tokenCount = 0;

    session.Generate(finalPrompt, maxTokens, sp,
        [&](const std::string& text) {
            fprintf(stderr, "%s", text.c_str()); fflush(stderr);
            tokenCount++;
            return true;
        });

    auto genMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - genStart).count();
    double tps = tokenCount > 0 ? tokenCount * 1000.0 / genMs : 0;

    fprintf(stderr, "\n\n--- Performance ---\n");
    fprintf(stderr, "  Prompt:   %zu tokens\n", promptTokens.size());
    fprintf(stderr, "  Generate: %d tokens in %.0fms (%.1f tok/s)\n", tokenCount, genMs, tps);
    return 0;
}
