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
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>

namespace fs = std::filesystem;

static int runApp(int argc, char* argv[]) {
    std::string modelPath, prompt, chatMessage, backendStr;
    std::string formatOverride;
    int maxTokens = 100;
    bool benchmark = false, profile = false, noFastDecode = false;
    bool fastPrefill = false;
    bool saveBaseline = false;
    std::string baselinePath;
    int benchPromptLen = 0, benchGenTokens = 128;
    int benchRepetitions = 1, benchWarmups = 1;
    std::string benchPromptFile, benchJsonPath;
    int maxSeqLenOverride = 0;
    int prefillChunkSize = 0;
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
        else if (arg == "--fast-prefill")             fastPrefill = true;
        else if (arg == "--bench-prompt-len" && i+1 < argc) benchPromptLen = atoi(argv[++i]);
        else if (arg == "--bench-gen-tokens" && i+1 < argc) benchGenTokens = atoi(argv[++i]);
        else if (arg == "--bench-repetitions" && i+1 < argc) benchRepetitions = atoi(argv[++i]);
        else if (arg == "--bench-warmup" && i+1 < argc) benchWarmups = atoi(argv[++i]);
        else if (arg == "--bench-prompt-file" && i+1 < argc) benchPromptFile = argv[++i];
        else if (arg == "--bench-json" && i+1 < argc) benchJsonPath = argv[++i];
        else if (arg == "--max-seq-len" && i+1 < argc)    maxSeqLenOverride = atoi(argv[++i]);
        else if (arg == "--prefill-chunk" && i+1 < argc) prefillChunkSize = atoi(argv[++i]);
        else if (arg == "--temperature" && i+1 < argc)   temperature = (float)atof(argv[++i]);
        else if (arg == "--top-k" && i+1 < argc)         topK = atoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc)           samplerSeed = (uint64_t)atoll(argv[++i]);
        else if (arg == "--save-baseline") {
            saveBaseline = true;
            if (i+1 < argc && argv[i+1][0] != '-') baselinePath = argv[++i];
        }
    }

    if (saveBaseline) benchmark = true;
    if (!benchJsonPath.empty() || !benchPromptFile.empty()) benchmark = true;
    if (benchmark && !chatMessage.empty()) {
        fprintf(stderr, "Benchmark expects raw prompt text; use --prompt or --bench-prompt-file\n");
        return 2;
    }
    if (benchmark && (benchGenTokens < 1 || benchRepetitions < 1 || benchWarmups < 0 || benchPromptLen < 0)) {
        fprintf(stderr, "Benchmark output/repetition counts must be positive; warmup/prompt limits must be non-negative\n");
        return 2;
    }
    if (!benchPromptFile.empty()) {
        if (!prompt.empty()) throw std::runtime_error("Use either --prompt or --bench-prompt-file");
        std::ifstream input(benchPromptFile, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read benchmark prompt file");
        prompt.assign(std::istreambuf_iterator<char>(input), {});
        if (prompt.empty()) throw std::runtime_error("Benchmark prompt file is empty");
    }
    if (maxTokens < 0 || maxSeqLenOverride < 0 || prefillChunkSize < 0) {
        fprintf(stderr, "Token and context limits must be non-negative\n");
        return 2;
    }

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
            "  --fast-prefill     Use experimental batched Qwen 3.5 prefill\n"
            "  --temperature <f>  Sampling temperature (0 = greedy)\n"
            "  --top-k <n>       Top-k sampling (0 = disabled)\n"
            "  --seed <n>        Random seed for sampling\n"
            "  --benchmark        Prefill+decode sweep\n"
            "  --bench-prompt-len <n>  Benchmark one prompt length instead of the full sweep\n"
            "  --bench-gen-tokens <n>  Generated tokens per benchmark row (default: 128)\n"
            "  --bench-repetitions <n>  Measured runs per prompt (default: 1)\n"
            "  --bench-warmup <n>  Separate complete warmup runs (default: 1)\n"
            "  --bench-prompt-file <path>  Exact raw benchmark text\n"
            "  --bench-json <path>  Per-run timings, counts and token IDs\n"
            "  --profile          GPU timestamp profiling + HTML timeline\n"
            "  --save-baseline [path]  Save benchmark results to JSON\n"
            "  --prefill-chunk <n> Generic ONNX prompt batch limit (default: whole prompt)\n"
            "  --max-seq-len <n>  Max context length (default: auto from GPU memory)\n", argv[0]);
        return 1;
    }

    modelPath = app::discoverModelPath(modelPath);

    if (prompt.empty() && chatMessage.empty() && !benchmark)
        prompt = "Hello";

    // Runtime selection defaults to validated device/model tuples. This flag
    // is an explicit experimental override for conformance investigation.
    if (fastPrefill) {
#ifdef _WIN32
        _putenv_s("BP_QWEN35_FORCE_FAST_PREFILL", "1");
#else
        setenv("BP_QWEN35_FORCE_FAST_PREFILL", "1", 1);
#endif
    }

    // 1. Create device
    auto device = app::createDevice(backendStr);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }

    // 2. Load model via LmSession
    bp::LmOptions opts;
    opts.fastDecode = !noFastDecode;
    opts.maxSeqLen = maxSeqLenOverride;
    opts.warmupPipelines = true;
    opts.prefillChunkSize = static_cast<uint32_t>(prefillChunkSize);

    auto t0 = std::chrono::steady_clock::now();

    auto session = formatOverride.empty()
        ? bp::LmSession::Create(device, modelPath, opts)
        : bp::LmSession::Create(device, modelPath, formatOverride, opts);

    if (!session.IsValid()) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    if (static_cast<GPUContext*>(device.GetGPUContext())->executionError)
        throw std::runtime_error("Model initialization reported a WebGPU error; see diagnostics above");

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
        auto baselineDir = fs::path("gitignore") / "logs" / "baseline";
        fs::create_directories(baselineDir);
        baselinePath = (baselineDir / (cfg.arch + ".json")).string();
    }

    // 3. Benchmark
    if (benchmark) {
        const uint32_t effectiveChunk = cfg.format == "onnx_generic" ? opts.prefillChunkSize : 0;
        const std::string benchmarkProtocol = "llm-fixed-text-reuse-first-token-v2" +
            (effectiveChunk ? "-chunk" + std::to_string(effectiveChunk) : std::string{});
        std::vector<app::BenchResultEntry> benchResults;
        std::vector<int> lens;
        std::vector<int32_t> explicitTokens;
        if (!prompt.empty()) {
            explicitTokens = session.TokenizeRaw(prompt);
            if (benchPromptLen > 0 && explicitTokens.size() != size_t(benchPromptLen))
                throw std::runtime_error("Benchmark prompt token count differs from --bench-prompt-len");
            lens.push_back(static_cast<int>(explicitTokens.size()));
        } else if (benchPromptLen > 0) lens.push_back(benchPromptLen);
        else lens = {128, 256, 512, 1024, 2048, 4096};

        std::ofstream details;
        if (!benchJsonPath.empty()) {
            const auto parent = fs::path(benchJsonPath).parent_path();
            if (!parent.empty()) fs::create_directories(parent);
            details.open(benchJsonPath);
            if (!details) throw std::runtime_error("Cannot write benchmark JSON");
            details << std::setprecision(12)
                << "{\"benchmark_protocol\":\"" << benchmarkProtocol << "\",\"warmup_runs\":" << benchWarmups
                << ",\"reuse_generator\":true,\"prefill_includes_first_token\":true,\"max_seq_len\":" << cfg.maxSeqLen
                << ",\"fast_decode_requested\":" << (opts.fastDecode?"true":"false")
                << ",\"prefill_chunk\":" << effectiveChunk << ",\"backend\":\"" << app::jsonEscape(device.GetBackendName())
                << "\",\"adapter\":\"" << app::jsonEscape(device.GetName()) << "\",\"runs\":[";
        }
        bool firstDetails = true;

        fprintf(stderr, "=== Benchmark: %s ===\n", cfg.arch.c_str());
        fprintf(stderr, "%-12s %10s %10s %10s %10s %12s %8s\n",
               "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s", "fence_ms", "fence%");
        fprintf(stderr, "%-12s %10s %10s %10s %10s %12s %8s\n",
               "----------", "----------", "--------", "---------", "--------", "--------", "------");

        int genTokens = benchGenTokens > 0 ? benchGenTokens : 128;

        for (int pl : lens) {
            std::string benchmarkText = prompt;
            auto tokens = explicitTokens;
            if (tokens.empty()) {
                benchmarkText = "A";
                for (int i = 1; i < pl; ++i) benchmarkText += " A";
                tokens = session.TokenizeRaw(benchmarkText);
                if (tokens.size() != size_t(pl)) throw std::runtime_error("Default benchmark text has a different token count");
            }
            bp::BenchmarkResult r;
            double prefillTotal=0, decodeTotal=0, fenceTotal=0;
            int measuredRuns=0;
            std::vector<int32_t> previousTokens;
            for (int repetition = 0; repetition < benchRepetitions; ++repetition) {
                const auto sample = session.BenchmarkTokens(tokens, genTokens, repetition == 0 ? benchWarmups : 0);
                if (sample.generatedTokens == 0) break;
                if (repetition && sample.tokenIds != previousTokens)
                    throw std::runtime_error("Benchmark continuation changed between identical measured runs");
                previousTokens = sample.tokenIds;
                r = sample;
                prefillTotal += sample.prefillMs; decodeTotal += sample.decodeMs;
                fenceTotal += sample.fenceWaitMs; ++measuredRuns;
                if (details.is_open()) {
                    details << (firstDetails ? "" : ",") << "{\"repetition\":" << repetition
                        << ",\"prompt\":\"" << app::jsonEscape(benchmarkText) << "\",\"input_tokens\":" << pl
                        << ",\"generated_tokens\":" << sample.generatedTokens << ",\"decode_sample_tokens\":" << sample.decodeSampleTokens
                        << ",\"final_position\":" << sample.finalPosition << ",\"prefill_ms\":" << sample.prefillMs
                        << ",\"decode_ms\":" << sample.decodeMs << ",\"prefill_tok_s\":" << sample.prefillTokPerSec
                        << ",\"decode_tok_s\":" << sample.decodeTokPerSec << ",\"prompt_token_ids\":[";
                    for (size_t i=0;i<tokens.size();++i) details << (i?",":"") << tokens[i];
                    details << "],\"generated_token_ids\":[";
                    for (size_t i=0;i<sample.tokenIds.size();++i) details << (i?",":"") << sample.tokenIds[i];
                    details << "]}";details.flush();firstDetails = false;
                }
                fprintf(stderr,"  Benchmark run %d/%d: %d input, %d output, %d decode calls\n",
                    repetition+1,benchRepetitions,pl,sample.generatedTokens,sample.decodeSampleTokens);
            }
            if (static_cast<GPUContext*>(device.GetGPUContext())->deviceLost) {
                fprintf(stderr, "Benchmark failed: GPU device was lost\n");
                return 1;
            }
            if (r.prefillMs == 0 && r.decodeTokPerSec == 0) {
                fprintf(stderr, "%-12d   (skipped — exceeds maxSeqLen)\n", pl);
                continue;
            }
            r.prefillMs=prefillTotal/measuredRuns; r.decodeMs=decodeTotal/measuredRuns;
            r.prefillTokPerSec=pl*1000.0/r.prefillMs;
            r.decodeTokPerSec=genTokens>1?(genTokens-1)*1000.0/r.decodeMs:0;
            r.ttftMs=r.prefillMs; r.fenceWaitMs=fenceTotal/measuredRuns;
            benchResults.push_back({pl,r.prefillMs,r.prefillTokPerSec,r.decodeMs,r.decodeTokPerSec,r.ttftMs});
            double fencePct = r.decodeMs > 0 ? 100.0 * r.fenceWaitMs / r.decodeMs : 0;
            fprintf(stderr, "%-12d %10.1f %10.1f %10.1f %10.1f %12.1f %7.1f%%\n",
                   pl, r.prefillMs, r.prefillTokPerSec, r.decodeMs, r.decodeTokPerSec,
                   r.fenceWaitMs, fencePct);
            fflush(stderr);
        }
        if (details.is_open()) {
            details << "]}\n";
            if (!details) throw std::runtime_error("Cannot finish benchmark JSON");
        }

        if (profile) {
            fprintf(stderr, "\n=== GPU Hardware Timestamp Profile ===\n");
            fs::path modelName = fs::path(modelPath).parent_path().filename();
            if (modelName.empty())
                modelName = fs::path(modelPath).stem();
            fs::path profileDir = fs::path("gitignore") / "models" / modelName;
            fs::create_directories(profileDir);
            std::string htmlPath = (profileDir / "profile.html").string();
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
                &loadInfo, &memInfo, benchmarkProtocol.c_str(), benchWarmups);
        }

        fprintf(stderr, "\n");
        return 0;
    }

    // 4. Generate
    std::string finalPrompt;
    bool chat = !chatMessage.empty();
    bool qwenOnnxChat = false;
    bool gemma4OnnxChat = false;
    if (chat) {
        qwenOnnxChat = (cfg.format == "onnx_generic" || cfg.format == "onnx") &&
            cfg.arch.find("qwen3") != std::string::npos;
        gemma4OnnxChat = (cfg.format == "onnx_generic" || cfg.format == "onnx") &&
            cfg.arch == "gemma4";
        if (qwenOnnxChat) {
            // ORT GenAI advances Qwen's recurrent state with the system turn
            // before appending the user ChatML turn. Preserve that ordering.
            session.Reset();
            auto systemTokens = session.Tokenize("You are a helpful AI assistant.");
            if (!systemTokens.empty())
                (void)session.Prefill(systemTokens.data(), (uint32_t)systemTokens.size());
            finalPrompt = app::applyQwenUserTemplate(chatMessage);
        } else if (gemma4OnnxChat) {
            // ORT GenAI applies Gemma 4's system and user turns as separate
            // generator appends. Each append starts with the tokenizer's BOS;
            // preserving that boundary is required for matching logits.
            session.Reset();
            auto systemTokens = session.Tokenize(app::applyGemma4SystemTemplate());
            if (!systemTokens.empty())
                (void)session.Prefill(systemTokens.data(), (uint32_t)systemTokens.size());
            finalPrompt = app::applyGemma4UserTemplate(chatMessage);
        } else {
            finalPrompt = app::applyChatTemplate(chatMessage, cfg.arch);
        }
        fprintf(stderr, "Chat: %s\n", chatMessage.c_str());
    } else {
        finalPrompt = prompt;
    }

    auto promptTokens = session.Tokenize(finalPrompt);
    const uint64_t totalPromptTokens = session.GetPosition() + promptTokens.size();
    if (cfg.maxSeqLen > 0 && totalPromptTokens + (uint64_t)maxTokens > (uint64_t)cfg.maxSeqLen) {
        fprintf(stderr, "Prompt including prior context (%llu) plus generation limit (%d) exceeds context length (%lld)\n",
                (unsigned long long)totalPromptTokens, maxTokens, (long long)cfg.maxSeqLen);
        return 2;
    }
    fprintf(stderr, "Prompt: %zu tokens\n", promptTokens.size());
    if (std::getenv("BP_DUMP_TOKENS")) {
        fprintf(stderr, "[debug] prompt tokens:");
        for (int32_t token : promptTokens) fprintf(stderr, " %d", token);
        fprintf(stderr, "\n");
    }
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
        }, !(qwenOnnxChat || gemma4OnnxChat));
    if (static_cast<GPUContext*>(device.GetGPUContext())->deviceLost ||
        static_cast<GPUContext*>(device.GetGPUContext())->executionError) {
        fprintf(stderr, "Generation failed: WebGPU reported an execution error\n");
        return 1;
    }

    auto genMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - genStart).count();
    double tps = tokenCount > 0 ? tokenCount * 1000.0 / genMs : 0;

    fprintf(stderr, "\n\n--- Performance ---\n");
    fprintf(stderr, "  Prompt:   %zu tokens\n", promptTokens.size());
    fprintf(stderr, "  Generate: %d tokens in %.0fms (%.1f tok/s)\n", tokenCount, genMs, tps);
    if (profile) {
        fprintf(stderr, "\n=== GPU Hardware Timestamp Profile ===\n");
        fs::path modelName = fs::path(modelPath).parent_path().filename();
        if (modelName.empty())
            modelName = fs::path(modelPath).stem();
        fs::path profileDir = fs::path("gitignore") / "models" / modelName;
        fs::create_directories(profileDir);
        session.PrintProfileReport((profileDir / "profile.html").string());
    }
    return 0;
}

static int checkedApp(int argc, char* argv[]) {
    try { return runApp(argc, argv); }
    catch (const std::exception& error) { fprintf(stderr, "Inference failed: %s\n", error.what()); return 1; }
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
    // Preserve multilingual prompts through the Windows command line.
    SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> utf8(argc);
    std::vector<char*> pointers(argc);
    for (int i = 0; i < argc; ++i) {
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        utf8[i].resize(bytes);
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, utf8[i].data(), bytes, nullptr, nullptr);
        pointers[i] = utf8[i].data();
    }
    return checkedApp(argc, pointers.data());
}
#else
int main(int argc, char* argv[]) { return checkedApp(argc, argv); }
#endif
