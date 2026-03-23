/**
 * Backpack LLM — End-to-end text generation using the Backpack C++ API.
 *
 * Supports all model architectures (Phi-4, Qwen, LLaMA, etc.) in both
 * GGUF and ONNX formats. Auto-detects format from the model path.
 *
 * Usage:
 *   backpack_llm --model path/to/phi-4/          --chat "What is 2+2?"
 *   backpack_llm --model path/to/model.gguf      --prompt "Hello"
 *   backpack_llm --model path/to/model            --benchmark
 *   backpack_llm --model path/to/phi-4/ --chat "Explain gravity" --max-tokens 200
 */

#include "backpack.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

// ─── Chat template ───────────────────────────────────────────────────────────

static std::string applyChatTemplate(const std::string& message,
                                      const std::string& arch) {
    // ChatML format — used by Phi, Qwen, and most instruct models
    return "<|im_start|>user\n" + message + "<|im_end|>\n<|im_start|>assistant\n";
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string modelPath;
    std::string prompt;
    std::string chatMessage;
    std::string backendStr;
    int maxTokens = 100;
    bool profile = false;
    bool benchmark = false;
    int benchPromptLen = 0;     // 0 = sweep {128..4096}
    int benchGenTokens = 128;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)           modelPath = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc)     prompt = argv[++i];
        else if (arg == "--chat" && i + 1 < argc)       chatMessage = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) maxTokens = atoi(argv[++i]);
        else if (arg == "--backend" && i + 1 < argc)    backendStr = argv[++i];
        else if (arg == "--profile")                    profile = true;
        else if (arg == "--benchmark")                  benchmark = true;
        else if (arg == "--bench-prompt-len" && i + 1 < argc) benchPromptLen = atoi(argv[++i]);
        else if (arg == "--bench-gen-tokens" && i + 1 < argc) benchGenTokens = atoi(argv[++i]);
    }

    if (modelPath.empty()) {
        fprintf(stderr,
            "Backpack LLM — WebGPU-accelerated text generation\n\n"
            "Usage: %s --model <path> [options]\n\n"
            "Model path:\n"
            "  GGUF file          --model model.gguf\n"
            "  Model directory    --model models/qwen-3/\n"
            "  ONNX directory     --model models/phi-4/\n\n"
            "Generation:\n"
            "  --prompt <text>    Raw text prompt\n"
            "  --chat <message>   Chat message (auto-wrapped in template)\n"
            "  --max-tokens <n>   Maximum tokens to generate (default: 100)\n\n"
            "Hardware:\n"
            "  --backend <name>   GPU backend: vulkan, d3d12, metal\n\n"
            "Benchmarking:\n"
            "  --benchmark        Run prefill+decode benchmark\n"
            "  --bench-prompt-len Benchmark prompt length (default: 1024)\n"
            "  --bench-gen-tokens Benchmark decode tokens (default: 128)\n"
            "  --profile          Enable GPU timestamp profiling\n",
            argv[0]);
        return 1;
    }

    // Default to "Hello" if no prompt or chat given
    if (prompt.empty() && chatMessage.empty() && !benchmark) {
        prompt = "Hello";
    }

    // ─── 1. Load model ───────────────────────────────────────────────────

    BpModelParams modelParams;
    if (backendStr == "d3d12")       modelParams.backend = BpBackend::D3D12;
    else if (backendStr == "vulkan") modelParams.backend = BpBackend::Vulkan;
    else if (backendStr == "metal")  modelParams.backend = BpBackend::Metal;

    printf("Loading model: %s\n", modelPath.c_str());
    fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();

    BpModel* model = bp_model_load(modelPath, modelParams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", modelPath.c_str());
        return 1;
    }
    fprintf(stderr, "[app] model loaded, getting info...\n"); fflush(stderr);

    auto info = bp_model_info(model);
    fprintf(stderr, "[app] info obtained: arch=%s\n", info.arch.c_str()); fflush(stderr);
    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    printf("\nModel: %s (%s, %uL, E=%u, HD=%u, V=%u)\n",
           info.arch.c_str(), info.format.c_str(),
           info.nLayer, info.nEmbd, info.headDim, info.nVocab);
    printf("GPU:   %s (%s)\n", info.gpuName.c_str(), info.backendName.c_str());

    // ─── 2. Create context ───────────────────────────────────────────────

    BpContext* ctx = bp_context_create(model);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        bp_model_free(model);
        return 1;
    }

    auto t2 = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
    printf("Ready: %lldms (load %lldms + warmup %lldms)\n\n",
           (long long)totalMs, (long long)loadMs,
           (long long)(totalMs - loadMs));

    if (profile) {
        bp_enable_profiling(ctx);
    }

    // ─── 3. Benchmark mode ───────────────────────────────────────────────

    if (benchmark) {
        printf("=== Benchmark: %s ===\n", info.arch.c_str());
        printf("%-12s %10s %10s %10s %10s\n",
               "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s");
        printf("%-12s %10s %10s %10s %10s\n",
               "----------", "----------", "--------", "---------", "--------");

        std::vector<int> promptLens;
        if (benchPromptLen > 0) {
            promptLens.push_back(benchPromptLen);
        } else {
            promptLens = {128, 256, 512, 1024, 2048, 4096};
        }

        for (int pl : promptLens) {
            auto r = bp_benchmark(ctx, pl, benchGenTokens);
            printf("%-12d %10.1f %10.1f %10.1f %10.1f\n",
                   pl, r.prefillMs, r.prefillTps, r.decodeMs, r.decodeTps);
            fflush(stdout);
        }
        printf("\n");

        if (profile) {
            bp_print_profile(ctx);
        }

        bp_context_free(ctx);
        bp_model_free(model);
        return 0;
    }

    // ─── 4. Format prompt ────────────────────────────────────────────────

    std::string finalPrompt;
    bool isChatMode = !chatMessage.empty();

    if (isChatMode) {
        finalPrompt = applyChatTemplate(chatMessage, info.arch);
        printf("Chat: %s\n", chatMessage.c_str());
    } else {
        finalPrompt = prompt;
    }

    // ─── 5. Generate ─────────────────────────────────────────────────────

    auto* tok = bp_tokenizer(model);
    auto promptTokens = bp_tokenize(tok, finalPrompt);
    printf("Prompt: %zu tokens\n", promptTokens.size());

    // Stream: echo prompt in raw mode, skip in chat mode
    printf("\n--- Output ---\n");
    if (!isChatMode) {
        printf("%s", finalPrompt.c_str());
    }
    fflush(stdout);

    auto genStart = std::chrono::steady_clock::now();
    int tokenCount = 0;

    BpGenerateParams genParams;
    genParams.maxTokens = maxTokens;

    std::string generated = bp_generate(ctx, finalPrompt, genParams,
        [&](const std::string& text) {
            printf("%s", text.c_str());
            fflush(stdout);
            tokenCount++;
            return true;
        });

    auto genEnd = std::chrono::steady_clock::now();
    auto genMs = std::chrono::duration<double, std::milli>(genEnd - genStart).count();
    double tps = (tokenCount > 0 && genMs > 0) ? tokenCount * 1000.0 / genMs : 0;

    printf("\n\n--- Performance ---\n");
    printf("  Prompt:   %zu tokens\n", promptTokens.size());
    printf("  Generate: %d tokens in %.0fms (%.1f tok/s)\n",
           tokenCount, genMs, tps);

    if (profile) {
        bp_print_profile(ctx);
    }

    // ─── 6. Cleanup ──────────────────────────────────────────────────────

    bp_context_free(ctx);
    bp_model_free(model);
    return 0;
}
