/**
 * Backpack Engine -- WebGPU inference directly from GGUF models.
 *
 * Reads model architecture and tokenizer from GGUF metadata.
 * All WGSL compute kernels are embedded in the binary.
 *
 * Usage:
 *   backpack_engine --model model.gguf [--prompt "Hello"] [--max-tokens 50]
 */

#include "gpu_context.h"
#include "model_runner.h"
#include "tokenizer.h"

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    std::string gguf_path, prompt = "Hello";
    int max_tokens = 50;
    std::string backend_str = "vulkan";
    bool profile = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "--gguf-file") && i+1 < argc)
            gguf_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backend_str = argv[++i];
        else if (arg == "--profile") profile = true;
    }

    if (gguf_path.empty()) {
        fprintf(stderr,
            "Backpack Engine -- WebGPU inference from GGUF models\n\n"
            "Usage: %s --model <path.gguf>\n"
            "  [--prompt <text>] [--max-tokens <n>] [--backend vulkan|d3d12]\n"
            "  [--profile]\n",
            argv[0]);
        return 1;
    }

    WGPUBackendType backend = WGPUBackendType_Vulkan;
    if (backend_str == "d3d12") backend = WGPUBackendType_D3D12;
    else if (backend_str == "metal") backend = WGPUBackendType_Metal;

    // 1. Initialize GPU
    GPUContext gpu;
    if (!gpu.init(backend)) {
        fprintf(stderr, "Failed to initialize GPU\n");
        return 1;
    }

    // 2. Load model
    ModelRunner model;
    auto t0 = std::chrono::steady_clock::now();
    if (!model.load(gpu, gguf_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // 3. Load tokenizer from same GGUF
    Tokenizer tokenizer;
    if (!tokenizer.load(model.gguf)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // 4. Enable profiling if requested
    if (profile) {
        model.enableProfiling();
        if (model.profiler)
            printf("  GPU profiling enabled (timestamp queries)\n");
    }

    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("Model loaded in %lldms\n\n", (long long)loadMs);

    // 4. Tokenize prompt
    auto promptTokens = tokenizer.encode(prompt);

    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Tokens (%zu): ", promptTokens.size());
    for (auto t : promptTokens) printf("%d ", t);
    printf("\n");

    // 5. Prefill
    auto prefill_t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    for (size_t i = 0; i < promptTokens.size(); i++)
        logits = model.decode(promptTokens[i], (uint32_t)i);

    auto prefill_t1 = std::chrono::steady_clock::now();
    auto prefillMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        prefill_t1 - prefill_t0).count();
    int32_t firstToken = ModelRunner::argmax(logits);

    // 6. Decode loop with text output
    printf("\n--- Output ---\n%s", prompt.c_str());
    fflush(stdout);

    auto decode_t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> generated;
    int32_t nextToken = firstToken;

    for (int step = 0; step < max_tokens; step++) {
        // Check for EOS
        if (nextToken == tokenizer.eos_token_id) break;

        generated.push_back(nextToken);

        // Print token text immediately (streaming)
        std::string text = tokenizer.decode_token(nextToken);
        printf("%s", text.c_str());
        fflush(stdout);

        // Generate next token (GPU argmax — reads back only 4 bytes)
        uint32_t pos = (uint32_t)(promptTokens.size() + step);
        nextToken = model.decodeArgmax(nextToken, pos);
    }

    auto decode_t1 = std::chrono::steady_clock::now();
    auto decodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_t1 - decode_t0).count();

    printf("\n\n--- Performance ---\n");
    printf("  Prefill: %lldms (%zu tokens)\n", (long long)prefillMs,
           promptTokens.size());
    int nDecode = (int)generated.size();
    double tps = decodeMs > 0 ? nDecode * 1000.0 / decodeMs : 0;
    printf("  Decode:  %d tokens in %lldms (%.1f tok/s)\n",
           nDecode, (long long)decodeMs, tps);

    // Print GPU profile report if profiling was enabled
    if (profile)
        model.printProfileReport();

    gpu.destroy();
    return 0;
}
