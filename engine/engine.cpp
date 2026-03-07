/**
 * Stage 2: C++ Execution Engine for Backpack models.
 *
 * Loads a compiled model bundle (manifest.json + WGSL kernels) and
 * GGUF weights, then runs inference using Dawn's native WebGPU C API.
 *
 * Architecture:
 *   gpu_context.{h,cpp}  — Common WebGPU runtime (model-agnostic)
 *   json_parser.{h,cpp}  — JSON manifest parser
 *   gguf_loader.{h,cpp}  — GGUF weight loader
 *   model_runner.{h,cpp} — Manifest-driven inference engine
 *   engine.cpp            — CLI entry point
 *
 * Usage:
 *   backpack_engine \
 *     --bundle build/qwen3-1.7B \
 *     --gguf-file weights/Qwen3-1.7B-Q8_0.gguf \
 *     --prompt "Hello world" \
 *     --max-tokens 100
 */

#include "gpu_context.h"
#include "model_runner.h"

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    std::string bundle_dir, gguf_path, prompt = "Hello world";
    int max_tokens = 50;
    std::string backend_str = "vulkan";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bundle" && i+1 < argc) bundle_dir = argv[++i];
        else if (arg == "--gguf-file" && i+1 < argc) gguf_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backend_str = argv[++i];
    }

    if (bundle_dir.empty() || gguf_path.empty()) {
        fprintf(stderr,
            "Backpack Engine — WebGPU inference from compiled WGSL bundles\n\n"
            "Usage: %s --bundle <dir> --gguf-file <path>\n"
            "  [--prompt <text>] [--max-tokens <n>] [--backend vulkan|d3d12]\n",
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

    // 2. Load model bundle + weights
    ModelRunner model;
    auto t0 = std::chrono::steady_clock::now();
    if (!model.load(gpu, bundle_dir, gguf_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("Model loaded in %lldms\n\n", (long long)loadMs);

    // 3. Tokenize prompt (simple: use raw byte IDs for now, or dummy)
    // For a real implementation, integrate a tokenizer library.
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n\n", max_tokens);

    // Simple tokenization: process each prompt token through prefill
    // For now, use token ID 9707 ("Hello") as a simple test
    std::vector<int32_t> promptTokens;
    // Hardcoded for "Hello world" — in production, use a tokenizer
    if (prompt == "Hello") {
        promptTokens = {9707};
    } else if (prompt == "Hello world") {
        promptTokens = {9707, 1879};
    } else {
        // Default: just use "The" (token 785)
        promptTokens = {785};
    }

    printf("Prompt tokens (%zu): ", promptTokens.size());
    for (auto t : promptTokens) printf("%d ", t);
    printf("\n");

    // 4. Prefill: process prompt tokens one at a time
    // (simplified — real prefill would use multi-token GPU path)
    printf("Prefilling %zu tokens...\n", promptTokens.size());
    auto prefill_t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    for (size_t i = 0; i < promptTokens.size(); i++) {
        logits = model.decode(promptTokens[i], (uint32_t)i);
    }
    auto prefill_t1 = std::chrono::steady_clock::now();
    auto prefillMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        prefill_t1 - prefill_t0).count();
    int32_t firstToken = ModelRunner::argmax(logits);
    printf("Prefill: %lldms, first token: %d\n", (long long)prefillMs, firstToken);

    // 5. Decode loop
    printf("\nGenerating %d tokens...\n", max_tokens);
    auto decode_t0 = std::chrono::steady_clock::now();

    std::vector<int32_t> generated;
    int32_t nextToken = firstToken;
    generated.push_back(nextToken);

    for (int step = 1; step < max_tokens; step++) {
        logits = model.decode(nextToken,
                              (uint32_t)(promptTokens.size() + step - 1));
        nextToken = ModelRunner::argmax(logits);
        generated.push_back(nextToken);
    }

    auto decode_t1 = std::chrono::steady_clock::now();
    auto decodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_t1 - decode_t0).count();

    // 6. Print results
    printf("\nGenerated token IDs: ");
    for (auto t : generated) printf("%d ", t);
    printf("\n");

    int decodeTokens = max_tokens - 1;
    double decodeTps = decodeMs > 0 ? decodeTokens * 1000.0 / decodeMs : 0;
    printf("\n--- Performance ---\n");
    printf("  Prefill: %lldms (%zu tokens)\n", (long long)prefillMs,
           promptTokens.size());
    printf("  Decode:  %d tokens in %lldms (%.1f tok/s)\n",
           decodeTokens, (long long)decodeMs, decodeTps);

    gpu.destroy();
    return 0;
}

