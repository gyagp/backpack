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

    // 3. TODO: Tokenize prompt
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n\n", max_tokens);

    // 4. Generate
    printf("[TODO] Full inference loop — weight loading verified OK.\n");
    printf("Next steps:\n");
    printf("  1. Integrate tokenizer (sentencepiece/tiktoken)\n");
    printf("  2. Implement prefill dispatch sequence\n");
    printf("  3. Implement pre-recorded fast decode loop\n");
    printf("  4. Measure decode speed (target: >300 tok/s)\n");

    gpu.destroy();
    return 0;
}

