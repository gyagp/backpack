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
#include <filesystem>

namespace fs = std::filesystem;

/// Resolve model path: accepts GGUF file or directory containing GGUF
static std::string resolveModelPath(const std::string& path) {
    // Direct GGUF file
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf") {
        if (fs::exists(path)) return path;
    }

    // Directory: search for GGUF files inside
    if (fs::is_directory(path)) {
        std::string bestQ8, bestQ4, first;
        for (auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                auto name = entry.path().filename().string();
                if (first.empty()) first = entry.path().string();
                if (name.find("Q8_0") != std::string::npos)
                    bestQ8 = entry.path().string();
                else if (name.find("Q4_K") != std::string::npos)
                    bestQ4 = entry.path().string();
            }
        }
        if (!bestQ8.empty()) return bestQ8;
        if (!bestQ4.empty()) return bestQ4;
        if (!first.empty()) return first;

        fprintf(stderr, "No GGUF file found in: %s\n", path.c_str());
    }

    return path;  // return as-is, let GGUF loader report the error
}

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
            "Usage: %s --model <model.gguf or model-dir/>\n"
            "  [--prompt <text>] [--max-tokens <n>] [--backend vulkan|d3d12]\n"
            "  [--profile]\n",
            argv[0]);
        return 1;
    }

    // Resolve model path (directory -> find GGUF inside)
    gguf_path = resolveModelPath(gguf_path);

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

    // 5. Warmup: run one dummy decode to trigger shader compilation
    {
        auto tw0 = std::chrono::steady_clock::now();
        model.decode(0, 0);
        model.resetKVCache();
        auto tw1 = std::chrono::steady_clock::now();
        auto warmupMs = std::chrono::duration_cast<std::chrono::milliseconds>(tw1 - tw0).count();
        printf("Warmup: %lldms (shader compilation)\n", (long long)warmupMs);
    }

    // 6. Tokenize prompt
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
    if (profile) {
        // Derive model name from GGUF path (parent directory name)
        auto modelName = fs::path(gguf_path).parent_path().filename().string();

        // Place profile.html under gitignore/models/<model_name>/ in this repo
        // Find repo root: walk up from executable looking for .gitignore
        fs::path repoRoot;
        auto exeDir = fs::path(argv[0]).parent_path();
        if (exeDir.empty()) exeDir = fs::current_path();
        for (auto p = fs::absolute(exeDir); !p.empty() && p != p.parent_path(); p = p.parent_path()) {
            if (fs::exists(p / ".gitignore") && fs::exists(p / "runtimes")) {
                repoRoot = p;
                break;
            }
        }

        std::string profilePath;
        if (!repoRoot.empty()) {
            auto profileDir = repoRoot / "gitignore" / "models" / modelName;
            fs::create_directories(profileDir);
            profilePath = (profileDir / "profile.html").string();
        } else {
            // Fallback: current directory
            profilePath = "profile.html";
        }

        model.printProfileReport(nDecode, (int)promptTokens.size(),
                                 (double)prefillMs, (double)decodeMs,
                                 profilePath);
    }

    gpu.destroy();
    return 0;
}
