/**
 * Backpack Runtime -- WebGPU inference directly from GGUF models.
 *
 * Reads model architecture and tokenizer from GGUF metadata.
 * All WGSL compute kernels are embedded in the binary.
 *
 * Usage:
 *   backpack_runtime --model model.gguf [--prompt "Hello"] [--max-tokens 50]
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
    bool benchDetail = false;
    bool usePassPerDispatch = false;
    bool benchmarkMode = false;
    int submitGroups = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "--gguf-file") && i+1 < argc)
            gguf_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backend_str = argv[++i];
        else if (arg == "--profile") profile = true;
        else if (arg == "--bench-detail") benchDetail = true;
        else if (arg == "--pass-per-dispatch") usePassPerDispatch = true;
        else if (arg == "--submit-groups" && i+1 < argc) submitGroups = atoi(argv[++i]);
        else if (arg == "--benchmark") benchmarkMode = true;
    }

    if (gguf_path.empty()) {
        fprintf(stderr,
            "Backpack Runtime -- WebGPU inference from GGUF models\n\n"
            "Usage: %s --model <model.gguf or model-dir/>\n"
            "  [--prompt <text>] [--max-tokens <n>] [--backend vulkan|d3d12]\n"
            "  [--profile] [--benchmark] [--bench-detail]\n",
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

    if (usePassPerDispatch) {
        model.passPerDispatch = true;
        printf("  Pass-per-dispatch: enabled\n");
    }

    if (submitGroups > 1) {
        model.nGroups = submitGroups;
        // Rebuild pool with new group count
        for (int s = 0; s < ModelRunner::POOL_DEPTH; s++)
            model.refillCBPool(s);
        printf("  Submit groups: %d (%zu dispatches / %d = %zu per group)\n",
               submitGroups, model.autoDecodeDispatches.size(), submitGroups,
               model.autoDecodeDispatches.size() / submitGroups);
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

    // ─── Benchmark mode ──────────────────────────────────────────────────
    if (benchmarkMode) {
        printf("\n=== Benchmark: %s ===\n", model.cfg.arch.c_str());
        printf("%-12s %10s %10s %10s %10s\n",
               "prompt_len", "prefill_ms", "pf_tok/s", "decode_ms", "dc_tok/s");
        printf("%-12s %10s %10s %10s %10s\n",
               "----------", "----------", "--------", "---------", "--------");

        int genTokens = 128;
        int promptLens[] = {5, 128, 256, 512, 1024, 2048, 4096};

        for (int pl : promptLens) {
            model.resetKVCache();

            // Prefill: use batched path (single weight read for all T tokens)
            std::vector<int32_t> dummyTokens(pl, 0);
            auto pf_t0 = std::chrono::steady_clock::now();
            auto logits = model.prefillBatched(dummyTokens.data(), (uint32_t)pl, 0);
            auto pf_t1 = std::chrono::steady_clock::now();
            auto pfMs = std::chrono::duration<double, std::milli>(pf_t1 - pf_t0).count();
            double pfTps = pl * 1000.0 / pfMs;

            // Seed the argmax buffer for autoregressive decode
            int32_t firstTok = ModelRunner::argmax(logits);
            gpu.writeBuffer(model.argmaxResultBuf, &firstTok, 4);

            // Decode: generate genTokens tokens
            auto dc_t0 = std::chrono::steady_clock::now();
            int generated = 0;
            const int DEPTH = ModelRunner::POOL_DEPTH;
            int submitted = 0, completed = 0;

            int primeCount = std::min(DEPTH, genTokens);
            for (int i = 0; i < primeCount; i++) {
                model.submitDecode((uint32_t)(pl + i), i);
                submitted++;
            }
            while (completed < submitted) {
                int slot = completed % DEPTH;
                int32_t tok = model.readArgmax(slot);
                completed++;
                generated++;
                if (submitted < genTokens) {
                    model.submitDecode((uint32_t)(pl + submitted), slot);
                    submitted++;
                }
            }
            auto dc_t1 = std::chrono::steady_clock::now();
            auto dcMs = std::chrono::duration<double, std::milli>(dc_t1 - dc_t0).count();
            double dcTps = generated * 1000.0 / dcMs;

            printf("%-12d %10.1f %10.1f %10.1f %10.1f\n",
                   pl, pfMs, pfTps, dcMs, dcTps);
        }

        printf("\n");
        gpu.destroy();
        return 0;
    }

    // 6. Tokenize prompt
    auto promptTokens = tokenizer.encode(prompt);

    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Tokens (%zu): ", promptTokens.size());
    for (auto t : promptTokens) printf("%d ", t);
    printf("\n");

    // 5. Prefill — simple sequential decode
    auto prefill_t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    for (size_t i = 0; i < promptTokens.size(); i++)
        logits = model.decode(promptTokens[i], (uint32_t)i);

    auto prefill_t1 = std::chrono::steady_clock::now();
    auto prefillMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        prefill_t1 - prefill_t0).count();
    int32_t firstToken = ModelRunner::argmax(logits);

    // Write first token to shared GPU argmax buffer for autoregressive chaining
    gpu.writeBuffer(model.argmaxResultBuf, &firstToken, 4);

    // 8. Decode loop — pipelined with POOL_DEPTH staging slots.
    //    Submit up to POOL_DEPTH tokens ahead, then read oldest + submit next.
    //    Buffers are shared (GPU executes in order); only staging differs per slot.
    printf("\n--- Output ---\n%s", prompt.c_str());
    fflush(stdout);

    auto decode_t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> generated;
    int32_t nextToken = firstToken;

    // Timing accumulators (nanoseconds)
    using hrc = std::chrono::high_resolution_clock;
    int64_t total_submit_ns = 0, total_read_ns = 0, total_print_ns = 0;
    int submit_count = 0, read_count = 0;

    // Reset GPU timing counters for decode-only measurement
    memset(&gpu.timing, 0, sizeof(gpu.timing));

    if (nextToken != tokenizer.eos_token_id && max_tokens > 0) {
        generated.push_back(nextToken);
        printf("%s", tokenizer.decode_token(nextToken).c_str());
        fflush(stdout);

        const int DEPTH = ModelRunner::POOL_DEPTH;
        int submitted = 0;  // total tokens submitted to GPU
        int completed = 0;  // total tokens read back

        // Prime: fill the pipeline with up to DEPTH tokens
        int primeCount = std::min(DEPTH, max_tokens);
        for (int i = 0; i < primeCount; i++) {
            uint32_t pos = (uint32_t)(promptTokens.size() + i);
            auto ts = hrc::now();
            model.submitDecode(pos, i);
            auto te = hrc::now();
            total_submit_ns += (te - ts).count(); submit_count++;
            submitted++;
        }

        // Steady state: read oldest result, print, submit next
        bool eos = false;
        while (completed < submitted) {
            int slot = completed % DEPTH;

            // Read result from this slot
            auto tr0 = hrc::now();
            nextToken = model.readArgmax(slot);
            auto tr1 = hrc::now();
            total_read_ns += (tr1 - tr0).count(); read_count++;
            completed++;

            if (nextToken == tokenizer.eos_token_id) { eos = true; break; }

            // Print + submit next (reusing the just-freed slot)
            auto tp0 = hrc::now();
            generated.push_back(nextToken);
            printf("%s", tokenizer.decode_token(nextToken).c_str());
            fflush(stdout);
            auto tp1 = hrc::now();
            total_print_ns += (tp1 - tp0).count();

            if (submitted < max_tokens) {
                uint32_t pos = (uint32_t)(promptTokens.size() + submitted);
                auto ts = hrc::now();
                model.submitDecode(pos, slot);
                auto te = hrc::now();
                total_submit_ns += (te - ts).count(); submit_count++;
                submitted++;
            }
        }
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

    if (benchDetail && submit_count > 0) {
        double avg_submit_us = (total_submit_ns / 1000.0) / submit_count;
        double avg_read_us = (total_read_ns / 1000.0) / read_count;
        double avg_print_us = submit_count > 2 ? (total_print_ns / 1000.0) / (submit_count - 2) : 0;
        double total_submit_ms = total_submit_ns / 1e6;
        double total_read_ms = total_read_ns / 1e6;
        double total_print_ms = total_print_ns / 1e6;
        printf("\n--- CPU Timing Detail ---\n");
        printf("  submitDecode:  %6.1fms total, %6.1fus avg (%d calls)\n",
               total_submit_ms, avg_submit_us, submit_count);
        printf("  readArgmax:    %6.1fms total, %6.1fus avg (%d calls)\n",
               total_read_ms, avg_read_us, read_count);
        printf("  printf+flush:  %6.1fms total, %6.1fus avg\n",
               total_print_ms, avg_print_us);
        printf("  wall per tok:  %6.1fus  (submit %.1f + read %.1f + print %.1f + other %.1f)\n",
               decodeMs * 1000.0 / nDecode,
               avg_submit_us, avg_read_us, avg_print_us,
               decodeMs * 1000.0 / nDecode - avg_submit_us - avg_read_us - avg_print_us);
        // GPU-internal breakdown
        auto& t = gpu.timing;
        if (t.count > 0) {
            printf("\n--- GPU API Breakdown (per call avg, %d calls) ---\n", t.count);
            printf("  cmd encode:    %6.1fus  (pre-recorded, amortized at init)\n",
                   t.encode_ns / 1000.0 / std::max(t.count, 1));
            printf("  queueSubmit:   %6.1fus  (submit pre-recorded CB)\n",
                   t.submit_ns / 1000.0 / t.count);
            printf("  mapAsync:      %6.1fus  (non-blocking initiation)\n",
                   t.map_start_ns / 1000.0 / t.count);
            printf("  waitAny:       %6.1fus  (GPU completion wait)\n",
                   t.wait_ns / 1000.0 / t.count);
            printf("  unmap+read:    %6.1fus\n", t.unmap_ns / 1000.0 / t.count);
            printf("  writeBuffer:   %6.1fms total (%d calls in decode)\n",
                   t.write_buf_ns / 1e6, t.count * 2);
        }
    }

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
