#include "../src/inference.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>

static std::string find_model() {
    const char* env = std::getenv("SMOLLM_GGUF_PATH");
    if (env && std::filesystem::exists(env)) return env;

    std::vector<std::string> candidates = {
        "models/SmolLM-135M-Instruct.Q4_K_M.gguf",
        "models/smollm-135m-instruct.Q4_K_M.gguf",
        "models/SmolLM-135M-Instruct-Q4_K_M.gguf",
        "../models/SmolLM-135M-Instruct.Q4_K_M.gguf",
    };
    for (auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    return "";
}

int main() {
    int passed = 0, failed = 0;

    std::string model_path = find_model();
    if (model_path.empty()) {
        std::cout << "SKIP: SmolLM-135M GGUF not found. Set SMOLLM_GGUF_PATH or place in models/." << std::endl;
        return 0;
    }

    // AC1+AC2: Load model and generate 16 tokens on Vulkan
    {
        std::cout << "Test: SmolLM-135M 16-token generation (Vulkan)... ";

        GenerateParams params;
        params.max_tokens = 16;
        params.temperature = 0.0f;
        params.backend = GpuBackend::Vulkan;

        auto t0 = std::chrono::steady_clock::now();
        GenerateResult result = generate(model_path, "Hello", params);
        auto t1 = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // AC4: Must complete in < 30s
        assert(elapsed < 30.0);

        // AC3: Output tokens are valid (not all zeros/repeating)
        assert(!result.tokens.empty());

        bool all_zero = std::all_of(result.tokens.begin(), result.tokens.end(),
                                     [](uint32_t t) { return t == 0; });
        assert(!all_zero);

        if (result.tokens.size() >= 4) {
            std::set<uint32_t> unique(result.tokens.begin(), result.tokens.end());
            assert(unique.size() > 1);
        }

        std::cout << "PASSED (" << result.tokens.size() << " tokens, "
                  << elapsed << "s)" << std::endl;
        passed++;
    }

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed > 0 ? 1 : 0;
}
