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
    const char* env = std::getenv("NEMOTRON_GGUF_PATH");
    if (env && std::filesystem::exists(env)) return env;

    std::vector<std::string> candidates = {
        "models/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf",
        "models/nemotron-mini-4b-instruct.Q4_K_M.gguf",
        "models/nemotron-mini-4b-instruct-Q4_K_M.gguf",
        "../models/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf",
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
        std::cout << "SKIP: Nemotron-Mini-4B-Instruct GGUF not found. Set NEMOTRON_GGUF_PATH or place in models/." << std::endl;
        return 0;
    }

    {
        std::cout << "Test: Nemotron-Mini-4B-Instruct 16-token generation (D3D12)... ";

        GenerateParams params;
        params.max_tokens = 16;
        params.temperature = 0.0f;

        auto t0 = std::chrono::steady_clock::now();
        GenerateResult result = generate(model_path, "Hello", params);
        auto t1 = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        assert(elapsed < 60.0);

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
