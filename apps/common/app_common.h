#pragma once
/**
 * app_common.h -- Shared utilities for Backpack C++ applications.
 *
 * Header-only. Provides system info collection, JSON helpers,
 * device creation, chat templates, and other common functionality
 * shared between the LLM and Image apps.
 */

#include "backpack.h"
#include "gpu_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <fstream>
#endif

namespace app {

// ─── Backend parsing ────────────────────────────────────────────────────────

inline bp::Backend parseBackend(const std::string& str) {
    if (str == "d3d12")  return bp::Backend::D3D12;
    if (str == "vulkan") return bp::Backend::Vulkan;
    if (str == "metal")  return bp::Backend::Metal;
    return bp::Backend::Default;
}

// ─── Device creation ────────────────────────────────────────────────────────

inline bp::Device createDevice(const std::string& backendStr) {
    return bp::Device::Create(parseBackend(backendStr));
}

// ─── GPU info ───────────────────────────────────────────────────────────────

inline void printGpuInfo(const bp::Device& device) {
    printf("GPU: %s (%s)\n", device.GetName().c_str(),
           device.GetBackendName().c_str());
}

// ─── Chat templates ─────────────────────────────────────────────────────────

inline std::string applyChatTemplate(const std::string& message,
                                     const std::string& arch) {
    // Qwen3 (ONNX reports "qwen3", GGUF reports "qwen2" since llama.cpp
    // treats Qwen3 as architecturally identical to Qwen2).
    // Empty <think></think> block disables thinking mode.
    if (arch.find("qwen3") != std::string::npos ||
        arch.find("qwen2") != std::string::npos)
        return "<|im_start|>user\n" + message + "<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n</think>\n";
    // Phi-3/Phi-4-mini: uses <|user|>/<|end|>/<|assistant|> tokens
    if (arch.find("phi3") != std::string::npos ||
        arch.find("phi4") != std::string::npos)
        return "<|user|>" + message + "<|end|><|assistant|>";
    if (arch.find("lfm2") != std::string::npos)
        return "<|startoftext|><|im_start|>user\n" + message +
               "<|im_end|>\n<|im_start|>assistant\n";
    return "<|im_start|>user\n" + message +
           "<|im_end|>\n<|im_start|>assistant\n";
}

// ─── fp16 conversion ────────────────────────────────────────────────────────

inline float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

// ─── System info ────────────────────────────────────────────────────────────

struct SystemInfo {
    std::string cpu;
    uint64_t memoryGB;
    std::string os;
};

inline SystemInfo getSystemInfo() {
    SystemInfo info;
#ifdef _WIN32
    info.os = "Windows";

    // CPU name from registry
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buf[256] = {};
        DWORD size = sizeof(buf);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                (LPBYTE)buf, &size) == ERROR_SUCCESS) {
            info.cpu = buf;
            // Trim leading/trailing spaces
            while (!info.cpu.empty() && info.cpu.front() == ' ')
                info.cpu.erase(info.cpu.begin());
            while (!info.cpu.empty() && info.cpu.back() == ' ')
                info.cpu.pop_back();
        }
        RegCloseKey(hKey);
    }
    if (info.cpu.empty()) info.cpu = "Unknown CPU";

    // Total RAM
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem))
        info.memoryGB = mem.ullTotalPhys / (1024ULL * 1024 * 1024);
    else
        info.memoryGB = 0;

#elif defined(__APPLE__)
    info.os = "macOS";

    char cpuBuf[256] = {};
    size_t cpuLen = sizeof(cpuBuf);
    if (sysctlbyname("machdep.cpu.brand_string", cpuBuf, &cpuLen, nullptr, 0) == 0)
        info.cpu = cpuBuf;
    else
        info.cpu = "Unknown CPU";

    uint64_t memBytes = 0;
    size_t memLen = sizeof(memBytes);
    if (sysctlbyname("hw.memsize", &memBytes, &memLen, nullptr, 0) == 0)
        info.memoryGB = memBytes / (1024ULL * 1024 * 1024);
    else
        info.memoryGB = 0;

#else
    info.os = "Linux";
    info.cpu = "Unknown CPU";
    info.memoryGB = 0;

    // Parse /proc/cpuinfo
    std::ifstream cpuFile("/proc/cpuinfo");
    if (cpuFile.is_open()) {
        std::string line;
        while (std::getline(cpuFile, line)) {
            if (line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    info.cpu = line.substr(pos + 2);
                    break;
                }
            }
        }
    }

    // Parse /proc/meminfo
    std::ifstream memFile("/proc/meminfo");
    if (memFile.is_open()) {
        std::string line;
        while (std::getline(memFile, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                uint64_t kb = 0;
                sscanf(line.c_str(), "MemTotal: %llu kB",
                       (unsigned long long*)&kb);
                info.memoryGB = kb / (1024 * 1024);
                break;
            }
        }
    }
#endif
    return info;
}

// ─── JSON helpers ───────────────────────────────────────────────────────────

inline std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

inline std::string isoTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm lt;
#ifdef _WIN32
    localtime_s(&lt, &t);
#else
    localtime_r(&t, &lt);
#endif
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &lt);
    return buf;
}

// ─── Baseline JSON writing ──────────────────────────────────────────────────

struct BenchResultEntry {
    int inputTokens;
    double prefillMs, prefillTokS;
    double decodeMs, decodeTokS;
};

inline bool writeBaselineJson(
    const std::string& path,
    const SystemInfo& sys,
    const std::string& gpuName,
    const std::string& gpuBackend,
    const std::string& gpuDriver,
    const std::string& modelName,
    const std::string& modelPath,
    const std::string& modelFormat,
    int layers, int hiddenSize, int vocabSize,
    int decodeTokens,
    const std::vector<BenchResultEntry>& results)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Error: could not write baseline to %s\n", path.c_str());
        return false;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"system\": {\n");
    fprintf(f, "    \"cpu\": \"%s\",\n", jsonEscape(sys.cpu).c_str());
    fprintf(f, "    \"memory_gb\": %llu,\n", (unsigned long long)sys.memoryGB);
    fprintf(f, "    \"os\": \"%s\"\n", jsonEscape(sys.os).c_str());
    fprintf(f, "  },\n");

    fprintf(f, "  \"gpu\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", jsonEscape(gpuName).c_str());
    fprintf(f, "    \"backend\": \"%s\",\n", jsonEscape(gpuBackend).c_str());
    fprintf(f, "    \"driver\": \"%s\"\n", jsonEscape(gpuDriver).c_str());
    fprintf(f, "  },\n");

    fprintf(f, "  \"model\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", jsonEscape(modelName).c_str());
    fprintf(f, "    \"path\": \"%s\",\n", jsonEscape(modelPath).c_str());
    fprintf(f, "    \"format\": \"%s\",\n", jsonEscape(modelFormat).c_str());
    fprintf(f, "    \"layers\": %d,\n", layers);
    fprintf(f, "    \"hidden_size\": %d,\n", hiddenSize);
    fprintf(f, "    \"vocab_size\": %d\n", vocabSize);
    fprintf(f, "  },\n");

    fprintf(f, "  \"benchmark\": {\n");
    fprintf(f, "    \"decode_tokens\": %d,\n", decodeTokens);
    fprintf(f, "    \"results\": [\n");
    for (size_t i = 0; i < results.size(); i++) {
        auto& r = results[i];
        fprintf(f, "      {\n");
        fprintf(f, "        \"input_tokens\": %d,\n", r.inputTokens);
        fprintf(f, "        \"prefill_ms\": %.1f,\n", r.prefillMs);
        fprintf(f, "        \"prefill_tok_s\": %.1f,\n", r.prefillTokS);
        fprintf(f, "        \"decode_ms\": %.1f,\n", r.decodeMs);
        fprintf(f, "        \"decode_tok_s\": %.1f\n", r.decodeTokS);
        fprintf(f, "      }%s\n", (i + 1 < results.size()) ? "," : "");
    }
    fprintf(f, "    ]\n");
    fprintf(f, "  },\n");

    fprintf(f, "  \"timestamp\": \"%s\"\n", isoTimestamp().c_str());
    fprintf(f, "}\n");

    fclose(f);
    printf("Baseline saved: %s\n", path.c_str());
    return true;
}

// ─── Model discovery ────────────────────────────────────────────────────────

/// Resolve a model name using the BACKPACK_MODELS environment variable.
/// If `name` is already a valid path, returns it unchanged.
/// Otherwise, checks BACKPACK_MODELS/<name>/ for model files.
inline std::string discoverModelPath(const std::string& name) {
    namespace fs = std::filesystem;

    // Already a valid path (file or directory)?
    if (fs::exists(name)) return name;

    // Try BACKPACK_MODELS env var
    const char* modelsDir = std::getenv("BACKPACK_MODELS");
    if (modelsDir) {
        auto candidate = fs::path(modelsDir) / name;
        if (fs::exists(candidate)) return candidate.string();
    }

    // Try common relative paths
    for (const char* prefix : {"models/", "../ai-models/"}) {
        auto candidate = fs::path(prefix) / name;
        if (fs::exists(candidate)) return candidate.string();
    }

    return name;  // Return as-is, let downstream error handling catch it
}

// ─── Sampling ──────────────────────────────────────────────────────────────

struct SamplingParams {
    float temperature = 0.0f;  // 0 = greedy (argmax)
    int top_k = 0;             // 0 = disabled
    uint64_t seed = 0;         // 0 = random seed
};

inline int32_t sampleToken(const float* logits, uint32_t vocabSize,
                           const SamplingParams& p, std::mt19937& rng) {
    // Greedy (argmax)
    if (p.temperature <= 0.0f) {
        return (int32_t)(std::max_element(logits, logits + vocabSize) - logits);
    }

    if (p.top_k > 0 && p.top_k < (int)vocabSize) {
        // Top-k sampling: partial sort to find k largest
        int k = p.top_k;
        std::vector<int32_t> indices(vocabSize);
        for (uint32_t i = 0; i < vocabSize; i++) indices[i] = (int32_t)i;
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });

        // Temperature-scaled softmax over top-k
        std::vector<float> probs(k);
        float maxVal = logits[indices[0]];
        for (int i = 0; i < k; i++)
            probs[i] = logits[indices[i]] / p.temperature - maxVal / p.temperature;
        // Stable softmax
        float sum = 0.0f;
        for (int i = 0; i < k; i++) { probs[i] = std::exp(probs[i]); sum += probs[i]; }
        for (int i = 0; i < k; i++) probs[i] /= sum;

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return indices[dist(rng)];
    }

    // Full-vocabulary temperature sampling
    std::vector<float> probs(vocabSize);
    float maxVal = *std::max_element(logits, logits + vocabSize);
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocabSize; i++) {
        probs[i] = std::exp((logits[i] - maxVal) / p.temperature);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < vocabSize; i++) probs[i] /= sum;

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

}  // namespace app
