#pragma once
/**
 * lm_session.h — Layer 2: High-level Language Model API.
 *
 * Provides bp::LmSession for text generation on top of the general-purpose
 * Layer 1 API (bp::Device, bp::Model, bp::Session, bp::Tensor).
 *
 * LmSession abstracts over both backends:
 *   - Standard transformers (GGUF or ONNX via ModelRunner)
 *   - Generic ONNX architectures (LFM2 conv+MoE via GraphExecutor)
 *
 * All GPU buffer management, KV cache, fast decode capture/replay, and
 * tokenization are handled internally. The caller only sees strings,
 * token IDs, and logits.
 */

#include "backpack.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bp {

// ─── Supporting Types ───────────────────────────────────────────────────────

/// Options for LmSession creation.
struct LmOptions {
    bool fastDecode = true;       ///< Enable fast decode (capture/replay) when supported
    int64_t maxSeqLen = 0;        ///< Max context length (0 = auto from GPU memory)
    bool warmupPipelines = true;  ///< Pre-compile GPU pipelines at load time
};

/// Read-only model metadata (populated at Create time).
struct LmConfig {
    std::string arch;             ///< Architecture name ("qwen3", "phi4", "lfm2", etc.)
    std::string format;           ///< Format string ("gguf", "onnx", "onnx_generic")
    int layers = 0;
    int hiddenSize = 0;
    int vocabSize = 0;
    int numHeads = 0;
    int numKvHeads = 0;
    int headDim = 0;
    int64_t maxSeqLen = 0;        ///< Effective context length
};

/// Sampling parameters for Generate().
struct SamplingParams {
    float temperature = 0.0f;     ///< 0 = greedy (argmax)
    int topK = 0;                 ///< 0 = disabled
    uint64_t seed = 0;            ///< 0 = random
};

/// Result from a single benchmark run at one prompt length.
struct BenchmarkResult {
    int promptLen = 0;
    double prefillMs = 0;
    double prefillTokPerSec = 0;
    double decodeMs = 0;
    double decodeTokPerSec = 0;
    double ttftMs = 0;            ///< Time to first token
};

/// Callback for streaming token output. Return false to stop generation.
using StreamCallback = std::function<bool(const std::string& token)>;

// ─── LmSession ──────────────────────────────────────────────────────────────

class BP_EXPORT LmSession {
public:
    struct Impl;

    // ─── Factory ────────────────────────────────────────────────────────

    /// Create an LmSession from a model path (auto-detect format).
    static LmSession Create(Device& device, const std::string& modelPath,
                            const LmOptions& options = {});

    /// Create with explicit format override ("gguf" or "onnx").
    static LmSession Create(Device& device, const std::string& modelPath,
                            const std::string& format,
                            const LmOptions& options = {});

    // ─── Metadata ───────────────────────────────────────────────────────

    /// Model metadata (populated at creation).
    LmConfig GetConfig() const;

    // ─── Tokenizer ──────────────────────────────────────────────────────

    std::vector<int32_t> Tokenize(const std::string& text) const;
    std::string Detokenize(int32_t tokenId) const;
    std::string Detokenize(const std::vector<int32_t>& tokenIds) const;
    int32_t GetEosTokenId() const;

    // ─── High-level Generation ──────────────────────────────────────────

    /// Generate text from a string prompt. Handles tokenize, prefill, decode.
    std::string Generate(const std::string& prompt, int maxTokens,
                         const SamplingParams& sampling = {},
                         StreamCallback onToken = nullptr);

    // ─── Low-level Stepping ─────────────────────────────────────────────

    /// Prefill prompt tokens, return predicted next token ID.
    int32_t Prefill(const int32_t* tokens, uint32_t count);

    /// Decode one step: return next token (greedy argmax).
    int32_t Decode();

    /// Decode one step: return raw logits (vocabSize floats).
    std::vector<float> DecodeLogits();

    /// Reset KV cache / internal state for a new conversation.
    void Reset();

    /// Current position (number of tokens processed).
    uint32_t GetPosition() const;

    // ─── Benchmarking + Profiling ───────────────────────────────────────

    BenchmarkResult Benchmark(int promptLen, int genTokens);
    void EnableProfiling();
    void PrintProfileReport(const std::string& htmlPath = "profile.html");

    // ─── Lifecycle ──────────────────────────────────────────────────────

    void Release();
    LmSession();
    ~LmSession();
    LmSession(LmSession&& o) noexcept;
    LmSession& operator=(LmSession&& o) noexcept;
    LmSession(const LmSession&) = delete;
    LmSession& operator=(const LmSession&) = delete;

    Impl* GetImpl() const { return impl_.get(); }
    bool IsValid() const { return impl_ != nullptr; }

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace bp
