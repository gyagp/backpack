#pragma once
/**
 * backpack.h — Public C++ API for the Backpack inference runtime.
 *
 * Minimal surface, opaque internals, format-agnostic (GGUF + ONNX).
 * WebGPU-accelerated LLM inference with pipelined decode.
 *
 * Linked as a shared library (backpack.dll / libbackpack.so).
 *
 * Quick start:
 *   auto* model = bp_model_load("path/to/model");
 *   auto* ctx   = bp_context_create(model);
 *   bp_generate(ctx, "Hello", {.maxTokens = 50},
 *       [](const std::string& t) { printf("%s", t.c_str()); return true; });
 *   bp_context_free(ctx);
 *   bp_model_free(model);
 */

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// ─── DLL export/import ──────────────────────────────────────────────────────

#if defined(_WIN32)
#  if defined(BACKPACK_EXPORTS)
#    define BP_API __declspec(dllexport)
#  else
#    define BP_API __declspec(dllimport)
#  endif
#else
#  if defined(BACKPACK_EXPORTS)
#    define BP_API __attribute__((visibility("default")))
#  else
#    define BP_API
#  endif
#endif

// ─── Forward declarations (opaque) ──────────────────────────────────────────

struct BpModel;
struct BpContext;
struct BpTokenizer;

// ─── Enums ──────────────────────────────────────────────────────────────────

enum class BpBackend { Vulkan, D3D12, Metal, Default };

// ─── Configuration ──────────────────────────────────────────────────────────

/// Parameters for model loading (GPU init + weight upload).
struct BpModelParams {
    BpBackend backend     = BpBackend::Default;  // GPU backend selection
    uint32_t  maxSeqLen   = 4096;                // Maximum sequence length
};

/// Parameters for context creation (KV cache + pipeline tuning).
struct BpContextParams {
    bool autoWarmup   = true;   // Warmup shader compilation on create
    bool autoAutotune = true;   // Auto-tune decode kernel selection
};

/// Parameters for text generation.
struct BpGenerateParams {
    int   maxTokens   = 100;    // Maximum tokens to generate
    float temperature = 0.0f;   // 0 = greedy (argmax)
};

// ─── Model info ─────────────────────────────────────────────────────────────

/// Read-only metadata about a loaded model.
struct BpModelInfo {
    std::string arch;           // e.g. "phi3", "qwen3", "llama"
    std::string format;         // "gguf" or "onnx"
    uint32_t nLayer;
    uint32_t nHead;
    uint32_t nKvHeads;
    uint32_t nEmbd;
    uint32_t headDim;
    uint32_t nVocab;
    uint32_t intermediateSize;
    float    ropeTheta;
    std::string gpuName;        // GPU adapter name
    std::string backendName;    // "vulkan", "d3d12", "metal"
};

// ─── Model ──────────────────────────────────────────────────────────────────
//
// Immutable after creation. Owns GPU device, compiled pipelines, and weights.
// Auto-detects model format (GGUF file, GGUF directory, or ONNX directory).

/// Load a model from a path. Accepts:
///   - GGUF file path       ("model.gguf")
///   - Directory with GGUF  ("models/qwen-3/")
///   - ONNX model dir       ("models/phi-4/")  (must contain model.onnx)
/// Returns nullptr on failure.
BP_API BpModel*     bp_model_load(const std::string& path,
                                   const BpModelParams& params = {});
BP_API void         bp_model_free(BpModel* model);
BP_API BpModelInfo  bp_model_info(const BpModel* model);

// ─── Tokenizer ──────────────────────────────────────────────────────────────
//
// Loaded with the model. Supports GGUF metadata and HuggingFace tokenizer.json.

/// Get the tokenizer (borrowed pointer — lifetime tied to model, do not free).
BP_API BpTokenizer*          bp_tokenizer(const BpModel* model);
BP_API std::vector<int32_t>  bp_tokenize(const BpTokenizer* tok,
                                          const std::string& text);
BP_API std::string           bp_token_to_text(const BpTokenizer* tok,
                                               int32_t token);
BP_API std::string           bp_detokenize(const BpTokenizer* tok,
                                            const std::vector<int32_t>& tokens);
BP_API int32_t               bp_token_eos(const BpTokenizer* tok);
BP_API int32_t               bp_token_bos(const BpTokenizer* tok);

// ─── Context ────────────────────────────────────────────────────────────────
//
// Mutable inference state: KV cache + pipelined decode pipeline.

BP_API BpContext*  bp_context_create(BpModel* model,
                                      const BpContextParams& params = {});
BP_API void        bp_context_free(BpContext* ctx);

/// Clear KV cache and reset sequence position to 0.
BP_API void        bp_context_reset(BpContext* ctx);

/// Current sequence length (number of tokens in KV cache).
BP_API uint32_t    bp_context_pos(const BpContext* ctx);

// ─── Low-level inference ────────────────────────────────────────────────────
//
// Caller controls the decode loop. Maximum flexibility.

/// Prefill: process prompt tokens in parallel (batched matmul).
/// Populates KV cache for positions [0, nTokens).
/// Returns the argmax token ID for the next position.
BP_API int32_t  bp_prefill(BpContext* ctx, const int32_t* tokens,
                            uint32_t nTokens);

/// Decode: generate one token using pipelined GPU execution.
/// Reads the previous argmax result and submits the next decode step.
/// Returns the next token ID (argmax).
BP_API int32_t  bp_decode(BpContext* ctx);

// ─── High-level generation ──────────────────────────────────────────────────
//
// Simple generate call with optional streaming. Good for applications.

/// Streaming callback: receives each decoded text piece.
/// Return true to continue, false to stop generation early.
using BpStreamCallback = std::function<bool(const std::string& text)>;

/// Generate text from a prompt. Handles tokenization, prefill, and decode.
/// Returns the full generated text (excluding the prompt).
/// If onToken is provided, each token is streamed to the callback.
BP_API std::string  bp_generate(BpContext* ctx,
                                 const std::string& prompt,
                                 const BpGenerateParams& params = {},
                                 BpStreamCallback onToken = nullptr);

// ─── Profiling ──────────────────────────────────────────────────────────────

BP_API void  bp_enable_profiling(BpContext* ctx);
BP_API void  bp_print_profile(BpContext* ctx,
                               const std::string& outputPath = "");

/// Benchmark result for a single prompt length.
struct BpBenchResult {
    double prefillMs;
    double prefillTps;
    double decodeMs;
    double decodeTps;
    int    nPrefillTokens;
    int    nDecodeTokens;
};

/// Run a benchmark: prefill + decode at the given prompt length.
BP_API BpBenchResult bp_benchmark(BpContext* ctx, int promptLen = 1024,
                                   int genTokens = 128);
