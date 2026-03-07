#pragma once
/**
 * model_runner.h — Manifest-driven model inference engine.
 *
 * Model-agnostic: reads the decode plan from manifest.json and
 * executes it as a sequence of GPU dispatches. No model-specific
 * C++ code is needed — all model structure is captured in the manifest.
 *
 * Supports any model that compile_model.py can export.
 */

#include "gpu_context.h"
#include "json_parser.h"
#include "gguf_loader.h"

#include <string>
#include <unordered_map>
#include <vector>

/// Per-layer pre-built dispatch sequence for fast decode.
struct LayerDispatches {
    std::vector<Dispatch> dispatches;
};

/// Complete model loaded and ready for inference.
struct ModelRunner {
    GPUContext* gpu = nullptr;
    JsonValue   manifest;
    std::string bundleDir;

    // Model config (from manifest)
    uint32_t nLayer = 0, nHead = 0, nKvHeads = 0, nEmbd = 0;
    uint32_t intermediateSize = 0, nVocab = 0, headDim = 0;
    bool tieWordEmbeddings = true;
    float rmsNormEps = 1e-6f;
    float ropeTheta = 1e6f;

    // Intermediate buffers
    GPUBuffer xBuf, normOutBuf, qkvBuf, qRotBuf, attnOutBuf;
    GPUBuffer projOutBuf, gateUpBuf, siluOutBuf, rstdBuf, logitsBuf;
    GPUBuffer attnPartialsBuf;

    // KV cache: [layer] → (K_buf, V_buf, cached_len)
    struct KVEntry { GPUBuffer K, V; uint32_t len = 0; };
    std::vector<KVEntry> kvCache;

    // Per-layer weight buffers
    struct LayerWeights {
        GPUBuffer qkvW, qkvS;       // QKV projection Q8 weights + scales
        GPUBuffer oW, oS;           // O projection
        GPUBuffer guW, guS;         // Gate+Up projection
        GPUBuffer dnW, dnS;         // Down projection
        GPUBuffer inputNorm;        // Input layernorm weight
        GPUBuffer postAttnNorm;     // Post-attention layernorm weight
        GPUBuffer qNorm, kNorm;     // QK norm weights
    };
    std::vector<LayerWeights> layerWeights;
    GPUBuffer finalNormW;
    GPUBuffer lmHeadW;              // fp16 LM head weight
    GPUBuffer embeddingW;           // fp32 embedding weight
    GPUBuffer zeroBiasE, zeroBiasQKV, zeroBiasGU, zeroBiasV;

    // Params buffers (static per-shape)
    std::unordered_map<std::string, GPUBuffer> paramsBufs;

    // Pre-built decode pipeline: all dispatches for one token
    std::vector<Dispatch> allDecodeDispatches;
    // Logits readback (persistently mapped if possible)
    GPUBuffer readbackBuf;
    void*     readbackPtr = nullptr;  // mapped pointer or nullptr

    // Embedding table (CPU-side for token lookup)
    std::vector<float> embeddingCPU;

    // RoPE tables
    GPUBuffer ropeCosBuf, ropeSinBuf;

    // Dynamic params buffers (written per-token via writeBuffer)
    GPUBuffer fusedRopeParamsBuf;
    GPUBuffer chunkedAttnParamsBuf;

    // Static param layout for fused rope
    std::vector<uint8_t> ropeParamData;
    std::vector<uint8_t> chunkedAttnParamData;

    // Derived dimensions
    uint32_t qDim = 0, kvDim = 0, qkvOut = 0;
    uint32_t maxSeqLen = 2048;
    uint32_t gqaChunkSize = 64;

    // --- API ---
    bool load(GPUContext& ctx, const std::string& bundleDir,
              const std::string& ggufPath);

    /// Run one decode step. Updates KV cache, returns logits.
    std::vector<float> decode(int32_t tokenId, uint32_t posOffset);

    /// Greedy decode: return the argmax token from logits.
    static int32_t argmax(const std::vector<float>& logits);

    /// Get embedding for a token (CPU lookup + GPU upload)
    void uploadEmbedding(int32_t tokenId);

    /// Update dynamic params for this decode step
    void updateDecodeParams(uint32_t pos, uint32_t cacheLen);

private:
    void loadWeights(const std::string& ggufPath);
    void buildDecodePipeline();
    void computeRopeTables();

    const CompiledPipeline& loadKernel(const std::string& name);
    WGPUBindGroup makeBG(const CompiledPipeline& pl,
                         const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);
    std::unordered_map<std::string, const CompiledPipeline*> kernelCache_;
};
