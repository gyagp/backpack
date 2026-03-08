#pragma once
/**
 * model_runner.h — GGUF-driven model inference engine.
 *
 * Reads model architecture directly from GGUF metadata.
 * Loads WGSL kernels from embedded shader constants.
 * No manifest.json or external kernel files needed.
 *
 * Supports any llama.cpp-compatible GGUF model with Q8_0 quantization.
 */

#include "gpu_context.h"
#include "gguf_loader.h"

#include <string>
#include <unordered_map>
#include <vector>

struct ModelRunner {
    GPUContext* gpu = nullptr;
    ModelConfig cfg;
    GGUFFile gguf;  // retained for tokenizer access

    // Intermediate buffers
    GPUBuffer xBuf, normOutBuf, qkvBuf, qRotBuf, attnOutBuf;
    GPUBuffer projOutBuf, gateUpBuf, siluOutBuf, rstdBuf, logitsBuf;
    GPUBuffer attnPartialsBuf;

    // KV cache
    struct KVEntry { GPUBuffer K, V; uint32_t len = 0; };
    std::vector<KVEntry> kvCache;

    // Per-layer weight buffers
    struct LayerWeights {
        GPUBuffer qkvW, qkvS;
        GPUBuffer oW, oS;
        GPUBuffer guW, guS;
        GPUBuffer dnW, dnS;
        GPUBuffer inputNorm, postAttnNorm;
        GPUBuffer qNorm, kNorm;
    };
    std::vector<LayerWeights> layerWeights;
    GPUBuffer finalNormW;
    GPUBuffer lmHeadW;
    GPUBuffer zeroBiasE, zeroBiasQKV, zeroBiasGU, zeroBiasV;

    // Pre-built decode pipeline
    std::vector<Dispatch> allDecodeDispatches;

    // GPU argmax (appended after LM head)
    GPUBuffer argmaxResultBuf;  // single i32

    // Embedding (CPU-side)
    std::vector<float> embeddingCPU;

    // RoPE tables
    GPUBuffer ropeCosBuf, ropeSinBuf;

    // Dynamic params
    GPUBuffer fusedRopeParamsBuf, chunkedAttnParamsBuf;
    std::vector<uint8_t> ropeParamData, chunkedAttnParamData;

    // Derived dimensions
    uint32_t qDim = 0, kvDim = 0, qkvOut = 0;
    uint32_t maxSeqLen = 2048;
    uint32_t gqaChunkSize = 64;

    // Pass mode: false = single compute pass (faster on D3D12)
    bool passPerDispatch = true;

    // Profiling
    GPUProfiler* profiler = nullptr;

    // --- API ---
    bool load(GPUContext& ctx, const std::string& ggufPath);
    std::vector<float> decode(int32_t tokenId, uint32_t posOffset);
    /// Fast decode: returns just the argmax token ID (no logits readback)
    int32_t decodeArgmax(int32_t tokenId, uint32_t posOffset);
    static int32_t argmax(const std::vector<float>& logits);

    void enableProfiling();
    void printProfileReport();

    void uploadEmbedding(int32_t tokenId);
    void updateDecodeParams(uint32_t pos, uint32_t cacheLen);

private:
    void loadWeights(const GGUFFile& gguf, const std::vector<uint8_t>& fileData);
    void buildDecodePipeline();
    void computeRopeTables();

    const CompiledPipeline& getKernel(const std::string& name);
    WGPUBindGroup makeBG(const CompiledPipeline& pl,
                         const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);
};
