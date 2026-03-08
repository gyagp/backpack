#pragma once
/**
 * model_runner.h — GGUF-driven model inference runtime.
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

    // --- Intermediate buffers (single set, GPU-sequential execution) ---
    // WebGPU queue executes command buffers in submission order, so
    // intermediates are safe to share. Only staging buffers need per-slot
    // duplication for async readback.
    GPUBuffer xBuf, normOutBuf, qkvBuf, qRotBuf, attnOutBuf;
    GPUBuffer projOutBuf, gateUpBuf, rstdBuf, logitsBuf;
    GPUBuffer attnPartialsBuf;

    // Dynamic params (single set — writeBuffer is queue-sequenced)
    GPUBuffer fusedRopeParamsBuf, chunkedAttnParamsBuf;

    // Pre-built dispatch lists (single — identical for every token)
    std::vector<Dispatch> allDecodeDispatches;
    std::vector<Dispatch> autoDecodeDispatches;  // embed_gather + all + argmax

    // Shared argmax result buffer
    GPUBuffer argmaxResultBuf;

    // --- Staging pool for pipelined async readback ---
    // Each pool slot has its own staging buffer + pre-recorded CBs.
    // The decode loop cycles through slots round-robin, allowing
    // N tokens to be in-flight simultaneously.
    static constexpr int POOL_DEPTH = 3;
    static constexpr int CB_POOL_BATCH = 128;
    struct PoolSlot {
        WGPUBuffer stagingBuf = nullptr;
        WGPUFuture pendingFuture{};
        // Multi-group pre-recorded CBs: each "token" is a sequence of
        // nGroups CBs. cbPool[tokenIdx * nGroups + groupIdx] is one CB.
        std::vector<WGPUCommandBuffer> cbPool;
        int cbIdx = 0;  // index of next token's first CB
    };
    PoolSlot pool[POOL_DEPTH];
    int nGroups = 1;  // number of CB groups per token (1 = single submit)

    // Dynamic param templates (CPU-side, modified per step)
    std::vector<uint8_t> ropeParamData, chunkedAttnParamData;

    // KV cache (shared — natural RAW dependency, no WAR)
    struct KVEntry { GPUBuffer K, V; uint32_t len = 0; };
    std::vector<KVEntry> kvCache;

    // Per-layer weight buffers (read-only, shared)
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
    GPUBuffer lmHeadW;           // fp16 LM head (fallback)
    GPUBuffer lmHeadQ8W, lmHeadQ8S;  // Q8 LM head (preferred)
    bool lmHeadIsQ8 = false;
    GPUBuffer zeroBiasE, zeroBiasQKV, zeroBiasGU, zeroBiasV;

    // Embedding (CPU-side for prefill, GPU-side for decode)
    GPUBuffer embeddingGpuBuf;
    std::vector<float> embeddingCPU;

    // RoPE tables (read-only, shared)
    GPUBuffer ropeCosBuf, ropeSinBuf;

    // Derived dimensions
    uint32_t qDim = 0, kvDim = 0, qkvOut = 0;
    uint32_t maxSeqLen = 4096;
    uint32_t gqaChunkSize = 64;

    // Pass mode: false = single compute pass (faster on D3D12)
    bool passPerDispatch = true;

    // Profiling
    GPUProfiler* profiler = nullptr;
    struct ClockCalibration* calibration = nullptr;

    // --- API ---
    bool load(GPUContext& ctx, const std::string& ggufPath);
    std::string ggufPath;  // stored for profile output location
    std::vector<float> decode(int32_t tokenId, uint32_t posOffset);
    int32_t decodeArgmax(int32_t tokenId, uint32_t posOffset);
    /// Submit decode to pool slot. Call readArgmax(slot) later.
    void submitDecode(uint32_t posOffset, int slot);
    /// Wait for and read the argmax result from pool slot.
    int32_t readArgmax(int slot);
    /// Fire-and-forget prefill step: upload embedding + params, submit, no readback.
    /// GPU queue executes in order, so T sequential prefillStep calls are safe.
    void prefillStep(int32_t tokenId, uint32_t posOffset);
    /// Finish prefill: submit last token and read back logits.
    std::vector<float> prefillFinish(int32_t tokenId, uint32_t posOffset);
    /// Batched prefill: process all T tokens in parallel (one weight read).
    /// Returns argmax token ID (computed on GPU, only 4 bytes readback).
    int32_t prefillBatched(const int32_t* tokenIds, uint32_t T,
                           uint32_t posOffset);
    static int32_t argmax(const std::vector<float>& logits);

    void enableProfiling();
    void printProfileReport(int nDecodeTokens = 0, int nPrefillTokens = 0,
                            double prefillMs = 0, double decodeMs = 0,
                            const std::string& profileOutputPath = "");

    void uploadEmbedding(int32_t tokenId);
    void updateDecodeParams(uint32_t pos, uint32_t cacheLen);
    void resetKVCache();
    void refillCBPool(int slot);

private:
    void loadWeights(const GGUFFile& gguf, const std::vector<uint8_t>& fileData);
    void buildDecodePipeline();
    void computeRopeTables();
    void initPrefillResources();

    const CompiledPipeline& getKernel(const std::string& name);
    WGPUBindGroup makeBG(const CompiledPipeline& pl,
                         const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);

    // --- Pre-allocated prefill resources (sized to maxSeqLen) ---
    struct PrefillCache {
        bool ready = false;
        GPUBuffer pX, pNorm, pQkv, pQRot, pAttn, pProj, pGU, pRstd;
        GPUBuffer pQkvP, pOpP, pGuP, pDnP, pRmsP, pLmP;
        std::vector<GPUBuffer> ropeParams;   // one per layer
        std::vector<GPUBuffer> attnParams;   // one per layer
        // Cached bind groups (stable since buffer handles don't change)
        struct LayerBGs {
            WGPUBindGroup rms, qkv, rope, attn, oproj, addrms, gateup, downsilu;
        };
        std::vector<LayerBGs> layerBGs;
        WGPUBindGroup finalRmsBG = nullptr;
        WGPUBindGroup lmBG = nullptr;
        WGPUBindGroup argmaxBG = nullptr;
    };
    PrefillCache pfCache;
};
