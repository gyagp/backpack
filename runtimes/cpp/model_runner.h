#pragma once
/**
 * model_runner.h — Model inference runtime (GGUF + ONNX).
 *
 * Reads model architecture from GGUF metadata or ONNX config.
 * Loads WGSL kernels from embedded shader constants.
 * No manifest.json or external kernel files needed.
 *
 * Supports:
 *   - Any llama.cpp-compatible GGUF model with Q8_0 quantization
 *   - ONNX GenAI models (e.g. Phi-4-mini) with Q4/Q8 quantization
 */

#include "gpu_context.h"
#include "gguf_loader.h"

#include <cstdint>
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
    struct PoolSlot {
        WGPUBuffer stagingBuf = nullptr;
        WGPUFuture pendingFuture{};
        // Multi-group pre-recorded CBs: each "token" is a sequence of
        // nGroups CBs. cbPool[tokenIdx * nGroups + groupIdx] is one CB.
        std::vector<WGPUCommandBuffer> cbPool;
        int cbIdx = 0;  // index of next token's first CB
    };
    std::vector<PoolSlot> pool;
    int nGroups = 1;  // number of CB groups per token (1 = single submit)
    int decodePoolCapacity = 3;
    int decodePoolDepth = 3;
    int decodeCbPoolBatch = 128;

    // Dynamic param templates (CPU-side, modified per step)
    std::vector<uint8_t> ropeParamData, chunkedAttnParamData;

    // KV cache (shared — natural RAW dependency, no WAR)
    struct KVEntry { GPUBuffer K, V; uint32_t len = 0; };
    std::vector<KVEntry> kvCache;

    // Per-layer weight buffers (read-only, shared)
    struct LayerWeights {
        GPUBuffer qkvW, qkvS;       // Q8_0: weight + scale buffers
        GPUBuffer oW, oS;
        GPUBuffer guW, guS;
        GPUBuffer dnW, dnS;
        GPUBuffer inputNorm, postAttnNorm;
        GPUBuffer qNorm, kNorm;
        // K-quant: single buffer per weight (raw block data as u32)
        GPUBuffer qkvKQ, oKQ, guKQ, dnKQ;
    };
    std::vector<LayerWeights> layerWeights;
    GPUBuffer finalNormW;
    GPUBuffer lmHeadW;           // fp16 LM head (fallback)
    GPUBuffer lmHeadQ8W, lmHeadQ8S;  // Q8 LM head (preferred)
    GPUBuffer lmHeadKQ;          // K-quant LM head
    bool lmHeadIsQ8 = false;
    bool lmHeadIsKQ = false;
    GGUFType weightQuantType = GGUF_TYPE_Q8_0;  // detected from first weight tensor
    // K-quant params per projection type (shared across layers)
    uint32_t kqQkvNBlocks = 0, kqQkvRowStride = 0;
    uint32_t kqONBlocks = 0, kqORowStride = 0;
    uint32_t kqGuNBlocks = 0, kqGuRowStride = 0;
    uint32_t kqDnNBlocks = 0, kqDnRowStride = 0;
    uint32_t kqLmNBlocks = 0, kqLmRowStride = 0;
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
    bool useMMA = false;   // true on Vulkan with subgroup_matrix support
    bool useDP4A = false;  // true when dot4I8Packed is available (D3D12)
    struct KernelTuning {
        bool decodeUseFastQkv = false;
        bool decodeUseFastOproj = false;
        bool decodeUseFastGateup = false;
        bool decodeUseWideFp16 = false;
        bool prefillUseWidePrequant = false;
        bool prefillUseWidePrequantAdd = false;
        uint32_t prefillMatM = 8;
        uint32_t prefillMatN = 32;
        uint32_t prefillWideMatM = 8;
        uint32_t prefillWideMatN = 32;
        uint32_t prefillDnM = 8;
        uint32_t prefillDnN = 32;
        uint32_t prefillAttnBlockQ = 4;
    } tuning;

    struct DecodeDispatchIndices {
        int qkv = -1;
        int oproj = -1;
        int gateup = -1;
    };
    struct DecodeVariantBindGroups {
        WGPUBindGroup qkvBase = nullptr;
        WGPUBindGroup qkvFast = nullptr;
        WGPUBindGroup oprojBase = nullptr;
        WGPUBindGroup oprojFast = nullptr;
        WGPUBindGroup gateupBase = nullptr;
        WGPUBindGroup gateupFast = nullptr;
    };
    std::vector<DecodeDispatchIndices> decodeDispatchIndices;
    std::vector<DecodeVariantBindGroups> decodeVariantBGs;
    bool decodeFastVariantsAvailable = false;

    // Profiling
    GPUProfiler* profiler = nullptr;
    struct ClockCalibration* calibration = nullptr;

    // ONNX-specific: partial RoPE and pre-computed tables
    uint32_t rotaryDim = 0;         // 0 = full RoPE (rotaryDim == headDim)
    bool hasPrecomputedRope = false; // true = use ONNX cos/sin cache
    std::string modelFormat;        // "gguf" or "onnx"

    // --- API ---
    bool load(GPUContext& ctx, const std::string& ggufPath);
    bool loadOnnx(GPUContext& ctx, const std::string& onnxDir);
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
    void autotuneDecodeDepth();
    void autotuneDecodeKernels();
    bool loadDecodeAutotuneCache();
    void saveDecodeAutotuneCache() const;
    void printActiveDecodeTuning(const char* prefix = "  Active decode tuning") const;
    void destroy();

private:
    void loadWeights(const GGUFFile& gguf, const std::vector<uint8_t>& fileData);
    void buildDecodePipeline();
    void computeRopeTables();
    void initPrefillResources();

    const CompiledPipeline& getKernel(const std::string& name);
    const CompiledPipeline& getKernelHD(const std::string& name);
    std::string patchShaderHD(const char* source) const;
    WGPUBindGroup makeBG(const CompiledPipeline& pl,
                         const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);
    void applyDecodeKernelSelection(bool useFastQkv, bool useFastOproj,
                                    bool useFastGateup);
    double benchmarkDecodeConfig(int depth, int nTokens, int repeats = 1);
    std::string decodeAutotuneCachePath() const;
    std::string decodeAutotuneCacheKey() const;

    // --- Pre-allocated prefill resources (sized to maxSeqLen) ---
    struct PrefillCache {
        bool ready = false;
        GPUBuffer pX, pNorm, pQkv, pQRot, pAttn, pProj, pGU, pRstd;
        GPUBuffer pNormQ, pNormQS, pAttnQ, pAttnQS, pGUQ, pGUQS;
        GPUBuffer pQkvP, pOpP, pGuP, pDnP, pRmsP, pLmP;
        GPUBuffer pNormQP, pAttnQP, pGUQP;
        std::vector<GPUBuffer> ropeParams;   // one per layer
        std::vector<GPUBuffer> attnParams;   // one per layer
        // Cached bind groups (stable since buffer handles don't change)
        struct LayerBGs {
            WGPUBindGroup rms, qnorm, qkv, rope, attn, attnq, oproj;
            WGPUBindGroup addrms, gateup, siluq, downsilu;
        };
        std::vector<LayerBGs> layerBGs;
        WGPUBindGroup finalRmsBG = nullptr;
        WGPUBindGroup lmBG = nullptr;
        WGPUBindGroup argmaxBG = nullptr;

        // Pre-recorded indirect dispatch table (static pipeline + bind group)
        struct IndirectEntry {
            WGPUComputePipeline pipeline;
            WGPUBindGroup       bindGroup;
            uint64_t            indirectOffset;  // byte offset into indirectBuf
            std::string         name;
        };
        std::vector<IndirectEntry> indirectTable;
        GPUBuffer indirectBuf;  // [gx, gy, gz] × N dispatches
    };
    PrefillCache pfCache;
};
