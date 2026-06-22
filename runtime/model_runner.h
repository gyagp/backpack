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

    // ── MoE intermediate buffers (allocated when cfg.numExperts > 0) ──────
    GPUBuffer moeRouterOutBuf;   // [nExperts] f32 — router logits per token
    GPUBuffer moeIndicesBuf;     // [k] u32 — top-k expert indices
    GPUBuffer moeWeightsBuf;     // [k] f32 — top-k expert weights (softmaxed)
    GPUBuffer moeExpertOutBuf;   // [E] f32 — one expert's down-projection output
    GPUBuffer moeShexpGateUpBuf; // [2*IM_s] f32 — shared expert gate+up output
    GPUBuffer moeShexpActBuf;    // [IM_s] f32 — silu(gate)*up for shared expert
    GPUBuffer moeRoutedGateBuf;  // [IM_e] f32 — one routed-expert gate output
    GPUBuffer moeRoutedUpBuf;    // [IM_e] f32 — one routed-expert up output
    GPUBuffer moeRoutedActBuf;   // [IM_e] f32 — silu(gate)*up for one expert

    // Per-layer routed-expert quant types (set during weight load).
    // Used at dispatch time to pick iq2s_matmul_moe vs iq3s_matmul_moe vs iq4xs_matmul_moe.
    std::vector<uint32_t> moeExpertsGateType;
    std::vector<uint32_t> moeExpertsUpType;
    std::vector<uint32_t> moeExpertsDownType;

    // ── qwen35moe attention intermediate buffers (allocated when needed) ──
    GPUBuffer q35QjBuf;       // [2*qDim_actual] joint Q+gate output
    GPUBuffer q35QBuf;        // [qDim_actual] post-split Q
    GPUBuffer q35GateBuf;     // [qDim_actual] post-split gate
    GPUBuffer q35KBuf;        // [kvDim_actual] K (separate from fused)
    GPUBuffer q35VBuf;        // [kvDim_actual] V
    GPUBuffer q35AttnOutBuf;  // [qDim_actual] attention output (pre-gated)
    GPUBuffer q35CosSinBuf;   // [4 sections × max_pairs × 2] MRoPE cos/sin table
    GPUBuffer q35ConvOutBuf;  // [ssm conv channels] convolved Q/K/V
    GPUBuffer q35SsmQBuf;     // [ssm group_count * state_size] normalized Q
    GPUBuffer q35SsmKBuf;     // [ssm group_count * state_size] normalized K
    GPUBuffer q35SsmVBuf;     // [ssm inner_size] V
    GPUBuffer q35SsmBetaBuf;  // [ssm time_step_rank]
    GPUBuffer q35SsmAlphaBuf; // [ssm time_step_rank]
    GPUBuffer q35SsmGateBuf;  // [ssm time_step_rank]
    GPUBuffer q35SsmYBuf;     // [ssm inner_size]
    GPUBuffer q35SsmNormBuf;  // [ssm inner_size]
    GPUBuffer q35SsmZBuf;     // [ssm inner_size]

    // ── SSM persistent state (per layer, allocated when cfg.ssmInnerSize > 0) ──
    // conv state: rolling buffer of last conv_kernel input vectors per channel
    // h_state:    Mamba recurrent state [d_inner, d_state]
    std::vector<GPUBuffer> ssmConvState;
    std::vector<GPUBuffer> ssmHState;
    GPUBuffer attnPartialsBuf;

    // Dynamic params (single set — writeBuffer is queue-sequenced)
    GPUBuffer fusedRopeParamsBuf, chunkedAttnParamsBuf;
    GPUBuffer chunkedAttnParamsBufSWA; // sliding window layers (separate T_total)
    GPUBuffer q35RopeQParamsBuf, q35RopeKParamsBuf, q35KvWriteParamsBuf;

    // Pre-built dispatch lists (single — identical for every token)
    std::vector<Dispatch> allDecodeDispatches;
    std::vector<Dispatch> autoDecodeDispatches;  // embed_gather + all + argmax

    // Indices into autoDecodeDispatches that reference dynamic param buffers
    // (used to clone per-slot bind groups).
    std::vector<int> ropeDispatchIndices;   // fused_rope per layer
    std::vector<int> attnP1DispatchIndices; // gqa_chunked_pass1 per layer
    std::vector<int> attnP2DispatchIndices; // gqa_chunked_pass2 per layer
    std::vector<int> q35QRoPEDispatchIndices; // Qwen3.5 Q RoPE per attention layer
    std::vector<int> q35KvWriteDispatchIndices; // Qwen3.5 K RoPE + KV write per attention layer
    int argmaxDispatchIndex = -1;
    int argmaxReduceDispatchIndex = -1;

    // Shared argmax result buffer
    GPUBuffer argmaxResultBuf;
    GPUBuffer argmaxPartialsBuf;

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
        // Per-slot param buffers — allow writing next token's params
        // while GPU still reads current token's params.
        GPUBuffer ropeParamsBuf, attnParamsBuf;
        GPUBuffer attnParamsBufSWA; // sliding window layers
        GPUBuffer q35RopeQParamsBuf, q35RopeKParamsBuf, q35KvWriteParamsBuf;
        GPUBuffer tokenInBuf, tokenOutBuf;
        // Per-slot dispatch list (cloned from autoDecodeDispatches with
        // per-slot bind groups for param-referencing dispatches).
        std::vector<Dispatch> dispatches;
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
        // Sandwich norms (Gemma 4)
        GPUBuffer postNorm;        // post-attention norm (sandwich, before residual)
        GPUBuffer ffnNorm;         // pre-FFN norm (separate from postAttnNorm)
        GPUBuffer postFfwNorm;     // post-FFN norm (sandwich, before residual)
        // PLE (Per-Layer Embedding)
        GPUBuffer pleInpGateW, pleInpGateS;  // [E, pleSize] gate projection
        GPUBuffer pleProjW, pleProjS;        // [pleSize, E] back-projection
        GPUBuffer plePostNorm;               // RMSNorm on PLE output
        // Per-layer output scale
        GPUBuffer outScale;        // scalar [1]
        // Custom RoPE frequencies
        GPUBuffer ropeFreqs;

        // ── MoE expert weights (qwen35moe and similar) ──────────────────────
        // Routed experts: 3D tensors [nExperts, dim_out, dim_in] dequantized
        // and uploaded as flat Q8 (or kept as raw IQ bytes once GPU IQ kernels exist).
        GPUBuffer routerW, routerS;        // ffn_gate_inp.weight [nExperts, E]
        GPUBuffer expertsGateW, expertsGateS;  // ffn_gate_exps.weight [nExperts, IM_e, E]
        GPUBuffer expertsUpW, expertsUpS;      // ffn_up_exps.weight   [nExperts, IM_e, E]
        GPUBuffer expertsDownW, expertsDownS;  // ffn_down_exps.weight [nExperts, E, IM_e]
        // Shared expert (always active, no routing)
        GPUBuffer shexpGateW, shexpGateS;  // ffn_gate_shexp.weight [IM_s, E]
        GPUBuffer shexpUpW, shexpUpS;      // ffn_up_shexp.weight   [IM_s, E]
        GPUBuffer shexpDownW, shexpDownS;  // ffn_down_shexp.weight [E, IM_s]
        GPUBuffer shexpRouterW, shexpRouterS;  // ffn_gate_inp_shexp.weight (gating scalar?)
        // Attention gate (Qwen3.6 / qwen35moe gated attention output)
        GPUBuffer attnGateW, attnGateS;    // attn_gate.weight

        // ── qwen35moe attention-layer separate Q/K/V (step 1 of correctness wiring)
        // For qwen35moe attention layers, the standard fuse-Q/K/V loader path
        // is wrong because attn_q.weight produces joint Q+gate (2x qDim).
        // These slots hold the un-fused weights for the separate-dispatch path.
        GPUBuffer qjW, qjS;    // joint Q+gate (size [2*qDim, E])
        GPUBuffer kSepW, kSepS; // K alone (size [kvDim, E])
        GPUBuffer vSepW, vSepS; // V alone (size [kvDim, E])

        // ── SSM / Mamba per-layer weights (hybrid archs like qwen35moe) ─────
        // 31 of 41 qwen35moe layers are SSM-only (full_attention_interval=4).
        // Tensor names: ssm_conv1d.weight, ssm_dt.bias, ssm_a, ssm_beta.weight,
        // ssm_alpha.weight, ssm_norm.weight, ssm_out.weight.
        GPUBuffer ssmConv1dW;       // [d_inner, conv_k] depthwise conv1d
        GPUBuffer ssmDtBias;        // [d_inner] bias for dt projection
        GPUBuffer ssmA;             // [d_inner, d_state] state matrix (init log-space)
        GPUBuffer ssmBetaAlphaW, ssmBetaAlphaS; // fused beta || alpha projection
        GPUBuffer ssmBetaW, ssmBetaS;    // beta projection (Q8 if quantized)
        GPUBuffer ssmAlphaW, ssmAlphaS;  // alpha projection
        GPUBuffer ssmNorm;          // [d_inner] RMSNorm weight
        GPUBuffer ssmOutW, ssmOutS; // output projection
    };
    std::vector<LayerWeights> layerWeights;
    GPUBuffer finalNormW;
    GPUBuffer lmHeadW;           // fp16 LM head (fallback)
    GPUBuffer lmHeadQ8W, lmHeadQ8S;  // Q8 LM head (preferred)
    GPUBuffer lmHeadKQ;          // K-quant LM head
    bool lmHeadIsQ8 = false;
    bool lmHeadIsKQ = false;
    GGUFType weightQuantType = GGUF_TYPE_Q8_0;

    // PLE (Per-Layer Embedding) global buffers
    std::vector<float> pleEmbCPU;     // per-layer token embeddings (CPU)
    GPUBuffer pleModelProjW, pleModelProjS;  // [E, pleSize*nLayer] projection
    GPUBuffer pleProjNormW;           // RMSNorm weights [pleSize]
    GPUBuffer pleBuf;                 // intermediate [pleSize] for PLE computation
    GPUBuffer pleOutBuf;              // intermediate [E] for PLE output
    std::vector<GPUBuffer> pleSliceBufs;  // per-layer PLE slice [pleSize]

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
    // Optional second pair for Gemma 3/4 SWA layers (base=10000 vs global=1000000).
    GPUBuffer ropeCosBufSWA, ropeSinBufSWA;

    // Derived dimensions
    uint32_t qDim = 0, kvDim = 0, qkvOut = 0;
    uint32_t maxSeqLen = 4096;
    uint32_t gqaChunkSize = 64;

    // IQ-quant codebooks uploaded once and bound to IQ matmul kernels.
    // iq3sCodebookBuf: 512 u32  (2 KB)  for iq3s_matmul
    // iq2sCodebookBuf: 2048 u32 (8 KB)  for iq2s_matmul (each iq2s_grid entry = 2 u32)
    GPUBuffer iq3sCodebookBuf;
    GPUBuffer iq2sCodebookBuf;

    // Pass mode: false = single compute pass (faster on D3D12)
    bool passPerDispatch = true;
    bool useMMA = false;   // true on Vulkan with subgroup_matrix support
    bool useDP4A = false;  // true when dot4I8Packed is available (D3D12)
    int autoDecodePrefixCount = 1;  // dispatches before allDecodeDispatches in autoDecodeDispatches
    bool decodeUsesFusedRopeParams = false;

    // Logit softcapping (Gemma): applied between LM head and argmax
    WGPUComputePipeline softcapPipeline = nullptr;
    WGPUBindGroup softcapBG = nullptr;
    uint32_t softcapDispatchX = 0;
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
    void seedDecodeTokenInputs(int32_t tokenId);
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
    void prepareDecodeParams(uint32_t pos, uint32_t cacheLen, int slot);
    void resetKVCache();
    void refillCBPool(int slot);
    void autotuneDecodeDepth();
    void autotuneDecodeKernels();
    bool loadDecodeAutotuneCache();
    void saveDecodeAutotuneCache() const;
    void printActiveDecodeTuning(const char* prefix = "  Active decode tuning") const;
    bool hasBatchedPrefill() const { return pfCache.ready; }
    void destroy();

    // ─── MTP (Multi-Token Prediction) ────────────────────────────────────
    // Supports Gemma 4 (Q-only attention, shared KV) and Qwen 3.6 (full decoder block)
    enum class MTPType { None, Gemma4, Qwen36 };
    struct MTPConfig {
        MTPType type = MTPType::None;
        uint32_t numLayers = 0;      // number of MTP decoder layers
        uint32_t hiddenSize = 0;     // MTP hidden dim (may differ from backbone)
        uint32_t numDraftTokens = 1; // max draft tokens per step
    };
    MTPConfig mtpCfg;

    struct MTPWeights {
        // Pre-projection: concat(emb, hidden) → mtp_hidden
        GPUBuffer preProjW, preProjS;  // Linear(2*E, mtp_hidden)
        // Per MTP layer
        struct Layer {
            GPUBuffer inputNorm;    // RMSNorm
            GPUBuffer qW, qS;      // Q projection (Gemma4: Q-only)
            GPUBuffer oW, oS;      // Output projection
            GPUBuffer qNorm;        // Q head norm
            GPUBuffer ffnNorm;      // FFN pre-norm
            GPUBuffer guW, guS;     // gate_up
            GPUBuffer dnW, dnS;     // down
            GPUBuffer postAttnNorm; // post-attention norm
            GPUBuffer postFfnNorm;  // post-FFN norm
            // Qwen3.6: also has K/V projections
            GPUBuffer kW, kS, vW, vS;
            GPUBuffer enorm, hnorm; // DeepSeek/Qwen: separate norms
        };
        std::vector<Layer> layers;
        // Post-projection: mtp_hidden → backbone_hidden
        GPUBuffer postProjW, postProjS;
        GPUBuffer finalNorm;        // final RMSNorm before LM head
    };
    MTPWeights mtpWeights;

    // MTP inference
    int32_t mtpDraft(int32_t lastToken, uint32_t pos, std::vector<int32_t>& draftTokens);
    int32_t mtpVerifyAndAccept(const std::vector<int32_t>& draftTokens,
                               uint32_t pos, uint32_t& acceptedCount);

private:
    void loadWeights(const GGUFFile& gguf, const uint8_t* fileData);
    void buildDecodePipeline();

    // Append MoE FFN dispatches for layer `layerIdx` into allDecodeDispatches.
    // Reads x_in (normed input) and writes the residual update into x_out.
    // Returns false if the layer's weights are missing (handled gracefully).
    //
    // Dispatches inserted (per-token decode):
    //   1× router matmul         (Q8 small)         → moeRouterOutBuf [nExperts]
    //   1× moe_gate              (top-k softmax)    → moeIndicesBuf, moeWeightsBuf
    //   N× per-expert {gate, up, silu+mul, down}    → moeRoutedActBuf, accumulate
    //   3× shared expert {gate, up, silu+mul, down} → moeShexpActBuf, accumulate
    //
    // (Phase 3d, currently a stub — returns false until implemented.)
    bool appendMoeFfnDispatches(uint32_t layerIdx, GPUBuffer xIn, GPUBuffer xOut);
    void computeRopeTables();
    void initPrefillResources();

    const CompiledPipeline& getKernel(const std::string& name);
    const CompiledPipeline& getKernelHD(const std::string& name);
    const CompiledPipeline& getKernelGelu(const std::string& siluName);
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
