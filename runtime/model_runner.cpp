#include "model_runner.h"
#include "onnx_loader.h"
#include "mapped_file.h"
#include "wgsl_shaders.h"
#include "clock_calibration.h"
#include "profile_html.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace {

const WGPULimits& effectiveLimits(const GPUContext& gpu) {
    return gpu.deviceLimits.maxComputeInvocationsPerWorkgroup != 0
        ? gpu.deviceLimits
        : gpu.adapterLimits;
}

bool canCompileEmbeddedKernel(GPUContext& gpu, const char* kernelName) {
    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(kernelName);
    if (it == kernels.end()) return false;

    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = {it->second.source, strlen(it->second.source)};
    WGPUShaderModuleDescriptor smD{};
    smD.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&src);
    auto sm = wgpuDeviceCreateShaderModule(gpu.device, &smD);
    if (!sm) return false;
    wgpuShaderModuleRelease(sm);
    return true;
}

int chooseDecodePoolDepth(const GPUContext& gpu) {
    const auto& limits = effectiveLimits(gpu);
    int depth = 2;
    if (limits.maxComputeInvocationsPerWorkgroup >= 256) depth++;
    if (limits.maxComputeWorkgroupStorageSize >= 24u * 1024u) depth++;
    if (gpu.backendType != WGPUBackendType_D3D12 && gpu.supportsSubgroups) depth++;
    return gpu.backendType == WGPUBackendType_D3D12
        ? std::clamp(depth, 3, 8)
        : std::clamp(depth, 2, 8);
}

int chooseDecodeCbPoolBatch(const GPUContext& gpu, const ModelConfig& cfg) {
    const auto& limits = effectiveLimits(gpu);
    int batch = 64;
    if (limits.maxComputeInvocationsPerWorkgroup >= 256) batch += 32;
    if (limits.maxComputeWorkgroupStorageSize >= 24u * 1024u) batch += 32;
    if (gpu.backendType != WGPUBackendType_D3D12 && gpu.supportsSubgroupMatrix) batch += 32;
    if (cfg.nLayer >= 32) batch += 32;
    if (limits.maxStorageBufferBindingSize >= (512ull << 20)) batch += 32;
    return gpu.backendType == WGPUBackendType_D3D12
        ? std::clamp(batch, 96, 160)
        : std::clamp(batch, 64, 256);
}

std::string backendName(WGPUBackendType backendType) {
    switch (backendType) {
        case WGPUBackendType_D3D12: return "d3d12";
        case WGPUBackendType_Vulkan: return "vulkan";
        case WGPUBackendType_Metal: return "metal";
        default: return "other";
    }
}

}  // namespace

// ─── Helpers ─────────────────────────────────────────────────────────────────

WGPUBindGroup ModelRunner::makeBG(
        const CompiledPipeline& pl,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings) {
    WGPUBindGroupEntry entries[16];
    for (size_t i = 0; i < bindings.size() && i < 16; i++) {
        memset(&entries[i], 0, sizeof(WGPUBindGroupEntry));
        entries[i].binding = bindings[i].first;
        entries[i].buffer  = bindings[i].second.handle;
        entries[i].size    = bindings[i].second.size;
    }
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = pl.bgLayout;
    d.entryCount = (uint32_t)bindings.size();
    d.entries = entries;
    return wgpuDeviceCreateBindGroup(gpu->device, &d);
}

const CompiledPipeline& ModelRunner::getKernel(const std::string& name) {
    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(name);
    if (it == kernels.end()) {
        fprintf(stderr, "Kernel not found: %s\n", name.c_str());
        exit(1);
    }
    return gpu->getOrCreatePipeline(name, it->second.source,
                                     it->second.numBindings);
}

// ─── HD-patched kernel loading ───────────────────────────────────────────────

std::string ModelRunner::patchShaderHD(const char* source) const {
    std::string s(source);
    uint32_t hd = cfg.headDim;
    if (hd == 128) return s;  // no patching needed

    // Replace "const HD: u32 = 128u;" with actual value
    {
        const char* pat = "const HD: u32 = 128u;";
        std::string rep = "const HD: u32 = " + std::to_string(hd) + "u;";
        auto pos = s.find(pat);
        if (pos != std::string::npos)
            s.replace(pos, strlen(pat), rep);
    }

    // HD_PER_THREAD: for 32-thread kernels, HD/32. Must divide evenly.
    if (hd % 32 == 0) {
        uint32_t hpt = hd / 32;
        {
            const char* pat = "const HD_PER_THREAD: u32 = 4u;";
            std::string rep = "const HD_PER_THREAD: u32 = " + std::to_string(hpt) + "u;";
            auto pos = s.find(pat);
            if (pos != std::string::npos)
                s.replace(pos, strlen(pat), rep);
        }

        // For unrolled 4-element load/store patterns, replace with loop.
        // Pattern: "let q0 = Q[...]; let q1 = Q[... + 1u]; let q2 = Q[... + 2u]; let q3 = Q[... + 3u];"
        // Replace with HD_PER_THREAD-element loop.
        // This is complex to do via string replacement, so for HD_PER_THREAD != 4,
        // we use the generic gqa_fused_attn kernel instead of gqa_chunked.
    }

    // HD_TILES = HD / 16 (flash_attn_vulkan)
    {
        const char* pat = "const HD_TILES: u32 = 8u;";
        if (hd % 16 == 0) {
            uint32_t hdt = hd / 16;
            std::string rep = "const HD_TILES: u32 = " + std::to_string(hdt) + "u;";
            auto pos = s.find(pat);
            if (pos != std::string::npos)
                s.replace(pos, strlen(pat), rep);
        }
    }

    // HALF = HD / 2 (rope_batched_simple)
    {
        const char* pat = "const HALF: u32 = 64u;";
        uint32_t half = hd / 2;
        std::string rep = "const HALF: u32 = " + std::to_string(half) + "u;";
        auto pos = s.find(pat);
        if (pos != std::string::npos)
            s.replace(pos, strlen(pat), rep);
    }

    // Local variable arrays sized to HD
    {
        std::string pat = "array<f32, 128>";
        std::string rep = "array<f32, " + std::to_string(hd) + ">";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }

    // Shared memory: out_acc for flash_attn (BQ=16 × HD)
    {
        const char* pat = "array<f32, 2048>";
        std::string rep = "array<f32, " + std::to_string(16 * hd) + ">";
        auto pos = s.find(pat);
        if (pos != std::string::npos)
            s.replace(pos, strlen(pat), rep);
    }

    // Triton-generated kernel: literal "* 128;" and "f32(128.0)"
    {
        std::string pat1 = "* 128;";
        std::string rep1 = "* " + std::to_string(hd) + ";";
        size_t pos = 0;
        while ((pos = s.find(pat1, pos)) != std::string::npos) {
            s.replace(pos, pat1.size(), rep1);
            pos += rep1.size();
        }
    }
    {
        std::string pat2 = "f32(128.0)";
        std::string rep2 = "f32(" + std::to_string(hd) + ".0)";
        size_t pos = 0;
        while ((pos = s.find(pat2, pos)) != std::string::npos) {
            s.replace(pos, pat2.size(), rep2);
            pos += rep2.size();
        }
    }

    return s;
}

const CompiledPipeline& ModelRunner::getKernelHD(const std::string& name) {
    if (cfg.headDim == 128) return getKernel(name);

    // Patched pipeline: keyed by name + "_HD" + headDim
    std::string patchedName = name + "_HD" + std::to_string(cfg.headDim);

    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(name);
    if (it == kernels.end()) {
        fprintf(stderr, "Kernel not found: %s\n", name.c_str());
        exit(1);
    }

    std::string patchedSource = patchShaderHD(it->second.source);
    return gpu->getOrCreatePipeline(patchedName, patchedSource,
                                     it->second.numBindings);
}

// ─── Patch SiLU→GELU in kernel source ───────────────────────────────────────

static std::string patchSiluToGelu(const std::string& source) {
    std::string s = source;

    // Replace inline SiLU: gate / (1.0 + exp(-gate)) → GELU approximation
    // The GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // For fused down+act+add kernels, the pattern is:
    //   let silu_gate = gate / (1.0 + exp(-gate));
    // or inline:
    //   gate / (1.0 + exp(-gate)) * up

    // Helper function approach: inject a GELU function and replace SiLU calls
    // First, replace the "silu_gate" named pattern
    {
        std::string pat = "gate / (1.0 + exp(-gate))";
        std::string rep = "(gate * 0.5 * (1.0 + tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate))))";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }
    // Also handle the f16 variant
    {
        std::string pat = "f16(gate / (1.0 + exp(-gate)) * up)";
        std::string rep = "f16(gate * 0.5 * (1.0 + tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate))) * up)";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }

    // Replace kernel description comments
    {
        std::string pat = "SiLU";
        std::string rep = "GELU";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }

    return s;
}

const CompiledPipeline& ModelRunner::getKernelGelu(const std::string& siluName) {
    std::string geluName = siluName;
    // Replace "silu" with "gelu" in the kernel name
    size_t pos = geluName.find("silu");
    if (pos != std::string::npos)
        geluName.replace(pos, 4, "gelu");
    else
        geluName += "_gelu";

    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(siluName);
    if (it == kernels.end()) {
        fprintf(stderr, "Kernel not found for GELU patching: %s\n", siluName.c_str());
        exit(1);
    }

    std::string patchedSource = patchSiluToGelu(it->second.source);
    return gpu->getOrCreatePipeline(geluName, patchedSource,
                                     it->second.numBindings);
}

// ─── fp16 conversion helpers ─────────────────────────────────────────────────

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0)       f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else                f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, 4);
    return result;
}

static uint16_t f32_to_fp16(float v) {
    uint32_t fb;
    memcpy(&fb, &v, 4);
    uint32_t s = (fb >> 16) & 0x8000;
    int32_t  e = ((fb >> 23) & 0xFF) - 112;
    uint32_t m = (fb >> 13) & 0x3FF;
    if (e <= 0)  return (uint16_t)s;
    if (e > 30)  return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | m);
}

// ─── Upload Q8 weight pair ──────────────────────────────────────────────────

static void uploadQ8Weight(GPUContext& gpu, const std::string& name,
                           const Q8Repacked& rep,
                           GPUBuffer& wBuf, GPUBuffer& sBuf) {
    uint64_t wSize = rep.weights.size() * 4;
    uint64_t sSize = rep.scales.size() * 4;
    wBuf = gpu.createBuffer(name + ".w", wSize);
    sBuf = gpu.createBuffer(name + ".s", sSize);
    gpu.writeBuffer(wBuf, rep.weights.data(), wSize);
    gpu.writeBuffer(sBuf, rep.scales.data(), sSize);
}

// ─── Load model ──────────────────────────────────────────────────────────────

bool ModelRunner::load(GPUContext& ctx, const std::string& path) {
    gpu = &ctx;
    ggufPath = path;
    modelFormat = "gguf";
    fprintf(stderr, "  [runner.load] opening GGUF: %s\n", path.c_str()); fflush(stderr);

    // Parse GGUF (metadata + tensor index)
    if (!gguf.open(ggufPath)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", ggufPath.c_str());
        return false;
    }
    fprintf(stderr, "  [runner.load] GGUF opened, extracting config...\n"); fflush(stderr);

    // Extract model config from GGUF metadata
    cfg = extractModelConfig(gguf);
    fprintf(stderr, "  [runner.load] config extracted: arch=%s nLayer=%u\n", cfg.arch.c_str(), cfg.nLayer); fflush(stderr);
    if (cfg.numExperts > 0) {
        fprintf(stderr, "  [runner.load] MoE detected: %u experts, top-%u active, expert FFN dim=%u\n",
                cfg.numExperts, cfg.numExpertsPerTok, cfg.moeIntermediateSize);
    }

    // ─── Early support gate ────────────────────────────────────────────────
    // Reject combinations that backpack cannot decode at all. MoE archs
    // proceed through loading (small weights are uploaded); the forward path
    // will refuse cleanly if expert dispatch isn't wired up yet.
    {
        // Unsupported GGUF quant formats (have no dequantizer yet).
        auto quantName = [](uint32_t t) -> const char* {
            switch (t) {
                case 16: return "IQ2_XXS";  // not yet ported
                case 19: return "IQ1_S";    // not yet ported
                case 29: return "IQ1_M";    // not yet ported
                case 34: return "TQ1_0";    // not yet ported
                case 35: return "TQ2_0";    // not yet ported
                default: return nullptr;
            }
        };
        const char* probeNames[] = {
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_qkv.weight",
        };
        for (auto* n : probeNames) {
            auto it = gguf.tensor_index.find(n);
            if (it == gguf.tensor_index.end()) continue;
            uint32_t t = (uint32_t)gguf.tensors[it->second].type;
            if (auto* qn = quantName(t)) {
                fprintf(stderr,
                    "\nERROR: GGUF quant format '%s' (type=%u) is not yet supported by backpack.\n"
                    "       Probed tensor: %s\n",
                    qn, t, n);
                return false;
            }
        }
    }

    fprintf(stderr, "  [runner.load] printing model info...\n"); fflush(stderr);

    fprintf(stderr, "Model: %s (%u layers, E=%u, HD=%u, V=%u, KV=%u)\n",
           cfg.arch.c_str(), cfg.nLayer, cfg.nEmbd, cfg.headDim,
           cfg.nVocab, cfg.nKvHeads);
    fprintf(stderr, "  RoPE theta=%.0f, RMSNorm eps=%.1e, QK-norm=%s\n",
           cfg.ropeTheta, cfg.rmsNormEps,
           cfg.hasQkNorm ? "yes" : "no");
    if (cfg.numExperts > 0) {
        fprintf(stderr, "  MoE: %u experts, top-%u active per token, expert FFN dim=%u, shared FFN dim=%u\n",
                cfg.numExperts, cfg.numExpertsPerTok, cfg.moeIntermediateSize, cfg.moeSharedIntermediateSize);
    }
    if (cfg.ssmInnerSize > 0) {
        fprintf(stderr, "  SSM (Mamba): d_inner=%u, d_state=%u, conv_k=%u, groups=%u, dt_rank=%u%s\n",
                cfg.ssmInnerSize, cfg.ssmStateSize, cfg.ssmConvKernel,
                cfg.ssmGroupCount, cfg.ssmTimeStepRank,
                cfg.fullAttentionInterval > 0
                    ? (std::string(" (full-attn every ") + std::to_string(cfg.fullAttentionInterval) + " layers)").c_str()
                    : "");
    }

    // Single compute pass for all backends — Dawn handles barriers internally
    passPerDispatch = false;
    fprintf(stderr, "  Backend: %s, single-pass dispatch\n",
           gpu->backendType == WGPUBackendType_D3D12 ? "D3D12" : "Vulkan");

    // Memory-map GGUF file for tensor data
    MappedFile ggufMap;
    if (!ggufMap.open(ggufPath)) {
        fprintf(stderr, "Failed to mmap GGUF file: %s\n", ggufPath.c_str());
        return false;
    }

    // Load weights
    loadWeights(gguf, ggufMap.data);
    fprintf(stderr, "  [runner.load] weights loaded\n"); fflush(stderr);

    // Upload IQ codebook buffers (small — 2 KB + 8 KB — once per session).
    // Only needed when iq3s_matmul / iq2s_matmul kernels will be dispatched,
    // i.e. MoE archs whose routed-expert weights are IQ-quantized. Skipping
    // for non-MoE / dense models saves ~10 KB GPU memory.
    if (cfg.numExperts > 0) {
        uint32_t cb3_n = 0; const uint32_t* cb3 = getIq3sGrid(&cb3_n);
        uint32_t cb2_n = 0; const uint32_t* cb2 = getIq2sGridU32(&cb2_n);
        iq3sCodebookBuf = gpu->createBuffer("iq3s_codebook", cb3_n * 4);
        iq2sCodebookBuf = gpu->createBuffer("iq2s_codebook", cb2_n * 4);
        gpu->writeBuffer(iq3sCodebookBuf, cb3, cb3_n * 4);
        gpu->writeBuffer(iq2sCodebookBuf, cb2, cb2_n * 4);
        fprintf(stderr, "  IQ codebooks uploaded: iq3s=%u u32, iq2s=%u u32\n", cb3_n, cb2_n);
    }

    // For hybrid MoE + SSM archs: MoE FFN dispatch IS wired (Phase 3d).
    // SSM layers emit NO dispatches (passthrough — output incorrect).
    // Allow run so a first tg/s number is measurable even though output is wrong.
    if (cfg.numExperts > 0) {
        fprintf(stderr,
            "  NOTE: hybrid arch — MoE FFN wired; SSM layers passthrough.\n"
            "        Output will be INCORRECT but forward pass completes.\n"
            "        Reported tg/s is a first-attempt timing — do NOT compare to\n"
            "        llama.cpp until SSM forward + attn_gate are also wired.\n");
    }

    // RoPE tables
    computeRopeTables();
    fprintf(stderr, "  [runner.load] RoPE tables computed\n"); fflush(stderr);

    // Build decode pipeline
    buildDecodePipeline();
    fprintf(stderr, "  [runner.load] decode pipeline built\n"); fflush(stderr);

    return true;
}

// ─── Load ONNX model ─────────────────────────────────────────────────────────

bool ModelRunner::loadOnnx(GPUContext& ctx, const std::string& onnxDir) {
    gpu = &ctx;
    ggufPath = onnxDir;
    modelFormat = "onnx";

    // 1. Load ONNX model (parse protobuf, extract & repack weights)
    OnnxLoadResult onnx;
    if (!loadOnnxModel(onnxDir, onnx)) {
        fprintf(stderr, "Failed to load ONNX model from: %s\n", onnxDir.c_str());
        return false;
    }

    cfg = onnx.cfg;
    rotaryDim = onnx.rotaryDim;
    hasPrecomputedRope = onnx.hasPrecomputedRope;

    fprintf(stderr, "Model: %s (%u layers, E=%u, HD=%u, V=%u, KV=%u) [ONNX]\n",
           cfg.arch.c_str(), cfg.nLayer, cfg.nEmbd, cfg.headDim,
           cfg.nVocab, cfg.nKvHeads);
    fprintf(stderr, "  RoPE theta=%.0f, RMSNorm eps=%.1e, QK-norm=%s\n",
           cfg.ropeTheta, cfg.rmsNormEps,
           cfg.hasQkNorm ? "yes" : "no");
    if (rotaryDim > 0 && rotaryDim != cfg.headDim)
        fprintf(stderr, "  Partial RoPE: rotary_dim=%u (head_dim=%u)\n", rotaryDim, cfg.headDim);
    if (cfg.headDim != 128 && cfg.headDim % 32 == 0)
        fprintf(stderr, "  Note: head_dim=%u (attention kernels will be HD-patched)\n", cfg.headDim);
    if (cfg.headDim % 32 != 0) {
        fprintf(stderr, "Error: head_dim=%u is not a multiple of 32\n", cfg.headDim);
        return false;
    }

    passPerDispatch = false;
    fprintf(stderr, "  Backend: %s, single-pass dispatch\n",
           gpu->backendType == WGPUBackendType_D3D12 ? "D3D12" : "Vulkan");

    // 2. Upload weights to GPU
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "  Uploading weights to GPU...\n");

    uint32_t qDimL = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;

    // Compute max per-layer dimensions for buffer allocation
    uint32_t maxQkvOut = qkvOutL;
    uint32_t maxIntermediateSize = cfg.intermediateSize;
    // For MoE archs (qwen35moe) without dense ffn dim, fall back to moe dim.
    if (maxIntermediateSize == 0 && cfg.moeIntermediateSize > 0) {
        maxIntermediateSize = std::max(cfg.moeIntermediateSize, cfg.moeSharedIntermediateSize);
        // For qwen35moe attention dispatch needs qDim_actual sized buffers too;
        // use a generous bound to cover all temp buffer uses.
        maxIntermediateSize = std::max(maxIntermediateSize, 4u * cfg.nEmbd);
    }
    uint32_t maxQDim = qDim;
    for (auto& pl : cfg.perLayer) {
        uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
        if (plQkvOut > maxQkvOut) maxQkvOut = plQkvOut;
        if (pl.intermediateSize > maxIntermediateSize) maxIntermediateSize = pl.intermediateSize;
        if (pl.qDim > maxQDim) maxQDim = pl.qDim;
    }

    // Zero bias buffers (sized to max across layers)
    uint32_t maxBias = std::max({cfg.nEmbd, maxQkvOut,
                                 2 * maxIntermediateSize, cfg.nVocab});
    fprintf(stderr, "  zero-bias sizing: nEmbd=%u maxQkvOut=%u maxIntermediateSize=%u maxBias=%u\n",
            cfg.nEmbd, maxQkvOut, maxIntermediateSize, maxBias);
    std::vector<float> zeros(maxBias, 0.0f);
    zeroBiasE   = gpu->createBuffer("zero_bias_E", cfg.nEmbd * 4);
    zeroBiasQKV = gpu->createBuffer("zero_bias_QKV", maxQkvOut * 4);
    zeroBiasGU  = gpu->createBuffer("zero_bias_GU", 2 * maxIntermediateSize * 4);
    zeroBiasV   = gpu->createBuffer("zero_bias_V", cfg.nVocab * 4);
    gpu->writeBuffer(zeroBiasE,   zeros.data(), cfg.nEmbd * 4);
    gpu->writeBuffer(zeroBiasQKV, zeros.data(), maxQkvOut * 4);
    gpu->writeBuffer(zeroBiasGU,  zeros.data(), 2 * maxIntermediateSize * 4);
    gpu->writeBuffer(zeroBiasV,   zeros.data(), cfg.nVocab * 4);

    // KV cache
    kvCache.resize(cfg.nLayer);
    uint64_t totalKvBytes = 0;
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& pl = cfg.perLayer[i];
        if (pl.kvSourceLayer >= 0) {
            // Shared KV: reuse source layer's buffers
            kvCache[i].K = kvCache[pl.kvSourceLayer].K;
            kvCache[i].V = kvCache[pl.kvSourceLayer].V;
            kvCache[i].len = 0;
        } else {
            uint32_t layerKvDim = pl.kvDim > 0 ? pl.kvDim : cfg.nKvHeads * cfg.headDim;
            uint64_t kvSize = (uint64_t)maxSeqLen * layerKvDim * 2;  // fp16
            kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
            kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
            kvCache[i].len = 0;
            totalKvBytes += kvSize * 2;
        }
    }
    fprintf(stderr, "  KV cache: %.0f MB (fp16, %u shared)\n",
            totalKvBytes / 1048576.0, cfg.sharedKvLayers);

    // Per-layer weights
    layerWeights.resize(cfg.nLayer);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        auto& ld = onnx.layers[i];

        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qkv", ld.qkv, lw.qkvW, lw.qkvS);
        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".o", ld.o, lw.oW, lw.oS);
        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".gu", ld.gateup, lw.guW, lw.guS);
        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".dn", ld.down, lw.dnW, lw.dnS);

        // Norm weights
        if (!ld.inputNorm.empty()) {
            lw.inputNorm = gpu->createBuffer("L" + std::to_string(i) + ".inorm",
                                              ld.inputNorm.size() * 4);
            gpu->writeBuffer(lw.inputNorm, ld.inputNorm.data(), ld.inputNorm.size() * 4);
        }
        if (!ld.postAttnNorm.empty()) {
            lw.postAttnNorm = gpu->createBuffer("L" + std::to_string(i) + ".panorm",
                                                 ld.postAttnNorm.size() * 4);
            gpu->writeBuffer(lw.postAttnNorm, ld.postAttnNorm.data(), ld.postAttnNorm.size() * 4);
        }
        // QK norm (optional)
        if (!ld.qNorm.empty()) {
            lw.qNorm = gpu->createBuffer("L" + std::to_string(i) + ".qnorm",
                                          ld.qNorm.size() * 4);
            gpu->writeBuffer(lw.qNorm, ld.qNorm.data(), ld.qNorm.size() * 4);
        }
        if (!ld.kNorm.empty()) {
            lw.kNorm = gpu->createBuffer("L" + std::to_string(i) + ".knorm",
                                          ld.kNorm.size() * 4);
            gpu->writeBuffer(lw.kNorm, ld.kNorm.data(), ld.kNorm.size() * 4);
        }

        if (i % 7 == 6 || i == cfg.nLayer - 1)
            fprintf(stderr, "  uploaded layer %u/%u\n", i + 1, cfg.nLayer);
    }

    // Final norm
    if (!onnx.finalNorm.empty()) {
        finalNormW = gpu->createBuffer("final_norm", onnx.finalNorm.size() * 4);
        gpu->writeBuffer(finalNormW, onnx.finalNorm.data(), onnx.finalNorm.size() * 4);
    }

    // Embedding
    embeddingCPU = std::move(onnx.embeddingCPU);
    fprintf(stderr, "  Embedding: %u × %u (fp32)\n", cfg.nVocab, cfg.nEmbd);

    // LM head
    if (onnx.hasLmHeadQ8) {
        uploadQ8Weight(*gpu, "lm_head_q8", onnx.lmHeadQ8, lmHeadQ8W, lmHeadQ8S);
        lmHeadIsQ8 = true;
        fprintf(stderr, "  LM head: separate (Q8)\n");
    } else if (cfg.tieWordEmbeddings) {
        // Build Q8 LM head from embedding table
        uint32_t nBlocksPerRow = (cfg.nEmbd + 31) / 32;
        size_t totalBlocks = (size_t)cfg.nVocab * nBlocksPerRow;
        struct Q8B { uint16_t d; int8_t qs[32]; };
        std::vector<Q8B> blocks(totalBlocks);
        for (uint32_t row = 0; row < cfg.nVocab; row++) {
            for (uint32_t blk = 0; blk < nBlocksPerRow; blk++) {
                auto& b = blocks[row * nBlocksPerRow + blk];
                float maxAbs = 0.0f;
                for (int q = 0; q < 32; q++) {
                    uint32_t col = blk * 32 + q;
                    float v = (col < cfg.nEmbd) ? embeddingCPU[row * cfg.nEmbd + col] : 0.0f;
                    maxAbs = std::max(maxAbs, std::abs(v));
                }
                float scale = maxAbs / 127.0f;
                float inv = (scale > 0) ? 1.0f / scale : 0.0f;
                uint32_t fb; memcpy(&fb, &scale, 4);
                uint32_t s16 = (fb >> 16) & 0x8000;
                int32_t e = ((fb >> 23) & 0xFF) - 112;
                uint32_t m = (fb >> 13) & 0x3FF;
                if (e <= 0) b.d = (uint16_t)s16;
                else if (e > 30) b.d = (uint16_t)(s16 | 0x7C00);
                else b.d = (uint16_t)(s16 | (e << 10) | m);
                for (int q = 0; q < 32; q++) {
                    uint32_t col = blk * 32 + q;
                    float v = (col < cfg.nEmbd) ? embeddingCPU[row * cfg.nEmbd + col] : 0.0f;
                    int iv = (int)roundf(v * inv);
                    iv = std::max(-128, std::min(127, iv));
                    b.qs[q] = (int8_t)iv;
                }
            }
        }
        auto rep = repack_q8_0(blocks.data(), cfg.nVocab, nBlocksPerRow * 32);
        uploadQ8Weight(*gpu, "lm_head_q8", rep, lmHeadQ8W, lmHeadQ8S);
        lmHeadIsQ8 = true;
        fprintf(stderr, "  LM head: tied embeddings (Q8)\n");
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "  Weights uploaded in %lldms\n", (long long)ms);

    // 3. RoPE tables
    if (hasPrecomputedRope && !onnx.ropeCos.empty()) {
        // Upload pre-computed ONNX cos/sin tables
        uint32_t ropeHalf = onnx.ropeHalfDim;
        uint32_t maxPos = std::min(onnx.ropeMaxPositions, maxSeqLen);
        uint64_t ropeBytes = (uint64_t)maxPos * ropeHalf * 4;
        ropeCosBuf = gpu->createBuffer("rope_cos", ropeBytes);
        ropeSinBuf = gpu->createBuffer("rope_sin", ropeBytes);
        // Only upload up to maxSeqLen positions
        if (onnx.ropeMaxPositions > maxSeqLen) {
            // Truncate
            gpu->writeBuffer(ropeCosBuf, onnx.ropeCos.data(), ropeBytes);
            gpu->writeBuffer(ropeSinBuf, onnx.ropeSin.data(), ropeBytes);
        } else {
            gpu->writeBuffer(ropeCosBuf, onnx.ropeCos.data(), ropeBytes);
            gpu->writeBuffer(ropeSinBuf, onnx.ropeSin.data(), ropeBytes);
        }
        fprintf(stderr, "  Using ONNX RoPE cache: %u positions × %u half-dim\n",
               maxPos, ropeHalf);
    }
    computeRopeTables();

    // 4. Build decode pipeline (identical to GGUF path from here)
    buildDecodePipeline();

    fprintf(stderr, "[onnx] loadOnnx complete\n"); fflush(stderr);
    return true;
}

// ─── Load weights ────────────────────────────────────────────────────────────

void ModelRunner::loadWeights(const GGUFFile& gguf,
                               const uint8_t* fileData) {
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "  Loading %llu tensors...\n", (unsigned long long)gguf.n_tensors);

    uint32_t qDim  = cfg.nHead * cfg.headDim;
    uint32_t kvDim = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOut = qDim + 2 * kvDim;
    // For MoE archs (qwen35moe) without dense FFN dim, use moe dim + generous
    // bound to cover qwen35moe attention temp buffers (qDim_actual up to 4*nEmbd).
    uint32_t effIntermediate = cfg.intermediateSize;
    if (effIntermediate == 0 && cfg.moeIntermediateSize > 0) {
        effIntermediate = std::max(cfg.moeIntermediateSize, cfg.moeSharedIntermediateSize);
        effIntermediate = std::max(effIntermediate, 4u * cfg.nEmbd);
    }
    uint32_t effQkvOut = qkvOut;
    // For qwen35moe attention dispatch: needs qOutDim=4*nEmbd for joint Q+gate.
    if (cfg.numExperts > 0 && cfg.fullAttentionInterval > 0) {
        effQkvOut = std::max(effQkvOut, 4u * cfg.nEmbd);
    }

    // Zero bias buffers
    uint32_t maxBias = std::max({cfg.nEmbd, effQkvOut,
                                 2 * effIntermediate, cfg.nVocab});
    fprintf(stderr, "  zero-bias sizing (GGUF): nEmbd=%u effQkvOut=%u effIntermediate=%u maxBias=%u\n",
            cfg.nEmbd, effQkvOut, effIntermediate, maxBias);
    std::vector<float> zeros(maxBias, 0.0f);
    zeroBiasE   = gpu->createBuffer("zero_bias_E", cfg.nEmbd * 4);
    zeroBiasQKV = gpu->createBuffer("zero_bias_QKV", effQkvOut * 4);
    zeroBiasGU  = gpu->createBuffer("zero_bias_GU", 2 * effIntermediate * 4);
    zeroBiasV   = gpu->createBuffer("zero_bias_V", cfg.nVocab * 4);
    gpu->writeBuffer(zeroBiasE,   zeros.data(), cfg.nEmbd * 4);
    gpu->writeBuffer(zeroBiasQKV, zeros.data(), effQkvOut * 4);
    gpu->writeBuffer(zeroBiasGU,  zeros.data(), 2 * effIntermediate * 4);
    gpu->writeBuffer(zeroBiasV,   zeros.data(), cfg.nVocab * 4);

    // KV cache (fp16 — halves attention bandwidth)
    kvCache.resize(cfg.nLayer);
    uint64_t kvSize = (uint64_t)maxSeqLen * cfg.nKvHeads * cfg.headDim * 2;  // 2 bytes per f16
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
        kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
        kvCache[i].len = 0;
    }
    fprintf(stderr, "  KV cache: %.0f MB (fp16)\n", cfg.nLayer * 2.0 * kvSize / 1048576.0);

    // Helper: load fp32/fp16 norm weight from GGUF tensor
    auto loadNorm = [&](const std::string& ggufName, GPUBuffer& buf) {
        auto it = gguf.tensor_index.find(ggufName);
        if (it == gguf.tensor_index.end()) return;
        auto& ti = gguf.tensors[it->second];
        const uint8_t* data = fileData + gguf.data_offset + ti.offset;
        uint32_t nel = 1;
        for (auto d : ti.shape) nel *= (uint32_t)d;
        std::vector<float> fp32(nel);
        if (ti.type == GGUF_TYPE_F16) {
            const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
            for (uint32_t j = 0; j < nel; j++)
                fp32[j] = fp16_to_f32(fp16[j]);
        } else {
            memcpy(fp32.data(), data, nel * 4);
        }
        buf = gpu->createBuffer(ggufName, nel * 4);
        gpu->writeBuffer(buf, fp32.data(), nel * 4);
    };

    // Helper: upload K-quant weight buffer
    auto uploadKQWeight = [&](const std::string& name, const KQuantPacked& kq,
                              GPUBuffer& buf) {
        uint64_t bytes = (uint64_t)kq.data.size() * 4;
        buf = gpu->createBuffer(name, bytes);
        gpu->writeBuffer(buf, kq.data.data(), bytes);
    };

    // Helper: repack any quantized tensor to Q8_0 format for GPU
    // For Q8_0: direct repack. For other types: dequant→fp32→quantize→Q8.
    auto repackToQ8 = [&](const uint8_t* data, uint32_t N, uint32_t K, GGUFType type) -> Q8Repacked {
        if (type == GGUF_TYPE_Q8_0) {
            return repack_q8_0(data, N, K);
        }
        // Dequantize to fp32 then quantize to Q8_0
        std::vector<float> fp32((size_t)N * K);
        dequant_tensor(data, fp32.data(), N, K, type);
        // Quantize fp32 → Q8_0 blocks then repack
        uint32_t nBlocksPerRow = K / 32;
        size_t totalBlocks = (size_t)N * nBlocksPerRow;
        std::vector<Q8_0Block> blocks(totalBlocks);
        for (uint32_t r = 0; r < N; r++) {
            for (uint32_t b = 0; b < nBlocksPerRow; b++) {
                auto& blk = blocks[r * nBlocksPerRow + b];
                float amax = 0;
                for (int j = 0; j < 32; j++) {
                    float v = fp32[r * K + b * 32 + j];
                    if (fabsf(v) > amax) amax = fabsf(v);
                }
                float d = amax / 127.0f;
                float id = (d != 0.0f) ? 1.0f / d : 0.0f;
                blk.d = f32_to_fp16(d);
                for (int j = 0; j < 32; j++) {
                    float v = fp32[r * K + b * 32 + j];
                    int q = (int)roundf(v * id);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    blk.qs[j] = (int8_t)q;
                }
            }
        }
        return repack_q8_0(blocks.data(), N, K);
    };

    // Helper: fuse multiple K-quant tensors vertically (concat rows)
    auto fuseKQ = [](const KQuantPacked& a, const KQuantPacked& b) -> KQuantPacked {
        KQuantPacked fused;
        fused.N = a.N + b.N;
        fused.K = a.K;
        fused.rowStrideWords = a.rowStrideWords;
        fused.nBlocks = a.nBlocks;
        fused.data.reserve(a.data.size() + b.data.size());
        fused.data.insert(fused.data.end(), a.data.begin(), a.data.end());
        fused.data.insert(fused.data.end(), b.data.begin(), b.data.end());
        return fused;
    };

    // Detect weight quantization type from first weight tensor
    // For mixed-quant models (Q4_K_M), use Q8 pipeline (dequant all to Q8)
    {
        auto it = gguf.tensor_index.find("blk.0.attn_q.weight");
        if (it == gguf.tensor_index.end())
            it = gguf.tensor_index.find("blk.0.attn_qkv.weight");
        if (it == gguf.tensor_index.end())
            it = gguf.tensor_index.find("blk.0.ffn_gate.weight");
        if (it == gguf.tensor_index.end())
            it = gguf.tensor_index.find("blk.0.ffn_down.weight");
        if (it != gguf.tensor_index.end()) {
            weightQuantType = (GGUFType)gguf.tensors[it->second].type;
        }
    }
    // Only use K-quant pipeline if ALL primary weight tensors are K-quant
    // Check multiple tensor types to detect mixed quantization
    bool allSameKQ = true;
    {
        const char* checkNames[] = {"blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"};
        for (auto name : checkNames) {
            auto it = gguf.tensor_index.find(name);
            if (it != gguf.tensor_index.end()) {
                auto t = (GGUFType)gguf.tensors[it->second].type;
                if (t != GGUF_TYPE_Q4_K && t != GGUF_TYPE_Q5_K && t != GGUF_TYPE_Q6_K) {
                    allSameKQ = false;
                    break;
                }
            }
        }
    }
    bool isKQuant = allSameKQ &&
                    (weightQuantType == GGUF_TYPE_Q4_K ||
                     weightQuantType == GGUF_TYPE_Q5_K ||
                     weightQuantType == GGUF_TYPE_Q6_K);
    auto packKQ = [&](const void* data, uint32_t N, uint32_t K) -> KQuantPacked {
        switch (weightQuantType) {
            case GGUF_TYPE_Q4_K: return pack_q4k(data, N, K);
            case GGUF_TYPE_Q5_K: return pack_q5k(data, N, K);
            case GGUF_TYPE_Q6_K: return pack_q6k(data, N, K);
            default: return {};
        }
    };
    const char* kqName = (weightQuantType == GGUF_TYPE_Q4_K) ? "Q4_K" :
                         (weightQuantType == GGUF_TYPE_Q5_K) ? "Q5_K" :
                         (weightQuantType == GGUF_TYPE_Q6_K) ? "Q6_K" : "Q8_0";
    fprintf(stderr, "  Weight format: %s\n", kqName);

    // Infer per-layer dimensions from tensor shapes (Gemma 4 has variable dims)
    cfg.perLayer.resize(cfg.nLayer);
    cfg.hasPerLayerDims = false;
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto pfx = "blk." + std::to_string(i) + ".";
        auto& pl = cfg.perLayer[i];
        pl.headDim = cfg.headDim;
        pl.qDim = cfg.nHead * cfg.headDim;
        pl.kvDim = cfg.nKvHeads * cfg.headDim;
        pl.intermediateSize = cfg.intermediateSize;
        pl.kvSourceLayer = -1;

        // Infer from Q tensor shape: Q=[E, qDim] or fused QKV=[E, qDim+2*kvDim]
        auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
        if (qi != gguf.tensor_index.end()) {
            auto& qt = gguf.tensors[qi->second];
            if (qt.shape.size() >= 2) {
                pl.qDim = (uint32_t)qt.shape[1];
                pl.headDim = pl.qDim / cfg.nHead;
            }
        } else {
            // Try fused QKV: qkv=[E, qDim + 2*kvDim]
            auto qkvi = gguf.tensor_index.find(pfx + "attn_qkv.weight");
            if (qkvi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qkvi->second];
                if (qt.shape.size() >= 2) {
                    // qkvOut = qDim + 2*kvDim; keep defaults since we can't decompose
                }
            }
        }
        auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
        if (ki != gguf.tensor_index.end()) {
            auto& kt = gguf.tensors[ki->second];
            if (kt.shape.size() >= 2)
                pl.kvDim = (uint32_t)kt.shape[1];
        }
        auto gi = gguf.tensor_index.find(pfx + "ffn_gate.weight");
        if (gi != gguf.tensor_index.end()) {
            auto& gt = gguf.tensors[gi->second];
            if (gt.shape.size() >= 2)
                pl.intermediateSize = (uint32_t)gt.shape[1];
        }

        if (i > 0 && (pl.qDim != cfg.perLayer[0].qDim ||
                      pl.kvDim != cfg.perLayer[0].kvDim ||
                      pl.intermediateSize != cfg.perLayer[0].intermediateSize))
            cfg.hasPerLayerDims = true;
    }

    // Shared KV mapping
    if (cfg.sharedKvLayers > 0) {
        uint32_t kvStart = cfg.nLayer - cfg.sharedKvLayers;
        // Find the last non-shared sliding and global layers
        int lastSliding = -1, lastGlobal = -1;
        for (uint32_t i = 0; i < kvStart; i++) {
            if (!cfg.layerAttnTypes.empty() && i < cfg.layerAttnTypes.size() &&
                cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow)
                lastSliding = (int)i;
            else
                lastGlobal = (int)i;
        }
        for (uint32_t i = kvStart; i < cfg.nLayer; i++) {
            bool isSWA = !cfg.layerAttnTypes.empty() && i < cfg.layerAttnTypes.size() &&
                         cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            cfg.perLayer[i].kvSourceLayer = isSWA ? lastSliding : lastGlobal;
        }
        fprintf(stderr, "  Shared KV: %u layers (from layer %u), sliding→L%d, global→L%d\n",
                cfg.sharedKvLayers, kvStart, lastSliding, lastGlobal);
    }

    if (cfg.hasPerLayerDims)
        fprintf(stderr, "  Variable per-layer dims detected\n");
    if (cfg.hasSandwichNorm)
        fprintf(stderr, "  Sandwich norms: 4 norms per layer\n");
    if (cfg.pleSize > 0)
        fprintf(stderr, "  PLE: dim=%u\n", cfg.pleSize);

    // Per-layer weights
    layerWeights.resize(cfg.nLayer);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto pfx = "blk." + std::to_string(i) + ".";
        auto& lw = layerWeights[i];
        auto& pl = cfg.perLayer[i];
        uint32_t layerQDim = pl.qDim;
        uint32_t layerKvDim = pl.kvDim;
        uint32_t layerQkvOut = layerQDim + 2 * layerKvDim;
        uint32_t layerIM = pl.intermediateSize;

        // qwen35moe attention layer: load Q/K/V SEPARATELY (no fuse) so Q can be
        // joint Q+gate (2x normal Q dim). Skip the fuse path below.
        bool isQ35AttnLayer = (cfg.numExperts > 0 && cfg.fullAttentionInterval > 0 &&
                               cfg.isAttentionLayer(i));
        if (isQ35AttnLayer) {
            auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
            auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
            auto vi = gguf.tensor_index.find(pfx + "attn_v.weight");
            if (qi != gguf.tensor_index.end() && ki != gguf.tensor_index.end() && vi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qi->second];
                auto& kt = gguf.tensors[ki->second];
                auto& vt = gguf.tensors[vi->second];
                // Native output dims from tensor shapes
                uint32_t qOutDim = (uint32_t)qt.shape[1];   // 2*qDim (joint Q+gate)
                uint32_t kOutDim = (uint32_t)kt.shape[1];   // kvDim (decoupled key_length)
                uint32_t vOutDim = (uint32_t)vt.shape[1];   // value_length
                auto qr = repackToQ8(fileData + gguf.data_offset + qt.offset, qOutDim, cfg.nEmbd, (GGUFType)qt.type);
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qj", qr, lw.qjW, lw.qjS);
                auto kr = repackToQ8(fileData + gguf.data_offset + kt.offset, kOutDim, cfg.nEmbd, (GGUFType)kt.type);
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".kSep", kr, lw.kSepW, lw.kSepS);
                auto vr = repackToQ8(fileData + gguf.data_offset + vt.offset, vOutDim, cfg.nEmbd, (GGUFType)vt.type);
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".vSep", vr, lw.vSepW, lw.vSepS);
                if (i == 3) {
                    fprintf(stderr, "  qwen35moe attn layer %u: Q(2x)=%u K=%u V=%u (E=%u)\n",
                            i, qOutDim, kOutDim, vOutDim, cfg.nEmbd);
                }
            }
        }

        // Fuse Q/K/V into single QKV (or load pre-fused attn_qkv.weight for archs that ship it)
        if (!isQ35AttnLayer) {
        {
            auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
            auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
            auto vi = gguf.tensor_index.find(pfx + "attn_v.weight");
            auto qkvi_pre = gguf.tensor_index.find(pfx + "attn_qkv.weight");
            if (qi == gguf.tensor_index.end() && qkvi_pre != gguf.tensor_index.end()) {
                // Pre-fused QKV (qwen35moe / some MoE archs)
                auto& qkvt = gguf.tensors[qkvi_pre->second];
                uint32_t qkvN = (uint32_t)qkvt.shape[1];  // out-dim = qDim + 2*kvDim
                bool tensorIsKQ = (qkvt.type == GGUF_TYPE_Q4_K || qkvt.type == GGUF_TYPE_Q5_K || qkvt.type == GGUF_TYPE_Q6_K);
                const uint8_t* src = fileData + gguf.data_offset + qkvt.offset;
                if (tensorIsKQ) {
                    KQuantPacked kq;
                    switch ((GGUFType)qkvt.type) {
                        case GGUF_TYPE_Q4_K: kq = pack_q4k(src, qkvN, cfg.nEmbd); break;
                        case GGUF_TYPE_Q5_K: kq = pack_q5k(src, qkvN, cfg.nEmbd); break;
                        case GGUF_TYPE_Q6_K: kq = pack_q6k(src, qkvN, cfg.nEmbd); break;
                        default: break;
                    }
                    if (i == 0) { kqQkvNBlocks = kq.nBlocks; kqQkvRowStride = kq.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".qkv_kq", kq, lw.qkvKQ);
                } else {
                    auto rep = repackToQ8(src, qkvN, cfg.nEmbd, (GGUFType)qkvt.type);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qkv", rep, lw.qkvW, lw.qkvS);
                }
                if (i == 0) {
                    fprintf(stderr, "  Pre-fused attn_qkv loaded: %u x %u, type=%u, %s path\n",
                            qkvN, cfg.nEmbd, (unsigned)qkvt.type, tensorIsKQ ? "K-quant" : "Q8");
                }
            } else if (qi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qi->second];
                auto& kt = gguf.tensors[ki->second];
                auto& vt = gguf.tensors[vi->second];
                bool allKQ = isKQuant &&
                    (qt.type == GGUF_TYPE_Q4_K || qt.type == GGUF_TYPE_Q5_K || qt.type == GGUF_TYPE_Q6_K) &&
                    (kt.type == GGUF_TYPE_Q4_K || kt.type == GGUF_TYPE_Q5_K || kt.type == GGUF_TYPE_Q6_K) &&
                    (vt.type == GGUF_TYPE_Q4_K || vt.type == GGUF_TYPE_Q5_K || vt.type == GGUF_TYPE_Q6_K);
                if (allKQ) {
                    auto qp = packKQ(fileData + gguf.data_offset + qt.offset, layerQDim, cfg.nEmbd);
                    auto kp = packKQ(fileData + gguf.data_offset + kt.offset, layerKvDim, cfg.nEmbd);
                    auto vp = packKQ(fileData + gguf.data_offset + vt.offset, layerKvDim, cfg.nEmbd);
                    auto fused = fuseKQ(fuseKQ(qp, kp), vp);
                    if (i == 0) { kqQkvNBlocks = fused.nBlocks; kqQkvRowStride = fused.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".qkv_kq", fused, lw.qkvKQ);
                } else {
                    auto qr = repackToQ8(fileData + gguf.data_offset + qt.offset, layerQDim, cfg.nEmbd, (GGUFType)qt.type);
                    auto kr = repackToQ8(fileData + gguf.data_offset + kt.offset, layerKvDim, cfg.nEmbd, (GGUFType)kt.type);
                    auto vr = repackToQ8(fileData + gguf.data_offset + vt.offset, layerKvDim, cfg.nEmbd, (GGUFType)vt.type);
                    Q8Repacked fused;
                    fused.N = layerQkvOut; fused.K = cfg.nEmbd;
                    fused.weights.reserve(qr.weights.size() + kr.weights.size() + vr.weights.size());
                    fused.weights.insert(fused.weights.end(), qr.weights.begin(), qr.weights.end());
                    fused.weights.insert(fused.weights.end(), kr.weights.begin(), kr.weights.end());
                    fused.weights.insert(fused.weights.end(), vr.weights.begin(), vr.weights.end());
                    fused.scales.reserve(qr.scales.size() + kr.scales.size() + vr.scales.size());
                    fused.scales.insert(fused.scales.end(), qr.scales.begin(), qr.scales.end());
                    fused.scales.insert(fused.scales.end(), kr.scales.begin(), kr.scales.end());
                    fused.scales.insert(fused.scales.end(), vr.scales.begin(), vr.scales.end());
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qkv", fused, lw.qkvW, lw.qkvS);
                }
            }
        }
        }  // close if (!isQ35AttnLayer)

        // O projection
        {
            auto it = gguf.tensor_index.find(pfx + "attn_output.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                bool tensorIsKQ = isKQuant && (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K || ti.type == GGUF_TYPE_Q6_K);
                if (tensorIsKQ) {
                    auto kq = packKQ(fileData + gguf.data_offset + ti.offset, cfg.nEmbd, layerQDim);
                    if (i == 0) { kqONBlocks = kq.nBlocks; kqORowStride = kq.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".o_kq", kq, lw.oKQ);
                } else {
                    auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                            cfg.nEmbd, layerQDim, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".o", rep, lw.oW, lw.oS);
                }
            }
        }

        // Fuse gate + up
        {
            auto gi = gguf.tensor_index.find(pfx + "ffn_gate.weight");
            auto ui = gguf.tensor_index.find(pfx + "ffn_up.weight");
            if (gi != gguf.tensor_index.end() && ui != gguf.tensor_index.end()) {
                auto& gt = gguf.tensors[gi->second];
                auto& ut = gguf.tensors[ui->second];
                bool bothKQ = isKQuant &&
                    (gt.type == GGUF_TYPE_Q4_K || gt.type == GGUF_TYPE_Q5_K || gt.type == GGUF_TYPE_Q6_K) &&
                    (ut.type == GGUF_TYPE_Q4_K || ut.type == GGUF_TYPE_Q5_K || ut.type == GGUF_TYPE_Q6_K);
                if (bothKQ) {
                    auto gp = packKQ(fileData + gguf.data_offset + gt.offset, layerIM, cfg.nEmbd);
                    auto up = packKQ(fileData + gguf.data_offset + ut.offset, layerIM, cfg.nEmbd);
                    auto fused = fuseKQ(gp, up);
                    if (i == 0) { kqGuNBlocks = fused.nBlocks; kqGuRowStride = fused.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".gu_kq", fused, lw.guKQ);
                } else {
                    auto gr = repackToQ8(fileData + gguf.data_offset + gt.offset,
                                           layerIM, cfg.nEmbd, (GGUFType)gt.type);
                    auto ur = repackToQ8(fileData + gguf.data_offset + ut.offset,
                                           layerIM, cfg.nEmbd, (GGUFType)ut.type);
                    Q8Repacked fused;
                    fused.N = 2 * layerIM; fused.K = cfg.nEmbd;
                    fused.weights.reserve(gr.weights.size() + ur.weights.size());
                    fused.weights.insert(fused.weights.end(), gr.weights.begin(), gr.weights.end());
                    fused.weights.insert(fused.weights.end(), ur.weights.begin(), ur.weights.end());
                    fused.scales.reserve(gr.scales.size() + ur.scales.size());
                    fused.scales.insert(fused.scales.end(), gr.scales.begin(), gr.scales.end());
                    fused.scales.insert(fused.scales.end(), ur.scales.begin(), ur.scales.end());
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".gu", fused, lw.guW, lw.guS);
                }
            }
        }

        // Down projection
        {
            auto it = gguf.tensor_index.find(pfx + "ffn_down.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                bool tensorIsKQ = isKQuant && (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K || ti.type == GGUF_TYPE_Q6_K);
                if (tensorIsKQ) {
                    auto kq = packKQ(fileData + gguf.data_offset + ti.offset, cfg.nEmbd, layerIM);
                    if (i == 0) { kqDnNBlocks = kq.nBlocks; kqDnRowStride = kq.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".dn_kq", kq, lw.dnKQ);
                } else {
                    auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                            cfg.nEmbd, layerIM, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".dn", rep, lw.dnW, lw.dnS);
                }
            }
        }

        // Norm weights
        loadNorm(pfx + "attn_norm.weight", lw.inputNorm);
        if (cfg.hasSandwichNorm) {
            // Gemma 4: 4-norm sandwich pattern
            loadNorm(pfx + "ffn_norm.weight", lw.ffnNorm);  // pre-FFN norm
            loadNorm(pfx + "post_norm.weight", lw.postNorm); // post-attention sandwich
            if (!lw.postNorm.handle)
                loadNorm(pfx + "attn_post_norm.weight", lw.postNorm);
            loadNorm(pfx + "post_ffw_norm.weight", lw.postFfwNorm); // post-FFN sandwich
            if (!lw.postFfwNorm.handle)
                loadNorm(pfx + "ffn_post_norm.weight", lw.postFfwNorm);
        } else {
            // Standard: 2-norm (pre-attn + pre-FFN fused with residual add)
            loadNorm(pfx + "ffn_norm.weight", lw.postAttnNorm);
            // Fallback for archs (qwen35moe / hybrid Mamba-MoE) that use
            // post_attention_norm.weight instead of ffn_norm.weight.
            if (!lw.postAttnNorm.handle)
                loadNorm(pfx + "post_attention_norm.weight", lw.postAttnNorm);
        }
        loadNorm(pfx + "attn_q_norm.weight", lw.qNorm);
        loadNorm(pfx + "attn_k_norm.weight", lw.kNorm);

        // PLE weights (Gemma 4)
        if (cfg.pleSize > 0) {
            auto loadWeight = [&](const std::string& name, uint32_t N, uint32_t K,
                                   GPUBuffer& wBuf, GPUBuffer& sBuf) {
                auto it = gguf.tensor_index.find(name);
                if (it != gguf.tensor_index.end()) {
                    auto& ti = gguf.tensors[it->second];
                    auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                           N, K, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, name, rep, wBuf, sBuf);
                }
            };
            loadWeight(pfx + "inp_gate.weight", cfg.pleSize, cfg.nEmbd,
                        lw.pleInpGateW, lw.pleInpGateS);
            loadWeight(pfx + "proj.weight", cfg.nEmbd, cfg.pleSize,
                        lw.pleProjW, lw.pleProjS);
            loadNorm(pfx + "per_layer_post_norm.weight", lw.plePostNorm);
            if (!lw.plePostNorm.handle)
                loadNorm(pfx + "post_norm_ple.weight", lw.plePostNorm);
        }

        // Per-layer output scale
        {
            auto it = gguf.tensor_index.find(pfx + "layer_output_scale.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                const uint8_t* data = fileData + gguf.data_offset + ti.offset;
                float scale;
                if (ti.type == GGUF_TYPE_F32) {
                    memcpy(&scale, data, 4);
                } else if (ti.type == GGUF_TYPE_F16) {
                    uint16_t h; memcpy(&h, data, 2);
                    uint32_t sign=(h>>15)&1, exp=(h>>10)&0x1F, mant=h&0x3FF, f;
                    if(exp==0)f=(sign<<31)|(mant<<13);
                    else if(exp==31)f=(sign<<31)|0x7F800000|(mant<<13);
                    else f=(sign<<31)|((exp+112)<<23)|(mant<<13);
                    memcpy(&scale, &f, 4);
                } else {
                    scale = 1.0f;
                }
                lw.outScale = gpu->createBuffer("L" + std::to_string(i) + ".out_scale", 4);
                gpu->writeBuffer(lw.outScale, &scale, 4);
            }
        }

        // ── MoE weights (qwen35moe and similar) ─────────────────────────────
        // Loads the small, always-resident MoE weights (router, shared expert,
        // attention gate). The large 3D routed-expert tensors are intentionally
        // NOT uploaded here — they need a native IQ GPU decode path (Phase 4)
        // because dequantizing them to Q8 would expand the model from 13 GiB
        // to ~31 GiB which won't fit on a 16 GiB GPU. A separate loader pass
        // (TODO) will keep them in their native IQ format as raw GPU bytes.
        if (cfg.numExperts > 0) {
            auto loadMoeSmall = [&](const std::string& name, uint32_t N, uint32_t K,
                                     GPUBuffer& wBuf, GPUBuffer& sBuf) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                       N, K, (GGUFType)ti.type);
                uploadQ8Weight(*gpu, name, rep, wBuf, sBuf);
                return true;
            };
            // Router: [nExperts, E]
            loadMoeSmall(pfx + "ffn_gate_inp.weight", cfg.numExperts, cfg.nEmbd,
                          lw.routerW, lw.routerS);
            // Shared-expert router (gating scalar) — shape varies; try common forms
            loadMoeSmall(pfx + "ffn_gate_inp_shexp.weight", 1, cfg.nEmbd,
                          lw.shexpRouterW, lw.shexpRouterS);
            // Shared expert (always active). IM_s = moeIntermediateSize for now;
            // the actual tensor shape will override if different (TODO: read shape).
            uint32_t IMs = cfg.moeSharedIntermediateSize;
            loadMoeSmall(pfx + "ffn_gate_shexp.weight", IMs, cfg.nEmbd,
                          lw.shexpGateW, lw.shexpGateS);
            loadMoeSmall(pfx + "ffn_up_shexp.weight",   IMs, cfg.nEmbd,
                          lw.shexpUpW,   lw.shexpUpS);
            loadMoeSmall(pfx + "ffn_down_shexp.weight", cfg.nEmbd, IMs,
                          lw.shexpDownW, lw.shexpDownS);
            // Attention gate (Qwen3.6 gated attention output)
            loadMoeSmall(pfx + "attn_gate.weight", cfg.nEmbd, cfg.nEmbd,
                          lw.attnGateW, lw.attnGateS);
            if (i == 0) {
                fprintf(stderr, "  MoE small weights loaded for layer 0 "
                        "(router%s, shexp%s, attn_gate%s)\n",
                        lw.routerW.handle ? "=ok" : "=MISSING",
                        lw.shexpGateW.handle ? "=ok" : "=MISSING",
                        lw.attnGateW.handle ? "=ok" : "=MISSING");
            }

            // ── Routed-expert weights (3D tensors, native IQ format on GPU) ─
            // Keep raw IQ bytes packed per-block (no dequant) so 13 GiB model
            // fits in 16 GiB VRAM. Shapes per llama.cpp / GGUF convention:
            //   ffn_gate_exps.weight: [nExperts, IM_e, E]
            //   ffn_up_exps.weight:   [nExperts, IM_e, E]
            //   ffn_down_exps.weight: [nExperts, E, IM_e]
            auto loadExpertsRaw = [&](const std::string& name,
                                       uint32_t rows_per_expert, uint32_t cols_per_expert,
                                       GPUBuffer& wBuf, uint32_t* outType) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                const uint8_t* src = fileData + gguf.data_offset + ti.offset;
                if (outType) *outType = (uint32_t)ti.type;
                uint32_t total_rows = cfg.numExperts * rows_per_expert;
                KQuantPacked packed;
                switch ((GGUFType)ti.type) {
                    case GGUF_TYPE_IQ3_S: packed = pack_iq3s(src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_IQ2_S: packed = pack_iq2s(src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_IQ4_XS:packed = pack_iq4xs(src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q2_K:  packed = pack_q2k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q3_K:  packed = pack_q3k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q4_K:  packed = pack_q4k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q5_K:  packed = pack_q5k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q6_K:  packed = pack_q6k (src, total_rows, cols_per_expert); break;
                    default:
                        fprintf(stderr, "  WARN: routed-expert quant type %u not handled for %s\n",
                                (unsigned)ti.type, name.c_str());
                        return false;
                }
                size_t bytes = (size_t)packed.data.size() * 4;
                wBuf = gpu->createBuffer(name, bytes);
                gpu->writeBuffer(wBuf, packed.data.data(), bytes);
                if (i == 0) {
                    fprintf(stderr, "    %s: %u rows × %u cols, type=%u, %.1f MB on GPU\n",
                            name.c_str(), total_rows, cols_per_expert,
                            (unsigned)ti.type, bytes / 1048576.0);
                }
                return true;
            };
            uint32_t IMe = cfg.moeIntermediateSize;
            if (i == 0) {
                moeExpertsGateType.resize(cfg.nLayer, 0);
                moeExpertsUpType.resize(cfg.nLayer, 0);
                moeExpertsDownType.resize(cfg.nLayer, 0);
            }
            loadExpertsRaw(pfx + "ffn_gate_exps.weight", IMe, cfg.nEmbd, lw.expertsGateW, &moeExpertsGateType[i]);
            loadExpertsRaw(pfx + "ffn_up_exps.weight",   IMe, cfg.nEmbd, lw.expertsUpW,   &moeExpertsUpType[i]);
            loadExpertsRaw(pfx + "ffn_down_exps.weight", cfg.nEmbd, IMe, lw.expertsDownW, &moeExpertsDownType[i]);
        }

        // ── SSM (Mamba) weights for hybrid archs ────────────────────────────
        // Loaded as raw fp16/Q8 buffers. Forward dispatch (selective scan + conv1d +
        // dt/alpha/beta projections) is a separate multi-week sub-project.
        // For hybrid archs: only SSM-only layers actually have these tensors —
        // attention layers (every Nth per full_attention_interval) skip.
        if (cfg.ssmInnerSize > 0 && !cfg.isAttentionLayer(i)) {
            auto loadRaw = [&](const std::string& name, GPUBuffer& buf) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                uint32_t nel = 1;
                for (auto d : ti.shape) nel *= (uint32_t)d;
                // Dequant to fp32 (small tensors; SSM weights aren't huge)
                std::vector<float> fp32((size_t)nel);
                dequant_tensor(fileData + gguf.data_offset + ti.offset,
                               fp32.data(), 1, nel, (GGUFType)ti.type);
                size_t bytes = (size_t)nel * 4;
                buf = gpu->createBuffer(name, bytes);
                gpu->writeBuffer(buf, fp32.data(), bytes);
                return true;
            };
            int loaded = 0;
            if (loadRaw(pfx + "ssm_conv1d.weight", lw.ssmConv1dW)) loaded++;
            if (loadRaw(pfx + "ssm_dt.bias",       lw.ssmDtBias))  loaded++;
            if (loadRaw(pfx + "ssm_a",             lw.ssmA))       loaded++;
            // Beta/alpha may be quantized; for now also dequant to fp32 (TODO: keep Q8)
            if (loadRaw(pfx + "ssm_beta.weight",   lw.ssmBetaW))   loaded++;
            if (loadRaw(pfx + "ssm_alpha.weight",  lw.ssmAlphaW))  loaded++;
            if (loadRaw(pfx + "ssm_norm.weight",   lw.ssmNorm))    loaded++;
            if (loadRaw(pfx + "ssm_out.weight",    lw.ssmOutW))    loaded++;
            if (i == 0) {
                fprintf(stderr, "  SSM weights loaded for layer 0: %d/7 tensors\n", loaded);
            }
        }

        if (i % 7 == 6 || i == cfg.nLayer - 1)
            fprintf(stderr, "  loaded layer %u/%u\n", i + 1, cfg.nLayer);
    }

    // Final norm
    loadNorm("output_norm.weight", finalNormW);

    // Embedding (dequantize to fp32 for CPU lookup)
    {
        auto it = gguf.tensor_index.find("token_embd.weight");
        if (it != gguf.tensor_index.end()) {
            auto& ti = gguf.tensors[it->second];
            uint32_t nel = 1;
            for (auto d : ti.shape) nel *= (uint32_t)d;
            const uint8_t* data = fileData + gguf.data_offset + ti.offset;
            embeddingCPU.resize(nel);
            if (ti.type == GGUF_TYPE_F16) {
                const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
                for (uint32_t j = 0; j < nel; j++)
                    embeddingCPU[j] = fp16_to_f32(fp16[j]);
            } else if (ti.type == GGUF_TYPE_Q8_0) {
                uint32_t rows = (uint32_t)ti.shape[1];
                uint32_t cols = (uint32_t)ti.shape[0];
                uint32_t nBlocks = cols / 32;
                const Q8_0Block* blocks = reinterpret_cast<const Q8_0Block*>(data);
                for (uint32_t r = 0; r < rows; r++) {
                    for (uint32_t b = 0; b < nBlocks; b++) {
                        const auto& blk = blocks[r * nBlocks + b];
                        float scale = fp16_to_f32(blk.d);
                        for (int q = 0; q < 32; q++)
                            embeddingCPU[r * cols + b * 32 + q] = (float)blk.qs[q] * scale;
                    }
                }
            } else if (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K ||
                       ti.type == GGUF_TYPE_Q6_K) {
                uint32_t rows = (uint32_t)ti.shape[1];
                uint32_t cols = (uint32_t)ti.shape[0];
                dequant_kquant(data, embeddingCPU.data(), rows, cols, (GGUFType)ti.type);
            } else {
                memcpy(embeddingCPU.data(), data, nel * 4);
            }
            fprintf(stderr, "  Embedding: %u × %u (%s)\n", cfg.nVocab, cfg.nEmbd,
                   ti.type == GGUF_TYPE_Q8_0 ? "Q8_0→f32" :
                   ti.type == GGUF_TYPE_Q4_K ? "Q4_K→f32" :
                   ti.type == GGUF_TYPE_Q5_K ? "Q5_K→f32" :
                   ti.type == GGUF_TYPE_Q6_K ? "Q6_K→f32" :
                   ti.type == GGUF_TYPE_F16 ? "f16→f32" : "f32");

            // LM head: use quantized format if embedding is quantized
            if (cfg.tieWordEmbeddings) {
                if (isKQuant && (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K ||
                                 ti.type == GGUF_TYPE_Q6_K)) {
                    // Upload as K-quant on GPU
                    auto kq = packKQ(data, cfg.nVocab, cfg.nEmbd);
                    kqLmNBlocks = kq.nBlocks;
                    kqLmRowStride = kq.rowStrideWords;
                    uploadKQWeight("lm_head_kq", kq, lmHeadKQ);
                    lmHeadIsKQ = true;
                    fprintf(stderr, "  LM head: tied embeddings (%s, %llu MB)\n", kqName,
                           (unsigned long long)(kq.data.size() * 4 / 1048576));
                } else if (ti.type == GGUF_TYPE_Q8_0) {
                    // Keep as Q8 on GPU — no dequant needed
                    auto rep = repack_q8_0(data, cfg.nVocab, cfg.nEmbd);
                    uploadQ8Weight(*gpu, "lm_head_q8", rep,
                                   lmHeadQ8W, lmHeadQ8S);
                    lmHeadIsQ8 = true;
                    uint64_t wBytes = (uint64_t)rep.weights.size() * 4;
                    uint64_t sBytes = (uint64_t)rep.scales.size() * 4;
                    fprintf(stderr, "  LM head: tied embeddings (Q8, %llu MB)\n",
                           (unsigned long long)((wBytes + sBytes) / 1048576));
                } else {
                    // fp16 fallback
                    std::vector<uint16_t> fp16(nel);
                    for (uint32_t j = 0; j < nel; j++)
                        fp16[j] = f32_to_fp16(embeddingCPU[j]);
                    uint64_t totalBytes = (uint64_t)nel * 2;
                    lmHeadW = gpu->createBuffer("lm_head_fp16", totalBytes);
                    const uint64_t CHUNK = 128 * 1024 * 1024;
                    for (uint64_t off = 0; off < totalBytes; off += CHUNK) {
                        uint64_t sz = std::min(CHUNK, totalBytes - off);
                        wgpuQueueWriteBuffer(gpu->queue, lmHeadW.handle, off,
                                             (const uint8_t*)fp16.data() + off, sz);
                    }
                    fprintf(stderr, "  LM head: tied embeddings (fp16, %llu MB)\n",
                           (unsigned long long)(totalBytes / 1048576));
                }
            }
        }
    }

    // ─── PLE global weight loading ──────────────────────────────────────
    if (cfg.pleSize > 0) {
        // per_layer_token_embd.weight — massive per-layer embedding table
        // Shape: [pleSize * nLayer, nVocab] — dequant to fp32 for CPU lookup
        auto it = gguf.tensor_index.find("per_layer_token_embd.weight");
        if (it != gguf.tensor_index.end()) {
            auto& ti = gguf.tensors[it->second];
            uint32_t rows = 1, cols = 1;
            if (ti.shape.size() >= 2) { cols = (uint32_t)ti.shape[0]; rows = (uint32_t)ti.shape[1]; }
            else if (ti.shape.size() == 1) { cols = (uint32_t)ti.shape[0]; }
            pleEmbCPU.resize((size_t)rows * cols);
            dequant_tensor(fileData + gguf.data_offset + ti.offset,
                           pleEmbCPU.data(), rows, cols, (GGUFType)ti.type);
            fprintf(stderr, "  PLE embedding: %u × %u (%zu MB fp32)\n",
                    rows, cols, (size_t)rows * cols * 4 / 1048576);
        }

        // per_layer_model_proj.weight — Q8 repack for GPU
        {
            auto it2 = gguf.tensor_index.find("per_layer_model_proj.weight");
            if (it2 != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it2->second];
                uint32_t N = (ti.shape.size() >= 2) ? (uint32_t)ti.shape[1] : 0;
                uint32_t K = (ti.shape.size() >= 2) ? (uint32_t)ti.shape[0] : 0;
                if (N > 0 && K > 0) {
                    auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                           N, K, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, "ple_model_proj", rep, pleModelProjW, pleModelProjS);
                }
            }
        }

        // per_layer_proj_norm.weight
        loadNorm("per_layer_proj_norm.weight", pleProjNormW);
    }

    // ─── MTP weight loading ──────────────────────────────────────────────
    if (cfg.hasMtp) {
        auto hasT = [&](const std::string& name) {
            return gguf.tensor_index.count(name) > 0;
        };

        if (hasT("blk.0.nextn.eh_proj.weight")) {
            mtpCfg.type = MTPType::Gemma4;
            uint32_t nMtp = 0;
            while (hasT("blk." + std::to_string(nMtp) + ".nextn.eh_proj.weight"))
                nMtp++;
            mtpCfg.numLayers = nMtp;
            mtpCfg.numDraftTokens = 1;
            fprintf(stderr, "  MTP: Gemma4 style, %u layers\n", nMtp);
        } else if (hasT("blk.0.nextn.enorm.weight")) {
            mtpCfg.type = MTPType::Qwen36;
            uint32_t nMtp = 0;
            while (hasT("blk." + std::to_string(nMtp) + ".nextn.enorm.weight"))
                nMtp++;
            mtpCfg.numLayers = nMtp;
            mtpCfg.numDraftTokens = 1;
            fprintf(stderr, "  MTP: Qwen3.6 style, %u layers\n", nMtp);
        } else {
            fprintf(stderr, "  MTP: detected but tensor naming not recognized\n");
            cfg.hasMtp = false;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "  Weights loaded in %lldms\n", (long long)ms);
}

// ─── RoPE tables ─────────────────────────────────────────────────────────────

void ModelRunner::computeRopeTables() {
    // For ONNX models with pre-computed RoPE tables, upload directly
    if (hasPrecomputedRope) return;  // already uploaded in loadOnnx()

    uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
    std::vector<float> cosTable(maxSeqLen * ropeHalf), sinTable(maxSeqLen * ropeHalf);
    for (uint32_t pos = 0; pos < maxSeqLen; pos++) {
        for (uint32_t i = 0; i < ropeHalf; i++) {
            float freq = 1.0f / powf(cfg.ropeTheta, (float)(2 * i) / (rotaryDim > 0 ? rotaryDim : cfg.headDim));
            float angle = pos * freq;
            cosTable[pos * ropeHalf + i] = cosf(angle);
            sinTable[pos * ropeHalf + i] = sinf(angle);
        }
    }
    ropeCosBuf = gpu->createBuffer("rope_cos", maxSeqLen * ropeHalf * 4);
    ropeSinBuf = gpu->createBuffer("rope_sin", maxSeqLen * ropeHalf * 4);
    gpu->writeBuffer(ropeCosBuf, cosTable.data(), maxSeqLen * ropeHalf * 4);
    gpu->writeBuffer(ropeSinBuf, sinTable.data(), maxSeqLen * ropeHalf * 4);
}

// ─── Build decode pipeline ───────────────────────────────────────────────────

// MoE FFN dispatch helper (Phase 3d stub).
// Returns false; appended dispatches happen in this function once wired.
bool ModelRunner::appendMoeFfnDispatches(uint32_t layerIdx, GPUBuffer xIn, GPUBuffer xOut) {
    (void)layerIdx; (void)xIn; (void)xOut;
    // Pre-flight: required tensors must be loaded.
    if (layerIdx >= layerWeights.size()) return false;
    auto& lw = layerWeights[layerIdx];
    if (!lw.routerW.handle || !lw.expertsGateW.handle ||
        !lw.expertsUpW.handle || !lw.expertsDownW.handle) {
        return false;
    }
    // TODO(Phase 3d): emit the per-token MoE FFN dispatch sequence:
    //   1. router matmul: xIn @ routerW.T → moeRouterOutBuf
    //   2. moe_gate (top-k softmax) → moeIndicesBuf / moeWeightsBuf
    //   3. for k in 0..numExpertsPerTok:
    //        idx = moeIndices[k], w = moeWeights[k]
    //        gate = xIn @ expertsGateW[idx]   (indirect IQ matmul w/ expert_row_off)
    //        up   = xIn @ expertsUpW[idx]     (indirect IQ matmul)
    //        act  = silu(gate) * up
    //        down = act @ expertsDownW[idx]   (indirect IQ matmul)
    //        xOut += w * down
    //   4. shared expert: xOut += silu(xIn@shexpGate)*shexpUp @ shexpDown
    return false;
}

void ModelRunner::buildDecodePipeline() {
    qDim = cfg.nHead * cfg.headDim;
    kvDim = cfg.nKvHeads * cfg.headDim;
    qkvOut = qDim + 2 * kvDim;
    const auto& limits = effectiveLimits(*gpu);
    const bool canUse512ThreadKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 512u &&
        limits.maxComputeWorkgroupStorageSize >= 32u * 1024u;
    const bool canUse256ThreadSubgroupKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 256u &&
        limits.maxComputeWorkgroupStorageSize >= 16u * 1024u;
    const bool decodeFastQ8Eligible =
        canUse256ThreadSubgroupKernels &&
        (cfg.nEmbd % 512u == 0u) &&
        (qDim % 512u == 0u);
    const bool decodeWideFp16Eligible = canUse256ThreadSubgroupKernels;
    uint32_t Q8_TILE = 8;
    uint32_t maxChunks = (maxSeqLen + gqaChunkSize - 1) / gqaChunkSize;

    // Load kernels from embedded shaders
    auto& plRmsNorm    = getKernel("rms_norm");
    auto& plAddRmsNorm = getKernel("add_rms_norm");
    auto& plQ8Matmul   = getKernel("q8_matmul");
    auto& plQ8MatmulNorm = getKernel("q8_matmul_norm");

    // K-quant kernel selection
    bool useKQ = (weightQuantType == GGUF_TYPE_Q4_K ||
                  weightQuantType == GGUF_TYPE_Q5_K ||
                  weightQuantType == GGUF_TYPE_Q6_K);
    const char* kqKernelName = (weightQuantType == GGUF_TYPE_Q4_K) ? "q4k_matmul" :
                               (weightQuantType == GGUF_TYPE_Q5_K) ? "q5k_matmul" :
                               (weightQuantType == GGUF_TYPE_Q6_K) ? "q6k_matmul" : nullptr;
    // Q4K uses 32-thread WG (1 col/WG), Q5K/Q6K use 256-thread WG (8 cols/WG)
    uint32_t kqTileN = (weightQuantType == GGUF_TYPE_Q4_K) ? 1u : 8u;
    const CompiledPipeline* plKQ = nullptr;
    if (useKQ && kqKernelName) {
        plKQ = &getKernel(kqKernelName);
    }

    const bool subgroupMatrixKernelReady =
        gpu->supportsSubgroupMatrix &&
        canUse512ThreadKernels &&
        canCompileEmbeddedKernel(*gpu, "test_subgroup_matrix");
    fprintf(stderr, "  Subgroup matrix: %s\n",
           subgroupMatrixKernelReady ? "available (i8×i8→i32 MMA)" : "not available");
    auto& plQ8Fast     = getKernel("q8_matmul_fast");
    auto& plFusedRope  = getKernelHD(cfg.hasQkNorm ? "fused_qknorm_rope" : "fused_rope");
    const bool useLargeHDAttn = (cfg.headDim > 128 && cfg.headDim % 32 == 0);
    auto& plChunkP1    = useLargeHDAttn ? getKernelHD("gqa_chunked_pass1_large_hd")
                                         : getKernelHD("gqa_chunked_pass1");
    auto& plChunkP2    = useLargeHDAttn ? getKernelHD("gqa_chunked_pass2_large_hd")
                                         : getKernelHD("gqa_chunked_pass2");
    auto& plFp16Gemm   = getKernel("fp16_gemm");
    auto& plFp16Wide   = getKernel("fp16_gemm_wide");
    auto& plArgmax     = getKernel("argmax");
    auto& plEmbGather  = getKernel("embed_gather");
    const bool useGelu = (cfg.activation == ActivationType::GELU);
    auto& plDownSilu   = useGelu ? getKernelGelu("q8_down_silu_add")
                                 : getKernel("q8_down_silu_add");

    tuning.decodeUseFastQkv = decodeFastQ8Eligible;
    tuning.decodeUseFastGateup = decodeFastQ8Eligible;
    tuning.decodeUseFastOproj = false;
    tuning.decodeUseWideFp16 = decodeWideFp16Eligible;
    decodeFastVariantsAvailable = decodeFastQ8Eligible && !useKQ;

    // Kernel selection per projection:
    auto& plQkv = tuning.decodeUseFastQkv ? plQ8Fast : plQ8Matmul;
    auto& plOp  = tuning.decodeUseFastOproj ? plQ8Fast : plQ8Matmul;
    auto& plGu  = tuning.decodeUseFastGateup ? plQ8Fast : plQ8Matmul;
    auto& plDnSilu = plDownSilu;

    useMMA = (gpu->backendType != WGPUBackendType_D3D12) && subgroupMatrixKernelReady;
    decodePoolCapacity = chooseDecodePoolDepth(*gpu);
    decodePoolDepth = decodePoolCapacity;
    decodeCbPoolBatch = chooseDecodeCbPoolBatch(*gpu, cfg);
        fprintf(stderr, "  Initial decode heuristic: qkv=%s oproj=%s gateup=%s lm_head=%s pool=%d batch=%d\n",
        tuning.decodeUseFastQkv ? "fast" : "base",
        tuning.decodeUseFastOproj ? "fast" : "base",
        tuning.decodeUseFastGateup ? "fast" : "base",
        tuning.decodeUseWideFp16 ? "fp16_wide" : "fp16",
        decodePoolDepth, decodeCbPoolBatch);

    // Static params (shared between both sets — read-only)
    auto makeQ8Params = [&](const std::string& name, uint32_t K, uint32_t N) -> GPUBuffer {
        uint32_t data[4] = {K, N, 0, 0};
        auto buf = gpu->createBuffer(name, 16);
        gpu->writeBuffer(buf, data, 16);
        return buf;
    };
    auto q8QkvParams   = makeQ8Params("p_qkv", cfg.nEmbd, qkvOut);
    GPUBuffer q8QkvNormParams;
    {
        uint32_t data[4] = {cfg.nEmbd, qkvOut, 0, 0};
        float eps = cfg.rmsNormEps; memcpy(&data[3], &eps, 4);
        q8QkvNormParams = gpu->createBuffer("p_qkv_norm", 16);
        gpu->writeBuffer(q8QkvNormParams, data, 16);
    }
    auto q8OprojParams = makeQ8Params("p_oproj", qDim, cfg.nEmbd);
    auto q8GuParams    = makeQ8Params("p_gu", cfg.nEmbd, 2 * cfg.intermediateSize);

    // Per-layer param buffers for variable-dim models
    std::vector<GPUBuffer> perLayerQkvParams, perLayerQkvNormParams;
    std::vector<GPUBuffer> perLayerOprojParams, perLayerGuParams, perLayerDnSiluParams;
    if (cfg.hasPerLayerDims) {
        perLayerQkvParams.resize(cfg.nLayer);
        perLayerQkvNormParams.resize(cfg.nLayer);
        perLayerOprojParams.resize(cfg.nLayer);
        perLayerGuParams.resize(cfg.nLayer);
        perLayerDnSiluParams.resize(cfg.nLayer);
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            auto& pl = cfg.perLayer[li];
            uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
            perLayerQkvParams[li] = makeQ8Params("p_qkv_" + std::to_string(li), cfg.nEmbd, plQkvOut);
            {
                uint32_t data[4] = {cfg.nEmbd, plQkvOut, 0, 0};
                float eps = cfg.rmsNormEps; memcpy(&data[3], &eps, 4);
                perLayerQkvNormParams[li] = gpu->createBuffer("p_qkv_norm_" + std::to_string(li), 16);
                gpu->writeBuffer(perLayerQkvNormParams[li], data, 16);
            }
            perLayerOprojParams[li] = makeQ8Params("p_oproj_" + std::to_string(li), pl.qDim, cfg.nEmbd);
            perLayerGuParams[li] = makeQ8Params("p_gu_" + std::to_string(li), cfg.nEmbd, 2 * pl.intermediateSize);
            {
                uint32_t data[4] = {pl.intermediateSize, cfg.nEmbd, pl.intermediateSize, 0};
                perLayerDnSiluParams[li] = gpu->createBuffer("p_dn_silu_" + std::to_string(li), 16);
                gpu->writeBuffer(perLayerDnSiluParams[li], data, 16);
            }
        }
    }
    // Fused down+silu params: [K=IM, N=E, IM, 0]
    GPUBuffer q8DnSiluParams;
    {
        uint32_t data[4] = {cfg.intermediateSize, cfg.nEmbd, cfg.intermediateSize, 0};
        q8DnSiluParams = gpu->createBuffer("p_dn_silu", 16);
        gpu->writeBuffer(q8DnSiluParams, data, 16);
    }

    // K-quant param buffers: [K, N, n_blocks, row_stride_words]
    auto makeKQParams = [&](const std::string& name, uint32_t K, uint32_t N,
                            uint32_t nBlocks, uint32_t rowStride) -> GPUBuffer {
        uint32_t data[4] = {K, N, nBlocks, rowStride};
        auto buf = gpu->createBuffer(name, 16);
        gpu->writeBuffer(buf, data, 16);
        return buf;
    };
    GPUBuffer kqQkvParams, kqOprojParams, kqGuParams, kqDnParams, kqLmParams;
    if (useKQ) {
        kqQkvParams   = makeKQParams("p_kq_qkv", cfg.nEmbd, qkvOut, kqQkvNBlocks, kqQkvRowStride);
        kqOprojParams = makeKQParams("p_kq_oproj", qDim, cfg.nEmbd, kqONBlocks, kqORowStride);
        kqGuParams    = makeKQParams("p_kq_gu", cfg.nEmbd, 2 * cfg.intermediateSize, kqGuNBlocks, kqGuRowStride);
        kqDnParams    = makeKQParams("p_kq_dn", cfg.intermediateSize, cfg.nEmbd, kqDnNBlocks, kqDnRowStride);
        if (lmHeadIsKQ)
            kqLmParams = makeKQParams("p_kq_lm", cfg.nEmbd, cfg.nVocab, kqLmNBlocks, kqLmRowStride);
    }

    GPUBuffer rmsParams;
    {
        uint32_t rn[4];
        rn[0] = cfg.nEmbd; rn[1] = cfg.nEmbd;
        float eps = cfg.rmsNormEps; memcpy(&rn[2], &eps, 4);
        rn[3] = 0;
        rmsParams = gpu->createBuffer("p_rms", 16);
        gpu->writeBuffer(rmsParams, rn, 16);
    }

    GPUBuffer lmheadParams;
    {
        uint32_t fp[4] = {cfg.nEmbd, cfg.nVocab, 0, 0};
        lmheadParams = gpu->createBuffer("p_lmhead", 16);
        gpu->writeBuffer(lmheadParams, fp, 16);
    }

    GPUBuffer argmaxParams;
    {
        uint32_t p[4] = {cfg.nVocab, 0, 0, 0};
        argmaxParams = gpu->createBuffer("p_argmax", 16);
        gpu->writeBuffer(argmaxParams, p, 16);
    }

    GPUBuffer embedParams;
    {
        uint32_t p[4] = {cfg.nEmbd, 0, 0, 0};
        embedParams = gpu->createBuffer("p_embed", 16);
        gpu->writeBuffer(embedParams, p, 16);
    }

    // Upload embedding table to GPU (shared)
    {
        uint64_t embBytes = (uint64_t)embeddingCPU.size() * 4;
        embeddingGpuBuf = gpu->createBuffer("embedding_gpu", embBytes);
        const uint64_t CHUNK = 128 * 1024 * 1024;
        for (uint64_t off = 0; off < embBytes; off += CHUNK) {
            uint64_t sz = std::min(CHUNK, embBytes - off);
            wgpuQueueWriteBuffer(gpu->queue, embeddingGpuBuf.handle, off,
                                 (const uint8_t*)embeddingCPU.data() + off, sz);
        }
    }

    // Initialize dynamic param templates
    ropeParamData.resize(32, 0);
    {
        auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
        uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        p[0] = cfg.nHead;  p[1] = qDim;  p[2] = kvDim;
        p[3] = 0;  p[4] = ropeHalf;  p[5] = 0;
        float eps = cfg.rmsNormEps;
        memcpy(&p[6], &eps, 4);
    }
    chunkedAttnParamData.resize(32, 0);
    {
        auto* p = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
        p[0] = cfg.nKvHeads * cfg.headDim;
        p[1] = cfg.nHead / cfg.nKvHeads;
        p[2] = 0;  p[3] = gqaChunkSize;  p[4] = 0;
        float scale = 1.0f / sqrtf((float)cfg.headDim);
        float neg_inf = -1e9f;
        memcpy(&p[5], &scale, 4);
        memcpy(&p[6], &neg_inf, 4);
        p[7] = maxChunks;
    }

    // Ensure layer norms exist (identity for models without QK norm)
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        if (!lw.qNorm.handle) {
            std::vector<float> ones(cfg.headDim, 1.0f);
            lw.qNorm = gpu->createBuffer("qnorm_id_" + std::to_string(i), cfg.headDim * 4);
            gpu->writeBuffer(lw.qNorm, ones.data(), cfg.headDim * 4);
        }
        if (!lw.kNorm.handle) {
            std::vector<float> ones(cfg.headDim, 1.0f);
            lw.kNorm = gpu->createBuffer("knorm_id_" + std::to_string(i), cfg.headDim * 4);
            gpu->writeBuffer(lw.kNorm, ones.data(), cfg.headDim * 4);
        }
    }

    // Shared argmax result buffer
    argmaxResultBuf = gpu->createBuffer("argmax_result", 4);

    // PLE buffers
    if (cfg.pleSize > 0) {
        pleBuf = gpu->createBuffer("ple_buf", cfg.pleSize * 4);
        pleOutBuf = gpu->createBuffer("ple_out", cfg.nEmbd * 4);
        pleSliceBufs.resize(cfg.nLayer);
        for (uint32_t li = 0; li < cfg.nLayer; li++)
            pleSliceBufs[li] = gpu->createBuffer("ple_slice_" + std::to_string(li), cfg.pleSize * 4);
    }

    // ─── Single set of intermediate buffers ───────────────────────────────
    // Sized to max per-layer dimensions for variable-dim models (Gemma 4).
    uint32_t maxQkvOutBuf = qkvOut;
    uint32_t maxQDimBuf = qDim;
    uint32_t maxIMBuf = cfg.intermediateSize;
    for (auto& pl : cfg.perLayer) {
        uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
        if (plQkvOut > maxQkvOutBuf) maxQkvOutBuf = plQkvOut;
        if (pl.qDim > maxQDimBuf) maxQDimBuf = pl.qDim;
        if (pl.intermediateSize > maxIMBuf) maxIMBuf = pl.intermediateSize;
    }
    xBuf          = gpu->createBuffer("x", cfg.nEmbd * 4);
    normOutBuf    = gpu->createBuffer("norm_out", cfg.nEmbd * 4);
    qkvBuf        = gpu->createBuffer("qkv_out", maxQkvOutBuf * 4);
    qRotBuf       = gpu->createBuffer("q_rot", maxQDimBuf * 4);
    attnOutBuf    = gpu->createBuffer("attn_out", maxQDimBuf * 4);
    projOutBuf    = gpu->createBuffer("proj_out", cfg.nEmbd * 4);
    gateUpBuf     = gpu->createBuffer("gate_up", 2 * maxIMBuf * 4);

    // MoE intermediate buffers (allocate when MoE arch detected)
    if (cfg.numExperts > 0) {
        moeRouterOutBuf  = gpu->createBuffer("moe_router_out", cfg.numExperts * 4);
        moeIndicesBuf    = gpu->createBuffer("moe_indices",    cfg.numExpertsPerTok * 4);
        moeWeightsBuf    = gpu->createBuffer("moe_weights",    cfg.numExpertsPerTok * 4);
        moeExpertOutBuf  = gpu->createBuffer("moe_expert_out", cfg.nEmbd * 4);
        moeShexpGateUpBuf= gpu->createBuffer("moe_shexp_gu",   2 * cfg.moeIntermediateSize * 4);
        moeShexpActBuf   = gpu->createBuffer("moe_shexp_act",  cfg.moeIntermediateSize * 4);
        moeRoutedGateBuf = gpu->createBuffer("moe_routed_gate", cfg.moeIntermediateSize * 4);
        moeRoutedUpBuf   = gpu->createBuffer("moe_routed_up",   cfg.moeIntermediateSize * 4);
        moeRoutedActBuf  = gpu->createBuffer("moe_routed_act",  cfg.moeIntermediateSize * 4);
        fprintf(stderr, "  MoE intermediate buffers allocated: router=%uB indices=%uB weights=%uB expert_out=%uB shexp_gu=%uB shexp_act=%uB routed=3x%uB\n",
                cfg.numExperts * 4u, cfg.numExpertsPerTok * 4u, cfg.numExpertsPerTok * 4u,
                cfg.nEmbd * 4u, 2 * cfg.moeIntermediateSize * 4u, cfg.moeIntermediateSize * 4u,
                cfg.moeIntermediateSize * 4u);
    }
    // SSM persistent buffers per layer (only for SSM layers in hybrid archs)
    if (cfg.ssmInnerSize > 0) {
        ssmConvState.resize(cfg.nLayer);
        ssmHState.resize(cfg.nLayer);
        size_t convBytes = (size_t)cfg.ssmInnerSize * cfg.ssmConvKernel * 4;
        size_t hBytes    = (size_t)cfg.ssmInnerSize * cfg.ssmStateSize * 4;
        uint32_t ssmLayerCount = 0;
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            if (cfg.isAttentionLayer(li)) continue;
            ssmConvState[li] = gpu->createBuffer("ssm_conv_state_L" + std::to_string(li), convBytes);
            ssmHState[li]    = gpu->createBuffer("ssm_h_state_L"    + std::to_string(li), hBytes);
            ssmLayerCount++;
        }
        fprintf(stderr, "  SSM state buffers: %u SSM layers x (conv=%zuB + h=%zuB) = %zu MB total\n",
                ssmLayerCount, convBytes, hBytes,
                (size_t)ssmLayerCount * (convBytes + hBytes) / (1024 * 1024));
    }

    // qwen35moe attention intermediate buffers (sized for actual Q/K/V dims)
    if (cfg.numExperts > 0 && cfg.fullAttentionInterval > 0) {
        // Discover sizes from first attention layer's qjW buffer.
        // qjW is sized [qOutDim * cfg.nEmbd] in Q8 (1 byte/elem + scale per 32);
        // qOutDim = 2 * qDim_actual. We use a generous bound: 2 * 4 * cfg.nEmbd
        // (covers qOutDim up to 4x nEmbd, fits qwen35moe's 8192 with nEmbd=2048).
        uint32_t maxQjDim = 4u * cfg.nEmbd;          // bound on 2*qDim_actual
        uint32_t maxQDim  = maxQjDim / 2u;
        uint32_t maxKvDim = 2u * cfg.nEmbd;          // bound on kvDim_actual
        q35QjBuf      = gpu->createBuffer("q35_qj", maxQjDim * 4);
        q35QBuf       = gpu->createBuffer("q35_q",  maxQDim * 4);
        q35GateBuf    = gpu->createBuffer("q35_gate", maxQDim * 4);
        q35KBuf       = gpu->createBuffer("q35_k",  maxKvDim * 4);
        q35VBuf       = gpu->createBuffer("q35_v",  maxKvDim * 4);
        q35AttnOutBuf = gpu->createBuffer("q35_attn_out", maxQDim * 4);
        // Cos/sin table for MRoPE: 4 sections × max 16 pairs × 2 (cos,sin) = 128 floats
        q35CosSinBuf  = gpu->createBuffer("q35_cossin", 128 * 4);
        fprintf(stderr, "  qwen35moe attn buffers: qj=%uB q=%uB gate=%uB k=%uB v=%uB attn=%uB\n",
                maxQjDim*4u, maxQDim*4u, maxQDim*4u, maxKvDim*4u, maxKvDim*4u, maxQDim*4u);
    }

    // K-quant needs a separate SiLU-mul output buffer (used for down proj input)
    GPUBuffer siluMulOutBuf;
    if (useKQ) {
        siluMulOutBuf = gpu->createBuffer("silu_mul_out", maxIMBuf * 4);
    }

    rstdBuf       = gpu->createBuffer("rstd", 16);
    logitsBuf     = gpu->createBuffer("logits", cfg.nVocab * 4);
    attnPartialsBuf = gpu->createBuffer("attn_partials",
        cfg.nHead * maxChunks * (cfg.headDim + 2) * 4);

    // Single set of dynamic params (writeBuffer is queue-sequenced)
    fusedRopeParamsBuf = gpu->createBuffer("p_frope", 32);
    gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);
    chunkedAttnParamsBuf = gpu->createBuffer("p_cattn", 32);
    gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);

    // Sliding window attention: separate param buffer with clamped T_total
    bool hasSWA = !cfg.layerAttnTypes.empty() && cfg.slidingWindow > 0;
    if (hasSWA) {
        chunkedAttnParamsBufSWA = gpu->createBuffer("p_cattn_swa", 32);
        gpu->writeBuffer(chunkedAttnParamsBufSWA, chunkedAttnParamData.data(), 32);
    }

    // ─── Build dispatch list (single — identical for every token) ─────────
    allDecodeDispatches.reserve(cfg.nLayer * 11 + 2);
    decodeDispatchIndices.assign(cfg.nLayer, {});
    decodeVariantBGs.assign(cfg.nLayer, {});
    ropeDispatchIndices.resize(cfg.nLayer);
    attnP1DispatchIndices.resize(cfg.nLayer);
    attnP2DispatchIndices.resize(cfg.nLayer);

    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        auto& di = decodeDispatchIndices[i];
        auto& vbg = decodeVariantBGs[i];
        std::string L = "L" + std::to_string(i) + "/";

        // ── DeltaNet layer passthrough (qwen35moe linear-attention layers) ──
        // CORRECTNESS NOTE: qwen35moe's "SSM" layers are actually DeltaNet
        // (linear attention with delta rule, NOT standard Mamba). My earlier
        // implementation using Mamba selective_scan was architecturally wrong.
        // Reverting to passthrough until proper DeltaNet kernels exist:
        //
        // DeltaNet forward (per layer):
        //   qkvz   = qkvz_proj(x)               -- combined Q/K/V/Z projection
        //   beta   = sigmoid(beta_proj(x))      -- delta gate
        //   alpha  = softplus(alpha_proj(x) + ssm_dt)
        //   gate   = exp(-A_log) * alpha        -- ssm_a is small [dt_rank]
        //   conv_out = silu(ssm_conv(qkv_mixed, conv1d))
        //   q, k, v = extract from conv_out
        //   state' = state * exp(gate) + beta * (k ⊗ v)   -- matrix-state recurrence
        //   y      = q ⊗ state'                            -- matrix-state @ vec
        //   y      = ssm_norm * y * silu(z)
        //   out    = ssm_out @ y                           -- residual add
        //
        // Reference: github.com/ggml-org/llama.cpp src/models/qwen35.cpp
        //   `build_layer_attn_linear` + `build_delta_net_*`
        //
        // Needed but NOT in backpack today: matrix-state recurrence kernel,
        // proper Q/K/V/Z extraction from conv output, delta rule kernel.
        // (~1-2 weeks of focused work to implement correctly.)
        //
        // Also: attention layers use multi-axis RoPE (MRoPE) which backpack
        // doesn't support either — those layers' attention is also wrong.
        if (cfg.ssmInnerSize > 0 && !cfg.isAttentionLayer(i)) {
            if (i == 0) {
                fprintf(stderr,
                    "  DeltaNet layers passthrough (%u layers). Output WILL BE WRONG.\n"
                    "  Proper DeltaNet impl needs matrix-state recurrence kernel.\n",
                    cfg.nLayer - cfg.nLayer / cfg.fullAttentionInterval);
            }
            continue;
        }

        // ── qwen35moe attention-layer dispatch (correctness wiring) ────────
        // Replaces the standard fused-QKV path for qwen35moe attention layers.
        // Output not yet validated against llama.cpp; bugs expected.
        // (Note: this dispatches ALONGSIDE the dense path below, so residual
        //  is added twice. Refactoring to skip the dense path requires
        //  wrapping ~400 lines in if/else; for now we live with the duplication.)
        if (cfg.numExperts > 0 && cfg.fullAttentionInterval > 0 && cfg.isAttentionLayer(i) &&
            layerWeights[i].qjW.handle && layerWeights[i].kSepW.handle && layerWeights[i].vSepW.handle) {
            auto& lwQ35 = layerWeights[i];
            if (i == 3) {
                fprintf(stderr,
                    "  qwen35moe attn dispatch wired for layer %u\n"
                    "  WARN: attention compute uses Q passthrough placeholder. Output INCORRECT.\n", i);
            }
            auto mkP35 = [&](const std::string& name, std::initializer_list<uint32_t> data) -> GPUBuffer {
                uint32_t buf[8] = {0};
                size_t i2 = 0; for (uint32_t v : data) { buf[i2++] = v; }
                auto b = gpu->createBuffer(name, 32);
                gpu->writeBuffer(b, buf, 32);
                return b;
            };

            // Discovered actual dims (qwen35moe IQ3_XXS): qOutDim=8192, qDim=4096,
            // nHead=16, headDim=256, kvDim=512, nKvHeads=2.
            uint32_t qOutDim = 4u * cfg.nEmbd;   // 8192
            uint32_t qDimAct = qOutDim / 2u;     // 4096
            uint32_t headDimAct = qDimAct / cfg.nHead;  // 256
            uint32_t kvDimAct = cfg.nKvHeads * headDimAct;  // 512

            // 2. Q matmul: normOutBuf @ qjW → q35QjBuf
            {
                auto p = mkP35("p_q35_q_L"+std::to_string(i), {cfg.nEmbd, qOutDim});
                auto bg = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lwQ35.qjW}, {2, lwQ35.qjS},
                    {3, zeroBiasV}, {4, q35QjBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (qOutDim + Q8_TILE - 1) / Q8_TILE, 1, L+"q35_q"});
            }
            // 3. K matmul
            {
                auto p = mkP35("p_q35_k_L"+std::to_string(i), {cfg.nEmbd, kvDimAct});
                auto bg = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lwQ35.kSepW}, {2, lwQ35.kSepS},
                    {3, zeroBiasV}, {4, q35KBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (kvDimAct + Q8_TILE - 1) / Q8_TILE, 1, L+"q35_k"});
            }
            // 4. V matmul
            {
                auto p = mkP35("p_q35_v_L"+std::to_string(i), {cfg.nEmbd, kvDimAct});
                auto bg = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lwQ35.vSepW}, {2, lwQ35.vSepS},
                    {3, zeroBiasV}, {4, q35VBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (kvDimAct + Q8_TILE - 1) / Q8_TILE, 1, L+"q35_v"});
            }
            // 5. attn_split_qg: q35QjBuf → q35QBuf + q35GateBuf
            {
                auto& plSplit = getKernel("attn_split_qg");
                auto p = mkP35("p_q35_split_L"+std::to_string(i), {cfg.nHead, headDimAct});
                auto bg = makeBG(plSplit, {
                    {0, q35QjBuf}, {1, q35QBuf}, {2, q35GateBuf}, {3, p}});
                allDecodeDispatches.push_back({plSplit.pipeline, bg,
                    (qDimAct + 255) / 256, 1, 1, L+"q35_split"});
            }
            // 6. Per-head RMSNorm on Q and K
            if (lwQ35.qNorm.handle) {
                auto& plNorm = getKernel("attn_head_rmsnorm");
                uint32_t epsBits; float eps = cfg.rmsNormEps; memcpy(&epsBits, &eps, 4);
                auto p = mkP35("p_q35_qnorm_L"+std::to_string(i), {cfg.nHead, headDimAct, epsBits});
                auto bg = makeBG(plNorm, {{0, q35QBuf}, {1, lwQ35.qNorm}, {2, p}});
                allDecodeDispatches.push_back({plNorm.pipeline, bg, cfg.nHead, 1, 1, L+"q35_qnorm"});
            }
            if (lwQ35.kNorm.handle) {
                auto& plNorm = getKernel("attn_head_rmsnorm");
                uint32_t epsBits; float eps = cfg.rmsNormEps; memcpy(&epsBits, &eps, 4);
                auto p = mkP35("p_q35_knorm_L"+std::to_string(i), {cfg.nKvHeads, headDimAct, epsBits});
                auto bg = makeBG(plNorm, {{0, q35KBuf}, {1, lwQ35.kNorm}, {2, p}});
                allDecodeDispatches.push_back({plNorm.pipeline, bg, cfg.nKvHeads, 1, 1, L+"q35_knorm"});
            }
            // 7. MRoPE — TODO: need per-decode cos/sin precompute. Skipped for now.

            // 8. Attention compute — PLACEHOLDER: copy Q → attn_out (no real attn).
            {
                auto& plCopy = getKernel("shared_copy_buffer");
                auto p = mkP35("p_q35_attn_ph_L"+std::to_string(i), {qDimAct, 0u, 0u});
                auto bg = makeBG(plCopy, {{0, q35QBuf}, {1, q35AttnOutBuf}, {2, p}});
                allDecodeDispatches.push_back({plCopy.pipeline, bg,
                    (qDimAct + 255) / 256, 1, 1, L+"q35_attn_ph"});
            }
            // 9. attn_gated_output
            {
                auto& plGate = getKernel("attn_gated_output");
                auto p = mkP35("p_q35_gate_L"+std::to_string(i), {qDimAct});
                auto bg = makeBG(plGate, {
                    {0, q35AttnOutBuf}, {1, q35GateBuf}, {2, q35AttnOutBuf}, {3, p}});
                allDecodeDispatches.push_back({plGate.pipeline, bg,
                    (qDimAct + 255) / 256, 1, 1, L+"q35_gated"});
            }
            // 10. wo matmul + residual (uses existing lw.oW from dense loader)
            if (lwQ35.oW.handle) {
                auto p = mkP35("p_q35_oproj_L"+std::to_string(i), {qDimAct, cfg.nEmbd});
                auto bg = makeBG(plQ8Matmul, {
                    {0, q35AttnOutBuf}, {1, lwQ35.oW}, {2, lwQ35.oS},
                    {3, zeroBiasE}, {4, projOutBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q35_oproj"});

                auto& plWAcc = getKernel("moe_weighted_accumulate_decode");
                auto pAdd = mkP35("p_q35_add_L"+std::to_string(i), {cfg.nEmbd, 0u});
                GPUBuffer one = gpu->createBuffer("q35_one_L"+std::to_string(i), 4);
                float oneF = 1.0f; gpu->writeBuffer(one, &oneF, 4);
                auto bgAdd = makeBG(plWAcc,
                    {{0, xBuf}, {1, projOutBuf}, {2, one}, {3, pAdd}});
                allDecodeDispatches.push_back({plWAcc.pipeline, bgAdd,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"q35_attn_add"});
            }
        }

        // ── qwen35moe attention-layer correctness wiring (WIP scaffold) ────
        // For qwen35moe attention layers, the dense Q dispatch is JOINT Q+gate
        // (output is 2x qDim per head: first half = Q, second half = gate).
        // We need split_qg → QK-norm on Q → MRoPE on Q/K → attention →
        // attn_out * sigmoid(gate) → wo → residual.
        //
        // The existing dense path already runs (incorrect output). This block
        // emits ADDITIONAL split/MRoPE/gated_output dispatches as scaffolding;
        // full correctness requires replacing the dense path entirely (the
        // existing QKV fuse + standard RoPE produce wrong intermediate state).
        //
        // NOT YET WIRED — kept as a documentation comment until the dense
        // path can be cleanly replaced with the correct qwen35moe-specific
        // attention sequence. Wiring this naively (adding to dense) would
        // double-rotate and produce worse output than just running dense.
        //
        // Wiring steps needed:
        //   1. Skip backpack's fuse-Q/K/V loader for qwen35moe attn layers
        //      (Q is 2x sized; fuse pack assumes Q sized = qDim)
        //   2. Emit 3 separate matmul dispatches: Q (2x), K, V
        //   3. attn_split_qg: split Q+gate from joint matmul output
        //   4. QK-norm on Q half, K-norm on K
        //   5. attn_rope_multi on Q and K (replace standard RoPE)
        //   6. Reuse existing attention compute (flash attention OK)
        //   7. attn_gated_output: attn_out * sigmoid(gate)
        //   8. wo matmul → projOutBuf (existing dense oproj path works)
        //   9. Residual add to xBuf
        //
        // TODO: skip dense attention for qwen35moe attn layers. Naive brace
        // wrap scopes `pl`/`plIM`/`layerDnP` away from MoE FFN dispatch.
        // Real fix needs variable-hoist refactor or splitting MoE FFN into
        // a helper that takes pl by value. For now both dense AND new q35
        // dispatches run — Dawn flags null qkvW/qkvKQ binds but the run
        // continues with garbage attention output.

        auto& pl = cfg.perLayer[i];
        uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
        uint32_t plQDim = pl.qDim;
        uint32_t plIM = pl.intermediateSize;
        auto& layerQkvP = cfg.hasPerLayerDims ? perLayerQkvParams[i] : q8QkvParams;
        auto& layerQkvNP = cfg.hasPerLayerDims ? perLayerQkvNormParams[i] : q8QkvNormParams;
        auto& layerOpP = cfg.hasPerLayerDims ? perLayerOprojParams[i] : q8OprojParams;
        auto& layerGuP = cfg.hasPerLayerDims ? perLayerGuParams[i] : q8GuParams;
        auto& layerDnP = cfg.hasPerLayerDims ? perLayerDnSiluParams[i] : q8DnSiluParams;

        // Shared KV: attention reads from source layer's cache
        uint32_t kvCacheLayer = (pl.kvSourceLayer >= 0) ? (uint32_t)pl.kvSourceLayer : i;

        // 1. RMSNorm (only for KQ path — Q8 path uses fused q8_matmul_norm)
        if (i == 0 && useKQ) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.inputNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_norm"});
        }

        // 2. QKV matmul
        {
            if (useKQ) {
                auto bg = makeBG(*plKQ, {
                    {0, normOutBuf}, {1, lw.qkvKQ}, {2, zeroBiasQKV},
                    {3, qkvBuf}, {4, kqQkvParams}});
                di.qkv = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plKQ->pipeline, bg,
                    1, (plQkvOut + 7) / 8, 1, L+"kq_qkv"});
            } else {
                vbg.qkvBase = makeBG(plQ8MatmulNorm, {
                    {0, xBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                    {3, zeroBiasQKV}, {4, qkvBuf}, {5, layerQkvNP},
                    {6, lw.inputNorm}});
                di.qkv = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plQ8MatmulNorm.pipeline, vbg.qkvBase,
                    1, (plQkvOut + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_qkv"});
            }
        }

        {
            auto bg = makeBG(plFusedRope, {
                {0, qkvBuf}, {1, qRotBuf},
                {2, kvCache[kvCacheLayer].K}, {3, kvCache[kvCacheLayer].V},
                {4, ropeCosBuf}, {5, ropeSinBuf},
                {6, lw.qNorm}, {7, lw.kNorm},
                {8, fusedRopeParamsBuf}});
            ropeDispatchIndices[i] = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({plFusedRope.pipeline, bg,
                cfg.nHead + cfg.nKvHeads, 1, 1, L+"fused_rope"});
        }

        {
            bool isSWA = hasSWA && i < cfg.layerAttnTypes.size() &&
                         cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            auto& attnParams = isSWA ? chunkedAttnParamsBufSWA : chunkedAttnParamsBuf;
            auto bg = makeBG(plChunkP1, {
                {0, qRotBuf}, {1, kvCache[kvCacheLayer].K}, {2, kvCache[kvCacheLayer].V},
                {3, attnPartialsBuf}, {4, attnParams}});
            attnP1DispatchIndices[i] = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({plChunkP1.pipeline, bg,
                cfg.nHead, maxChunks, 1, L+"attn_p1"});
        }

        {
            bool isSWA2 = hasSWA && i < cfg.layerAttnTypes.size() &&
                          cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            auto& attnParams2 = isSWA2 ? chunkedAttnParamsBufSWA : chunkedAttnParamsBuf;
            auto bg = makeBG(plChunkP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, attnParams2}});
            attnP2DispatchIndices[i] = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({plChunkP2.pipeline, bg,
                cfg.nHead, 1, 1, L+"attn_p2"});
        }

        {
            if (useKQ) {
                auto bg = makeBG(*plKQ, {
                    {0, attnOutBuf}, {1, lw.oKQ}, {2, zeroBiasE},
                    {3, projOutBuf}, {4, kqOprojParams}});
                di.oproj = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plKQ->pipeline, bg,
                    1, (cfg.nEmbd + kqTileN - 1) / kqTileN, 1, L+"kq_oproj"});
            } else {
                vbg.oprojBase = makeBG(plQ8Matmul, {
                    {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                    {3, zeroBiasE}, {4, projOutBuf}, {5, layerOpP}});
                if (decodeFastVariantsAvailable) {
                    vbg.oprojFast = makeBG(plQ8Fast, {
                        {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                        {3, zeroBiasE}, {4, projOutBuf}, {5, layerOpP}});
                }
                auto bg = tuning.decodeUseFastOproj && vbg.oprojFast ? vbg.oprojFast : vbg.oprojBase;
                auto pipeline = tuning.decodeUseFastOproj && vbg.oprojFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
                di.oproj = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({pipeline, bg,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_oproj"});
            }
        }

        // Post-attention: either fused add+norm (standard) or sandwich norms (Gemma 4)
        if (cfg.hasSandwichNorm && lw.postNorm.handle) {
            // Sandwich: RMSNorm(projOutBuf) in-place, then xBuf += projOutBuf, then norm for FFN
            static const char* RMS_NORM_INPLACE_WGSL = R"(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0];
    let eps = bitcast<f32>(_p_[1]);
    let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var i = tid; i < N; i += 256u) { let v = X[i]; sum_sq += v * v; }
    let warp_sum = subgroupAdd(sum_sq);
    let total = subgroupBroadcastFirst(warp_sum);
    let rms = 1.0 / sqrt(total / f32(N) + eps);
    for (var i = tid; i < N; i += 256u) { X[i] = X[i] * rms * W[i]; }
}
)";
            auto& plRmsIP = gpu->getOrCreatePipeline("rms_norm_inplace",
                std::string(RMS_NORM_INPLACE_WGSL), 3);
            GPUBuffer rmsIPParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                rmsIPParams = gpu->createBuffer("p_rmsip_post_" + std::to_string(i), 16);
                gpu->writeBuffer(rmsIPParams, data, 16);
            }
            // Post-attention sandwich norm (in-place on projOutBuf)
            auto bgPostNorm = makeBG(plRmsIP, {
                {0, projOutBuf}, {1, lw.postNorm}, {2, rmsIPParams}});
            allDecodeDispatches.push_back({plRmsIP.pipeline, bgPostNorm, 1, 1, 1, L+"post_norm"});

            // Residual add: xBuf += projOutBuf
            static const char* ADD_INPLACE_WGSL2 = R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0];
    let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)";
            auto& plAddIP = gpu->getOrCreatePipeline("add_inplace_sw",
                std::string(ADD_INPLACE_WGSL2), 3);
            GPUBuffer addParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                addParams = gpu->createBuffer("p_add_post_" + std::to_string(i), 16);
                gpu->writeBuffer(addParams, data, 16);
            }
            auto bgAdd = makeBG(plAddIP, {
                {0, xBuf}, {1, projOutBuf}, {2, addParams}});
            allDecodeDispatches.push_back({plAddIP.pipeline, bgAdd,
                (cfg.nEmbd + 255) / 256, 1, 1, L+"add_res1"});

            // Pre-FFN RMSNorm (using ffnNorm)
            auto bgFfnNorm = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.ffnNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bgFfnNorm, 1, 1, 1, L+"ffn_norm"});
        } else {
            // Standard: fused residual-add + pre-FFN norm
            auto bg = makeBG(plAddRmsNorm, {
                {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                {3, lw.postAttnNorm}, {4, rstdBuf}, {5, rmsParams}});
            allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg, 1, 1, 1, L+"add_rms"});
        }

        // ── MoE FFN dispatch (Phase 3d) ─────────────────────────────────────
        // Sequence per MoE layer per decode step:
        //   router_matmul  → moeRouterOutBuf
        //   topk_softmax   → moeIndicesBuf, moeWeightsBuf
        //   compute_offsets → moeGateOffsets, moeUpOffsets, moeDownOffsets
        //   for k in 0..numExpertsPerTok:
        //     iq2s_matmul_moe(slot=k, gate_offsets) → moeRoutedGateBuf
        //     iq2s_matmul_moe(slot=k, up_offsets)   → moeRoutedUpBuf
        //     silu_mul(gate,up)                     → moeRoutedActBuf
        //     iq3s_matmul_moe(slot=k, down_offsets) → moeExpertOutBuf
        //     weighted_accumulate(slot=k)           : xBuf += w[k] * moeExpertOutBuf
        //   shared_expert: 4 Q8 dispatches + add into xBuf
        if (cfg.numExperts > 0) {
            // ── per-layer buffers we need that aren't in member state ──
            // These persist for the life of this build (each layer gets its own).
            uint32_t IMe = cfg.moeIntermediateSize;
            uint32_t IMs = cfg.moeSharedIntermediateSize;
            uint32_t topk = cfg.numExpertsPerTok;

            // 3 per-direction offset buffers (k slots each, u32 each = 4*k bytes)
            GPUBuffer gateOffsets = gpu->createBuffer("moe_gate_off_L"+std::to_string(i), topk*4);
            GPUBuffer upOffsets   = gpu->createBuffer("moe_up_off_L"+std::to_string(i),   topk*4);
            GPUBuffer downOffsets = gpu->createBuffer("moe_down_off_L"+std::to_string(i), topk*4);

            // Param buffers (16 B each)
            auto mkP = [&](const std::string& name, std::initializer_list<uint32_t> data) -> GPUBuffer {
                uint32_t buf[8] = {0};
                size_t i2 = 0; for (uint32_t v : data) { buf[i2++] = v; }
                auto b = gpu->createBuffer(name, 32);
                gpu->writeBuffer(b, buf, 32);
                return b;
            };

            // 1. Router matmul (Q8): normOutBuf @ routerW → moeRouterOutBuf
            {
                auto p = mkP("p_router_L"+std::to_string(i), {cfg.nEmbd, cfg.numExperts});
                auto bg = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.routerW}, {2, lw.routerS},
                    {3, zeroBiasV}, {4, moeRouterOutBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (cfg.numExperts + Q8_TILE - 1) / Q8_TILE, 1, L+"moe_router"});
            }

            // 2. topk_softmax: moeRouterOutBuf → moeIndicesBuf, moeWeightsBuf
            {
                auto& plTopK = getKernel("moe_topk_softmax");
                auto p = mkP("p_topk_L"+std::to_string(i), {cfg.numExperts, topk, 1u});
                auto bg = makeBG(plTopK, {
                    {0, moeRouterOutBuf}, {1, moeIndicesBuf}, {2, moeWeightsBuf}, {3, p}});
                allDecodeDispatches.push_back({plTopK.pipeline, bg, 1, 1, 1, L+"moe_topk"});
            }

            // 3. compute_offsets: moeIndicesBuf → gate/up/down offsets
            {
                auto& plCO = getKernel("moe_compute_offsets");
                auto p = mkP("p_co_L"+std::to_string(i), {topk, IMe, cfg.nEmbd});
                auto bg = makeBG(plCO, {
                    {0, moeIndicesBuf}, {1, gateOffsets}, {2, upOffsets}, {3, downOffsets}, {4, p}});
                allDecodeDispatches.push_back({plCO.pipeline, bg, 1, 1, 1, L+"moe_co"});
            }

            // 4. Per-expert loop (8 slots)
            // For each slot, dispatch: gate (IQ2_S), up (IQ2_S), silu_mul, down, accumulate.
            // We assume gate/up are IQ2_S and down is IQ3_S — most layers in this model.
            // (TODO: per-layer down quant detection; this hardcodes IQ3_S which is wrong
            //  for layers 34, 38, 39 where down is IQ4_XS.)
            auto& plIQ2Moe = getKernel("iq2s_matmul_moe");
            auto& plIQ3Moe = getKernel("iq3s_matmul_moe");
            auto& plIQ4XSMoe = getKernel("iq4xs_matmul_moe");
            auto& plSiluMul = getKernel("silu_mul_fused");  // existing fused kernel
            auto& plWAcc = getKernel("moe_weighted_accumulate_decode");

            // Compute n_blocks + row_stride for the fused expert buffers.
            // For IQ2_S blocks of 82 bytes padded to 84 = 21 words; rowStride = 21*nBlocks.
            // K-dim of gate/up = nEmbd = 2048; K/QK_K = 8 blocks per row → rowStride=168 words.
            uint32_t iq2_nBlocks_e   = cfg.nEmbd / 256u;            // K dim is nEmbd for gate/up
            uint32_t iq2_rowStride_e = iq2_nBlocks_e * 21u;
            uint32_t iq3_nBlocks_im  = IMe / 256u;                  // K dim is IMe for down
            uint32_t iq3_rowStride_im= iq3_nBlocks_im * 28u;
            uint32_t iq4xs_rowStride_im = iq3_nBlocks_im * 34u;
            // Pick down dispatch based on this layer's recorded quant type.
            bool downIsIQ4XS = (moeExpertsDownType.size() > i && moeExpertsDownType[i] == (uint32_t)GGUF_TYPE_IQ4_XS);
            auto& plDownMoe = downIsIQ4XS ? plIQ4XSMoe : plIQ3Moe;
            uint32_t downRowStride = downIsIQ4XS ? iq4xs_rowStride_im : iq3_rowStride_im;

            for (uint32_t k = 0; k < topk; k++) {
                std::string ks = std::to_string(k);
                // 4a. gate: normOutBuf @ expertsGateW(slot=k) → moeRoutedGateBuf
                {
                    auto p = mkP("p_gate_L"+std::to_string(i)+"_s"+ks,
                                 {cfg.nEmbd, IMe, iq2_nBlocks_e, iq2_rowStride_e, 0u, k});
                    auto bg = makeBG(plIQ2Moe, {
                        {0, normOutBuf}, {1, lw.expertsGateW}, {2, iq2sCodebookBuf},
                        {3, zeroBiasGU}, {4, moeRoutedGateBuf}, {5, gateOffsets}, {6, p}});
                    allDecodeDispatches.push_back({plIQ2Moe.pipeline, bg,
                        1, (IMe + 8u - 1u) / 8u, 1, L+"moe_gate_s"+ks});
                }
                // 4b. up: → moeRoutedUpBuf
                {
                    auto p = mkP("p_up_L"+std::to_string(i)+"_s"+ks,
                                 {cfg.nEmbd, IMe, iq2_nBlocks_e, iq2_rowStride_e, 0u, k});
                    auto bg = makeBG(plIQ2Moe, {
                        {0, normOutBuf}, {1, lw.expertsUpW}, {2, iq2sCodebookBuf},
                        {3, zeroBiasGU}, {4, moeRoutedUpBuf}, {5, upOffsets}, {6, p}});
                    allDecodeDispatches.push_back({plIQ2Moe.pipeline, bg,
                        1, (IMe + 8u - 1u) / 8u, 1, L+"moe_up_s"+ks});
                }
                // 4c. silu_mul(gate, up) → moeRoutedActBuf
                {
                    auto p = mkP("p_silumul_L"+std::to_string(i)+"_s"+ks, {IMe});
                    // silu_mul_fused expects [gate, up] packed in gateUpBuf typically;
                    // for MoE we pass them as separate inputs. The kernel signature
                    // may differ — using shared_silu + shared_mul as fallback.
                    auto& plSilu = getKernel("shared_silu");
                    auto& plMul  = getKernel("shared_mul");
                    auto bgSilu = makeBG(plSilu,
                        {{0, moeRoutedGateBuf}, {1, moeRoutedGateBuf}, {2, p}});
                    allDecodeDispatches.push_back({plSilu.pipeline, bgSilu,
                        (IMe + 255) / 256, 1, 1, L+"moe_silu_s"+ks});
                    auto bgMul = makeBG(plMul,
                        {{0, moeRoutedGateBuf}, {1, moeRoutedUpBuf}, {2, moeRoutedActBuf}, {3, p}});
                    allDecodeDispatches.push_back({plMul.pipeline, bgMul,
                        (IMe + 255) / 256, 1, 1, L+"moe_mul_s"+ks});
                }
                // 4d. down: moeRoutedActBuf @ expertsDownW(slot=k) → moeExpertOutBuf
                {
                    auto p = mkP("p_down_L"+std::to_string(i)+"_s"+ks,
                                 {IMe, cfg.nEmbd, iq3_nBlocks_im, downRowStride, 0u, k});
                    // IQ3_S kernel needs codebook binding; IQ4_XS does not.
                    GPUBuffer codebookBuf = downIsIQ4XS ? zeroBiasE : iq3sCodebookBuf; // codebook irrelevant for iq4xs
                    if (downIsIQ4XS) {
                        auto bg = makeBG(plDownMoe, {
                            {0, moeRoutedActBuf}, {1, lw.expertsDownW},
                            {2, zeroBiasE}, {3, moeExpertOutBuf}, {4, downOffsets}, {5, p}});
                        allDecodeDispatches.push_back({plDownMoe.pipeline, bg,
                            1, (cfg.nEmbd + 8u - 1u) / 8u, 1, L+"moe_down_iq4xs_s"+ks});
                    } else {
                        auto bg = makeBG(plDownMoe, {
                            {0, moeRoutedActBuf}, {1, lw.expertsDownW}, {2, iq3sCodebookBuf},
                            {3, zeroBiasE}, {4, moeExpertOutBuf}, {5, downOffsets}, {6, p}});
                        allDecodeDispatches.push_back({plDownMoe.pipeline, bg,
                            1, (cfg.nEmbd + 8u - 1u) / 8u, 1, L+"moe_down_iq3s_s"+ks});
                    }
                }
                // 4e. weighted_accumulate: xBuf += weights[k] * moeExpertOutBuf
                {
                    auto p = mkP("p_wacc_L"+std::to_string(i)+"_s"+ks, {cfg.nEmbd, k});
                    auto bg = makeBG(plWAcc,
                        {{0, xBuf}, {1, moeExpertOutBuf}, {2, moeWeightsBuf}, {3, p}});
                    allDecodeDispatches.push_back({plWAcc.pipeline, bg,
                        (cfg.nEmbd + 255) / 256, 1, 1, L+"moe_wacc_s"+ks});
                }
            }

            // 5. Shared expert (Q8 dense)
            {
                auto pGate = mkP("p_sh_gate_L"+std::to_string(i), {cfg.nEmbd, IMs});
                auto bgGate = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.shexpGateW}, {2, lw.shexpGateS},
                    {3, zeroBiasGU}, {4, moeShexpGateUpBuf}, {5, pGate}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgGate,
                    1, (IMs + Q8_TILE - 1) / Q8_TILE, 1, L+"shexp_gate"});

                // up writes to moeShexpGateUpBuf offset IMs — but our buffer alloc is 2*IMs.
                // For simplicity we use a separate small slice via creating another buffer.
                // (Could pack via y_offset param; using moeShexpActBuf as up temp for now.)
                auto pUp = mkP("p_sh_up_L"+std::to_string(i), {cfg.nEmbd, IMs});
                auto bgUp = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.shexpUpW}, {2, lw.shexpUpS},
                    {3, zeroBiasGU}, {4, moeShexpActBuf}, {5, pUp}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgUp,
                    1, (IMs + Q8_TILE - 1) / Q8_TILE, 1, L+"shexp_up"});

                // silu(gate) * up → gate buf (in-place)
                auto pSiluP = mkP("p_sh_silu_L"+std::to_string(i), {IMs});
                auto& plSilu = getKernel("shared_silu");
                auto& plMul  = getKernel("shared_mul");
                auto bgSilu = makeBG(plSilu,
                    {{0, moeShexpGateUpBuf}, {1, moeShexpGateUpBuf}, {2, pSiluP}});
                allDecodeDispatches.push_back({plSilu.pipeline, bgSilu,
                    (IMs + 255) / 256, 1, 1, L+"shexp_silu"});
                auto bgMul = makeBG(plMul,
                    {{0, moeShexpGateUpBuf}, {1, moeShexpActBuf},
                     {2, moeShexpGateUpBuf}, {3, pSiluP}});
                allDecodeDispatches.push_back({plMul.pipeline, bgMul,
                    (IMs + 255) / 256, 1, 1, L+"shexp_mul"});

                // down: moeShexpGateUpBuf @ shexpDownW → moeExpertOutBuf, then add to xBuf
                auto pDn = mkP("p_sh_dn_L"+std::to_string(i), {IMs, cfg.nEmbd});
                auto bgDn = makeBG(plQ8Matmul, {
                    {0, moeShexpGateUpBuf}, {1, lw.shexpDownW}, {2, lw.shexpDownS},
                    {3, zeroBiasE}, {4, moeExpertOutBuf}, {5, pDn}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgDn,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"shexp_dn"});

                // residual add: xBuf += moeExpertOutBuf (use a constant-weight accumulator)
                auto pAdd = mkP("p_sh_add_L"+std::to_string(i), {cfg.nEmbd, 0u});
                // Reuse plWAcc but with a "weights" buffer that's 1.0 at index 0.
                // For simplicity create a tiny [1] buffer with value 1.0.
                GPUBuffer one = gpu->createBuffer("one_L"+std::to_string(i), 4);
                float oneF = 1.0f;
                gpu->writeBuffer(one, &oneF, 4);
                auto bgAdd = makeBG(plWAcc,
                    {{0, xBuf}, {1, moeExpertOutBuf}, {2, one}, {3, pAdd}});
                allDecodeDispatches.push_back({plWAcc.pipeline, bgAdd,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"shexp_add"});
            }

            continue;  // skip the dense FFN block below
        }

        {
            if (useKQ) {
                auto bg = makeBG(*plKQ, {
                    {0, normOutBuf}, {1, lw.guKQ}, {2, zeroBiasGU},
                    {3, gateUpBuf}, {4, kqGuParams}});
                di.gateup = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plKQ->pipeline, bg,
                    1, (2 * cfg.intermediateSize + kqTileN - 1) / kqTileN, 1, L+"kq_gateup"});
            } else {
                vbg.gateupBase = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                    {3, zeroBiasGU}, {4, gateUpBuf}, {5, layerGuP}});
                if (decodeFastVariantsAvailable) {
                    vbg.gateupFast = makeBG(plQ8Fast, {
                        {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                        {3, zeroBiasGU}, {4, gateUpBuf}, {5, layerGuP}});
                }
                auto bg = tuning.decodeUseFastGateup && vbg.gateupFast ? vbg.gateupFast : vbg.gateupBase;
                auto pipeline = tuning.decodeUseFastGateup && vbg.gateupFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
                di.gateup = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({pipeline, bg,
                    1, (2 * plIM + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_gateup"});
            }
        }

        // 9. Down projection + SiLU + residual add
        {
            if (useKQ) {
                // K-quant: activation+mul → K-quant matmul → add residual (3 dispatches)
                auto& plActMul = useGelu ? getKernel("gelu_mul_fused")
                                          : getKernel("silu_mul_fused");
                GPUBuffer siluParams;
                {
                    uint32_t data[4] = {cfg.intermediateSize, 0, 0, 0};
                    siluParams = gpu->createBuffer("p_silu_" + std::to_string(i), 16);
                    gpu->writeBuffer(siluParams, data, 16);
                }
                auto bgSilu = makeBG(plActMul, {
                    {0, gateUpBuf}, {1, siluMulOutBuf}, {2, siluParams}});
                allDecodeDispatches.push_back({plActMul.pipeline, bgSilu,
                    (cfg.intermediateSize + 127) / 128, 1, 1, L+"act_mul"});

                auto bgDn = makeBG(*plKQ, {
                    {0, siluMulOutBuf}, {1, lw.dnKQ}, {2, zeroBiasE},
                    {3, projOutBuf}, {4, kqDnParams}});
                allDecodeDispatches.push_back({plKQ->pipeline, bgDn,
                    1, (cfg.nEmbd + kqTileN - 1) / kqTileN, 1, L+"kq_down"});

                // Add residual: xBuf[i] += projOutBuf[i] using inline add_inplace kernel
                static const char* ADD_INPLACE_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0];
    let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)";
                auto& plAddIP = gpu->getOrCreatePipeline("add_inplace",
                    std::string(ADD_INPLACE_WGSL), 3);
                GPUBuffer addIPParams;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    addIPParams = gpu->createBuffer("p_addip_" + std::to_string(i), 16);
                    gpu->writeBuffer(addIPParams, data, 16);
                }
                // Post-FFN sandwich norm (before residual add)
                if (cfg.hasSandwichNorm && lw.postFfwNorm.handle) {
                    auto& plRmsIP2 = gpu->getOrCreatePipeline("rms_norm_inplace",
                        std::string(R"(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0]; let eps = bitcast<f32>(_p_[1]); let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; sum_sq += v * v; }
    let total = subgroupAdd(sum_sq);
    let rms = 1.0 / sqrt(subgroupBroadcastFirst(total) / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { X[j] = X[j] * rms * W[j]; }
}
)"), 3);
                    GPUBuffer rmsP;
                    {
                        uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                        float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                        rmsP = gpu->createBuffer("p_rmsip_ffw_" + std::to_string(i), 16);
                        gpu->writeBuffer(rmsP, data, 16);
                    }
                    auto bgFfwNorm = makeBG(plRmsIP2, {
                        {0, projOutBuf}, {1, lw.postFfwNorm}, {2, rmsP}});
                    allDecodeDispatches.push_back({plRmsIP2.pipeline, bgFfwNorm,
                        1, 1, 1, L+"post_ffw_norm"});
                }

                auto bgAdd = makeBG(plAddIP, {
                    {0, xBuf}, {1, projOutBuf}, {2, addIPParams}});
                allDecodeDispatches.push_back({plAddIP.pipeline, bgAdd,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"residual_add"});
            } else if (cfg.hasSandwichNorm && lw.postFfwNorm.handle) {
                // Sandwich norm Q8 path: split into act → matmul → norm → add
                // 1. Activation: gateUpBuf → siluMulOutBuf (reuse K-quant temp buffer)
                if (!siluMulOutBuf.handle)
                    siluMulOutBuf = gpu->createBuffer("silu_mul_out", maxIMBuf * 4);
                auto& plActMulSW = useGelu ? getKernel("gelu_mul_fused")
                                            : getKernel("silu_mul_fused");
                GPUBuffer actParams;
                {
                    uint32_t data[4] = {plIM, 0, 0, 0};
                    actParams = gpu->createBuffer("p_act_sw_" + std::to_string(i), 16);
                    gpu->writeBuffer(actParams, data, 16);
                }
                auto bgAct = makeBG(plActMulSW, {
                    {0, gateUpBuf}, {1, siluMulOutBuf}, {2, actParams}});
                allDecodeDispatches.push_back({plActMulSW.pipeline, bgAct,
                    (plIM + 127) / 128, 1, 1, L+"act_mul_sw"});

                // 2. Down matmul: siluMulOutBuf → projOutBuf (no residual add)
                auto bgDn = makeBG(plQ8Matmul, {
                    {0, siluMulOutBuf}, {1, lw.dnW}, {2, lw.dnS},
                    {3, zeroBiasE}, {4, projOutBuf}, {5, layerDnP}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgDn,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_down_sw"});

                // 3. Post-FFN sandwich norm (in-place on projOutBuf)
                auto& plRmsIPfw = gpu->getOrCreatePipeline("rms_norm_inplace",
                    std::string(R"(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0]; let eps = bitcast<f32>(_p_[1]); let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; sum_sq += v * v; }
    let total = subgroupAdd(sum_sq);
    let rms = 1.0 / sqrt(subgroupBroadcastFirst(total) / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { X[j] = X[j] * rms * W[j]; }
}
)"), 3);
                GPUBuffer rmsP2;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                    rmsP2 = gpu->createBuffer("p_rmsip_ffw2_" + std::to_string(i), 16);
                    gpu->writeBuffer(rmsP2, data, 16);
                }
                auto bgFfwNorm = makeBG(plRmsIPfw, {
                    {0, projOutBuf}, {1, lw.postFfwNorm}, {2, rmsP2}});
                allDecodeDispatches.push_back({plRmsIPfw.pipeline, bgFfwNorm,
                    1, 1, 1, L+"post_ffw_norm"});

                // 4. Residual add: xBuf += projOutBuf
                auto& plAddIPsw = gpu->getOrCreatePipeline("add_inplace",
                    std::string(R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0]; let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)"), 3);
                GPUBuffer addP2;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    addP2 = gpu->createBuffer("p_add_ffw_" + std::to_string(i), 16);
                    gpu->writeBuffer(addP2, data, 16);
                }
                auto bgAdd2 = makeBG(plAddIPsw, {
                    {0, xBuf}, {1, projOutBuf}, {2, addP2}});
                allDecodeDispatches.push_back({plAddIPsw.pipeline, bgAdd2,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"add_res2"});
            } else {
                auto bg = makeBG(plDnSilu, {
                    {0, gateUpBuf}, {1, lw.dnW}, {2, lw.dnS},
                    {3, zeroBiasE}, {4, xBuf}, {5, layerDnP}});
                allDecodeDispatches.push_back({plDnSilu.pipeline, bg,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_down_silu_add"});
            }
        }

        // PLE (Per-Layer Embedding) injection
        if (cfg.pleSize > 0 && lw.pleInpGateW.handle && lw.pleProjW.handle) {
            uint32_t pleDim = cfg.pleSize;

            // 1. inp_gate matmul: xBuf [E] → pleBuf [pleSize]
            GPUBuffer pleGateP = makeQ8Params("p_ple_gate_" + std::to_string(i), cfg.nEmbd, pleDim);
            auto bgGate = makeBG(plQ8Matmul, {
                {0, xBuf}, {1, lw.pleInpGateW}, {2, lw.pleInpGateS},
                {3, zeroBiasE}, {4, pleBuf}, {5, pleGateP}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgGate,
                1, (pleDim + Q8_TILE - 1) / Q8_TILE, 1, L+"ple_gate"});

            // 2. GELU activation (in-place on pleBuf)
            static const char* GELU_INPLACE_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0]; let i = gid.x;
    if (i >= N) { return; }
    let x = X[i];
    X[i] = x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}
)";
            auto& plGeluIP = gpu->getOrCreatePipeline("gelu_inplace",
                std::string(GELU_INPLACE_WGSL), 2);
            GPUBuffer geluP;
            {
                uint32_t data[4] = {pleDim, 0, 0, 0};
                geluP = gpu->createBuffer("p_ple_gelu_" + std::to_string(i), 16);
                gpu->writeBuffer(geluP, data, 16);
            }
            auto bgGelu = makeBG(plGeluIP, {
                {0, pleBuf}, {1, geluP}});
            allDecodeDispatches.push_back({plGeluIP.pipeline, bgGelu,
                (pleDim + 255) / 256, 1, 1, L+"ple_gelu"});

            // 3. Element-wise multiply: pleBuf *= pleSliceBufs[i]
            static const char* EMUL_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0]; let i = gid.x;
    if (i < N) { A[i] = A[i] * B[i]; }
}
)";
            auto& plEMul = gpu->getOrCreatePipeline("elementwise_mul",
                std::string(EMUL_WGSL), 3);
            auto bgMul = makeBG(plEMul, {
                {0, pleBuf}, {1, pleSliceBufs[i]}, {2, geluP}});
            allDecodeDispatches.push_back({plEMul.pipeline, bgMul,
                (pleDim + 255) / 256, 1, 1, L+"ple_mul"});

            // 4. Back-projection: pleBuf [pleSize] → pleOutBuf [E]
            GPUBuffer pleProjP = makeQ8Params("p_ple_proj_" + std::to_string(i), pleDim, cfg.nEmbd);
            auto bgProj = makeBG(plQ8Matmul, {
                {0, pleBuf}, {1, lw.pleProjW}, {2, lw.pleProjS},
                {3, zeroBiasE}, {4, pleOutBuf}, {5, pleProjP}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgProj,
                1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"ple_proj"});

            // 5. RMSNorm on pleOutBuf (in-place)
            if (lw.plePostNorm.handle) {
                auto& plRmsIPple = gpu->getOrCreatePipeline("rms_norm_inplace",
                    std::string(R"(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0]; let eps = bitcast<f32>(_p_[1]); let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; sum_sq += v * v; }
    let total = subgroupAdd(sum_sq);
    let rms = 1.0 / sqrt(subgroupBroadcastFirst(total) / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { X[j] = X[j] * rms * W[j]; }
}
)"), 3);
                GPUBuffer rmsPlP;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                    rmsPlP = gpu->createBuffer("p_ple_norm_" + std::to_string(i), 16);
                    gpu->writeBuffer(rmsPlP, data, 16);
                }
                auto bgPleNorm = makeBG(plRmsIPple, {
                    {0, pleOutBuf}, {1, lw.plePostNorm}, {2, rmsPlP}});
                allDecodeDispatches.push_back({plRmsIPple.pipeline, bgPleNorm,
                    1, 1, 1, L+"ple_norm"});
            }

            // 6. Residual add: xBuf += pleOutBuf
            auto& plAddPLE = gpu->getOrCreatePipeline("add_inplace",
                std::string(R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0]; let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)"), 3);
            GPUBuffer addPleP;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                addPleP = gpu->createBuffer("p_ple_add_" + std::to_string(i), 16);
                gpu->writeBuffer(addPleP, data, 16);
            }
            auto bgPleAdd = makeBG(plAddPLE, {
                {0, xBuf}, {1, pleOutBuf}, {2, addPleP}});
            allDecodeDispatches.push_back({plAddPLE.pipeline, bgPleAdd,
                (cfg.nEmbd + 255) / 256, 1, 1, L+"ple_add"});
        }

        // Per-layer output scale
        if (lw.outScale.handle) {
            static const char* SCALE_BUF_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0];
    let i = gid.x;
    if (i < N) { X[i] = X[i] * Scale[0]; }
}
)";
            auto& plScale = gpu->getOrCreatePipeline("scale_buf",
                std::string(SCALE_BUF_WGSL), 3);
            GPUBuffer scaleParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                scaleParams = gpu->createBuffer("p_scale_" + std::to_string(i), 16);
                gpu->writeBuffer(scaleParams, data, 16);
            }
            auto bgScale = makeBG(plScale, {
                {0, xBuf}, {1, lw.outScale}, {2, scaleParams}});
            allDecodeDispatches.push_back({plScale.pipeline, bgScale,
                (cfg.nEmbd + 255) / 256, 1, 1, L+"out_scale"});
        }

        // 10. RMSNorm for next layer (only needed for KQ path — Q8 path uses fused kernel)
        if (i < cfg.nLayer - 1 && useKQ) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, layerWeights[i+1].inputNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_next"});
        }
    }

    // Final RMSNorm
    {
        auto bg = makeBG(plRmsNorm, {
            {0, xBuf}, {1, normOutBuf}, {2, finalNormW},
            {3, rstdBuf}, {4, rmsParams}});
        allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, "final_rms"});
    }

    // LM head
    if (lmHeadIsKQ && plKQ) {
        auto bg = makeBG(*plKQ, {
            {0, normOutBuf}, {1, lmHeadKQ}, {2, zeroBiasV},
            {3, logitsBuf}, {4, kqLmParams}});
        allDecodeDispatches.push_back({plKQ->pipeline, bg,
            1, (cfg.nVocab + 7) / 8, 1, "lm_head"});
    } else if (lmHeadIsQ8) {
        auto q8LmParams = makeQ8Params("p_lmhead_q8", cfg.nEmbd, cfg.nVocab);
        auto bg = makeBG(plQ8Matmul, {
            {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
            {3, zeroBiasV}, {4, logitsBuf}, {5, q8LmParams}});
        allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
            1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1, "lm_head"});
    } else {
        const auto& plLmHead = tuning.decodeUseWideFp16 ? plFp16Wide : plFp16Gemm;
        uint32_t FP16_TILE = tuning.decodeUseWideFp16 ? 32u : 8u;
        auto bg = makeBG(plLmHead, {
            {0, normOutBuf}, {1, lmHeadW}, {2, zeroBiasV},
            {3, logitsBuf}, {4, lmheadParams}});
        allDecodeDispatches.push_back({plLmHead.pipeline, bg,
            1, (cfg.nVocab + FP16_TILE - 1) / FP16_TILE, 1, "lm_head"});
    }

    // Logit softcapping (Gemma 2/4): tanh(logits/cap) * cap
    if (cfg.logitSoftcap > 0.0f) {
        static const char* LOGIT_SOFTCAP_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> _params_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let cap = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= N) { return; }
    Y[i] = tanh(Y[i] / cap) * cap;
}
)";
        auto& plSoftcap = gpu->getOrCreatePipeline("logit_softcap",
            std::string(LOGIT_SOFTCAP_WGSL), 2);
        GPUBuffer softcapParams;
        {
            uint32_t data[4] = {cfg.nVocab, 0, 0, 0};
            memcpy(&data[1], &cfg.logitSoftcap, 4);
            softcapParams = gpu->createBuffer("p_softcap", 16);
            gpu->writeBuffer(softcapParams, data, 16);
        }
        auto bg = makeBG(plSoftcap, {
            {0, logitsBuf}, {1, softcapParams}});
        softcapPipeline = plSoftcap.pipeline;
        softcapBG = bg;
        softcapDispatchX = (cfg.nVocab + 255) / 256;
        allDecodeDispatches.push_back({softcapPipeline, bg,
            softcapDispatchX, 1, 1, "logit_softcap"});
    }

    // GPU argmax
    {
        auto bg = makeBG(plArgmax, {
            {0, logitsBuf}, {1, argmaxResultBuf}, {2, argmaxParams}});
        allDecodeDispatches.push_back({plArgmax.pipeline, bg, 1, 1, 1, "argmax"});
    }

    // Auto-decode: embed_gather + full pipeline
    {
        auto bg = makeBG(plEmbGather, {
            {0, embeddingGpuBuf}, {1, argmaxResultBuf},
            {2, xBuf}, {3, embedParams}});
        autoDecodeDispatches.clear();
        autoDecodeDispatches.push_back({plEmbGather.pipeline, bg,
            (cfg.nEmbd + 255) / 256, 1, 1, "embed_gather"});

        // Embedding scaling (Gemma: multiply by sqrt(n_embd))
        if (cfg.embeddingScale > 0.0f) {
            static const char* SCALE_INPLACE_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> _params_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let scale = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= N) { return; }
    X[i] = X[i] * scale;
}
)";
            auto& plScale = gpu->getOrCreatePipeline("scale_inplace",
                std::string(SCALE_INPLACE_WGSL), 2);
            GPUBuffer scaleParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                memcpy(&data[1], &cfg.embeddingScale, 4);
                scaleParams = gpu->createBuffer("p_emb_scale", 16);
                gpu->writeBuffer(scaleParams, data, 16);
            }
            auto scaleBg = makeBG(plScale, {
                {0, xBuf}, {1, scaleParams}});
            autoDecodeDispatches.push_back({plScale.pipeline, scaleBg,
                (cfg.nEmbd + 255) / 256, 1, 1, "embed_scale"});
        }

        autoDecodeDispatches.insert(autoDecodeDispatches.end(),
            allDecodeDispatches.begin(), allDecodeDispatches.end());
    }

    fprintf(stderr, "  %zu decode dispatches (%u layers)\n",
           allDecodeDispatches.size(), cfg.nLayer);

    // ─── Create staging pool ──────────────────────────────────────────────
    pool.resize(decodePoolCapacity);
    for (int s = 0; s < decodePoolCapacity; s++) {
        auto& ps = pool[s];
        WGPUBufferDescriptor bd{};
        bd.usage = BUF_MAP_READ | BUF_COPY_DST;
        bd.size = 4;
        char label[32]; snprintf(label, 32, "staging_%d", s);
        bd.label = {label, (uint32_t)strlen(label)};
        ps.stagingBuf = wgpuDeviceCreateBuffer(gpu->device, &bd);

        // Per-slot param buffers
        char rl[32]; snprintf(rl, 32, "p_frope_%d", s);
        ps.ropeParamsBuf = gpu->createBuffer(std::string(rl), 32);
        gpu->writeBuffer(ps.ropeParamsBuf, ropeParamData.data(), 32);
        char al[32]; snprintf(al, 32, "p_cattn_%d", s);
        ps.attnParamsBuf = gpu->createBuffer(std::string(al), 32);
        gpu->writeBuffer(ps.attnParamsBuf, chunkedAttnParamData.data(), 32);

        // SWA per-slot param buffer
        if (hasSWA) {
            char sl[32]; snprintf(sl, 32, "p_cattn_swa_%d", s);
            ps.attnParamsBufSWA = gpu->createBuffer(std::string(sl), 32);
            gpu->writeBuffer(ps.attnParamsBufSWA, chunkedAttnParamData.data(), 32);
        }

        // Clone autoDecodeDispatches with per-slot bind groups for
        // dispatches that reference dynamic param buffers.
        ps.dispatches = autoDecodeDispatches;
        // Prefix dispatches before allDecodeDispatches: embed_gather [+ embed_scale]
        autoDecodePrefixCount = (cfg.embeddingScale > 0.0f) ? 2 : 1;
        int prefixCount = autoDecodePrefixCount;
        for (uint32_t layer = 0; layer < cfg.nLayer; layer++) {
            int ropeIdx = ropeDispatchIndices[layer] + prefixCount;
            int p1Idx   = attnP1DispatchIndices[layer] + prefixCount;
            int p2Idx   = attnP2DispatchIndices[layer] + prefixCount;

            ps.dispatches[ropeIdx].bindGroup = makeBG(plFusedRope, {
                {0, qkvBuf}, {1, qRotBuf},
                {2, kvCache[layer].K}, {3, kvCache[layer].V},
                {4, ropeCosBuf}, {5, ropeSinBuf},
                {6, layerWeights[layer].qNorm}, {7, layerWeights[layer].kNorm},
                {8, ps.ropeParamsBuf}});

            bool layerIsSWA = hasSWA && layer < cfg.layerAttnTypes.size() &&
                             cfg.layerAttnTypes[layer] == AttnLayerType::SlidingWindow;
            auto& slotAttnBuf = layerIsSWA ? ps.attnParamsBufSWA : ps.attnParamsBuf;

            ps.dispatches[p1Idx].bindGroup = makeBG(plChunkP1, {
                {0, qRotBuf}, {1, kvCache[layer].K}, {2, kvCache[layer].V},
                {3, attnPartialsBuf}, {4, slotAttnBuf}});

            ps.dispatches[p2Idx].bindGroup = makeBG(plChunkP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, slotAttnBuf}});
        }

        refillCBPool(s);
    }
    fprintf(stderr, "  Pool: %d slots × %d pre-recorded CBs\n",
            decodePoolCapacity, decodeCbPoolBatch);

    // Pre-allocate prefill resources (buffers + bind groups at maxSeqLen)
    // Skip for K-quant models — prefill kernels only support Q8/fp32 weights
    bool isKQ = (weightQuantType == GGUF_TYPE_Q4_K ||
                 weightQuantType == GGUF_TYPE_Q5_K ||
                 weightQuantType == GGUF_TYPE_Q6_K);
    if (!isKQ) {
        initPrefillResources();
    } else {
        fprintf(stderr, "  Prefill: skipped (K-quant weights, using serial decode)\n");
    }
}

// ─── Pre-allocate prefill resources ──────────────────────────────────────────

void ModelRunner::initPrefillResources() {
    uint32_t T = maxSeqLen;
    uint32_t qDimL  = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;
    const bool useGelu = (cfg.activation == ActivationType::GELU);
    uint32_t Q8_TILE = 8;

    // Intermediate buffers sized to maxSeqLen
    pfCache.pX    = gpu->createBuffer("pf_x",    T * cfg.nEmbd * 4);
    pfCache.pNorm = gpu->createBuffer("pf_norm", T * cfg.nEmbd * 4);
    pfCache.pQkv  = gpu->createBuffer("pf_qkv",  T * qkvOutL * 4);
    pfCache.pQRot = gpu->createBuffer("pf_qrot", T * qDimL * 4);
    pfCache.pAttn = gpu->createBuffer("pf_attn", T * qDimL * 4);
    pfCache.pProj = gpu->createBuffer("pf_proj", T * cfg.nEmbd * 4);
    pfCache.pGU   = gpu->createBuffer("pf_gu",   T * 2 * cfg.intermediateSize * 4);
    pfCache.pRstd = gpu->createBuffer("pf_rstd", T * 4);
    pfCache.pNormQ  = gpu->createBuffer("pf_norm_q",  T * cfg.nEmbd);
    pfCache.pNormQS = gpu->createBuffer("pf_norm_qs", T * (cfg.nEmbd / 32) * 4);
    pfCache.pAttnQ  = gpu->createBuffer("pf_attn_q",  T * qDimL);
    pfCache.pAttnQS = gpu->createBuffer("pf_attn_qs", T * (qDimL / 32) * 4);
    pfCache.pGUQ    = gpu->createBuffer("pf_gu_q",    T * cfg.intermediateSize);
    pfCache.pGUQS   = gpu->createBuffer("pf_gu_qs",   T * (cfg.intermediateSize / 32) * 4);

    // Global param buffers (written per-call with actual T)
    // Both backends use var<uniform> for early loop exit
    bool isVulkan = (gpu->backendType != WGPUBackendType_D3D12);
    uint64_t paramUsage = BUF_UNIFORM | BUF_COPY_DST;
    pfCache.pQkvP = gpu->createBuffer("pp_qkv", 16, paramUsage);
    pfCache.pOpP  = gpu->createBuffer("pp_op",  16, paramUsage);
    pfCache.pGuP  = gpu->createBuffer("pp_gu",  16, paramUsage);
    pfCache.pDnP  = gpu->createBuffer("pp_dn",  16, paramUsage);
    pfCache.pNormQP = gpu->createBuffer("pp_norm_q", 16, paramUsage);
    pfCache.pAttnQP = gpu->createBuffer("pp_attn_q", 16, paramUsage);
    pfCache.pGUQP   = gpu->createBuffer("pp_gu_q",   16, paramUsage);
    pfCache.pLmP  = gpu->createBuffer("pp_lm",  16);
    {
        uint32_t d[4] = {cfg.nEmbd, cfg.nEmbd, 0, 0};
        float eps = cfg.rmsNormEps; memcpy(&d[2], &eps, 4);
        pfCache.pRmsP = gpu->createBuffer("pp_rms", 16);
        gpu->writeBuffer(pfCache.pRmsP, d, 16);
    }

    // Per-layer param buffers
    pfCache.ropeParams.resize(cfg.nLayer);
    pfCache.attnParams.resize(cfg.nLayer);
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        pfCache.ropeParams[li] = gpu->createBuffer(
            "pp_rope_L" + std::to_string(li), 32);
        // Attention params always uniform (enables early exit on both backends)
        pfCache.attnParams[li] = gpu->createBuffer(
            "pp_attn_L" + std::to_string(li), 32,
            BUF_UNIFORM | BUF_COPY_DST);
    }

    // Get kernels — MMA on Vulkan, prequantized DP4A on D3D12 when available
    auto& kRmsB    = getKernel("rms_norm_batched");
    auto& kAddRmsB = getKernel("add_rms_norm_batched");
    auto& kRopeB   = getKernelHD(cfg.hasQkNorm ? "rope_batched_simple" : "rope_batched");

    const auto& limits = effectiveLimits(*gpu);
    const bool canUse512ThreadKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 512u &&
        limits.maxComputeWorkgroupStorageSize >= 32u * 1024u;
    const bool canUse256ThreadSubgroupKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 256u &&
        limits.maxComputeWorkgroupStorageSize >= 16u * 1024u;

    // Select matmul + attention kernels based on capability, not backend alone.
    useMMA = (gpu->backendType != WGPUBackendType_D3D12) &&
             gpu->supportsSubgroupMatrix &&
             canUse512ThreadKernels;

    // Detect DP4A support on D3D12 (dot4I8Packed → 4× INT8 throughput)
    useDP4A = false;
    if (!useMMA) {
        useDP4A = canUse256ThreadSubgroupKernels &&
                  canCompileEmbeddedKernel(*gpu, "q8_matmul_dp4a_d3d12");
        fprintf(stderr, "  DP4A (dot4I8Packed): %s\n",
               useDP4A ? "available" : "not available");
    }

    bool usePrequant = !useMMA && useDP4A;
    tuning.prefillUseWidePrequant = usePrequant && canUse256ThreadSubgroupKernels;
    tuning.prefillUseWidePrequantAdd = usePrequant && canUse256ThreadSubgroupKernels;
    tuning.prefillMatM = useMMA ? 64u : 8u;
    tuning.prefillMatN = useMMA ? 64u : 32u;
    tuning.prefillWideMatM = tuning.prefillUseWidePrequant ? 4u : tuning.prefillMatM;
    tuning.prefillWideMatN = tuning.prefillUseWidePrequant ? 64u : tuning.prefillMatN;
    tuning.prefillDnM = tuning.prefillUseWidePrequantAdd ? 4u : tuning.prefillMatM;
    tuning.prefillDnN = tuning.prefillUseWidePrequantAdd ? 64u : tuning.prefillMatN;
    tuning.prefillAttnBlockQ = useMMA ? 16u : 4u;

    const char* wideMatKernel = tuning.prefillUseWidePrequant
        ? "q8_matmul_prequant_wide_d3d12"
        : nullptr;
    const char* wideAddKernel = tuning.prefillUseWidePrequantAdd
        ? "q8_matmul_prequant_add_wide_d3d12"
        : nullptr;
    const char* matKernel = useMMA ? "q8_matmul_vulkan"
                          : (usePrequant ? "q8_matmul_prequant_d3d12" : "q8_matmul_d3d12");
    const char* dnSiluKernel = useMMA ? "q8_down_silu_add_vulkan"
                             : (usePrequant ? (tuning.prefillUseWidePrequantAdd
                                                ? "q8_matmul_prequant_add_wide_d3d12"
                                                : "q8_matmul_prequant_add_d3d12")
                                            : "q8_down_silu_add_d3d12");
    const char* attnKernel = useMMA ? "flash_attn_vulkan" : "causal_attn";
    const CompiledPipeline* kQuant = usePrequant ? &getKernel("quantize_fp32_rows_d3d12") : nullptr;
    const CompiledPipeline* kSiluQ = usePrequant ? &(useGelu ? getKernelGelu("silu_quantize_rows_d3d12") : getKernel("silu_quantize_rows_d3d12")) : nullptr;
    const CompiledPipeline* kMatWide = tuning.prefillUseWidePrequant
        ? &getKernel(wideMatKernel)
        : nullptr;
    const CompiledPipeline* kDnSiluWide = tuning.prefillUseWidePrequantAdd
        ? &getKernel(wideAddKernel)
        : nullptr;
    auto& kMat     = getKernel(matKernel);
    auto& kDnSilu  = useGelu ? getKernelGelu(dnSiluKernel) : getKernel(dnSiluKernel);
    auto& kAttn    = getKernelHD(attnKernel);
    fprintf(stderr, "  Prefill tuning: mat=%s qkv/gateup=%s down=%s attn=%s tiles=%ux%u wide=%ux%u pool=%d\n",
           matKernel,
           tuning.prefillUseWidePrequant ? wideMatKernel : matKernel,
           dnSiluKernel,
           attnKernel,
           tuning.prefillMatM, tuning.prefillMatN,
           tuning.prefillWideMatM, tuning.prefillWideMatN,
           decodePoolDepth);

    // Build bind groups (stable — buffer handles don't change)
    pfCache.layerBGs.resize(cfg.nLayer);
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        auto& lw = layerWeights[li];
        auto& bg = pfCache.layerBGs[li];

        if (!lw.qkvW.handle && !lw.qkvKQ.handle) {
            continue;
        }

        bg.rms = makeBG(kRmsB, {
            {0, pfCache.pX}, {1, pfCache.pNorm}, {2, lw.inputNorm},
            {3, pfCache.pRstd}, {4, pfCache.pRmsP}});

        if (usePrequant) {
            bg.qnorm = makeBG(*kQuant, {
                {0, pfCache.pNorm}, {1, pfCache.pNormQ}, {2, pfCache.pNormQS},
                {3, pfCache.pNormQP}});

            bg.qkv = makeBG(tuning.prefillUseWidePrequant ? *kMatWide : kMat, {
                {0, pfCache.pNormQ}, {1, pfCache.pNormQS}, {2, lw.qkvW},
                {3, lw.qkvS}, {4, zeroBiasQKV}, {5, pfCache.pQkv}, {6, pfCache.pQkvP}});
        } else {
            bg.qkv = makeBG(kMat, {
                {0, pfCache.pNorm}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, pfCache.pQkv}, {5, pfCache.pQkvP}});
        }

        bg.rope = makeBG(kRopeB, {
            {0, pfCache.pQkv}, {1, pfCache.pQRot},
            {2, kvCache[li].K}, {3, kvCache[li].V},
            {4, ropeCosBuf}, {5, ropeSinBuf},
            {6, lw.qNorm}, {7, lw.kNorm},
            {8, pfCache.ropeParams[li]}});

        bg.attn = makeBG(kAttn, {
            {0, pfCache.pQRot}, {1, kvCache[li].K}, {2, kvCache[li].V},
            {3, pfCache.pAttn}, {4, pfCache.attnParams[li]}});

        if (usePrequant) {
            bg.attnq = makeBG(*kQuant, {
                {0, pfCache.pAttn}, {1, pfCache.pAttnQ}, {2, pfCache.pAttnQS},
                {3, pfCache.pAttnQP}});

            bg.oproj = makeBG(kMat, {
                {0, pfCache.pAttnQ}, {1, pfCache.pAttnQS}, {2, lw.oW},
                {3, lw.oS}, {4, zeroBiasE}, {5, pfCache.pProj}, {6, pfCache.pOpP}});
        } else {
            bg.oproj = makeBG(kMat, {
                {0, pfCache.pAttn}, {1, lw.oW}, {2, lw.oS},
                {3, zeroBiasE}, {4, pfCache.pProj}, {5, pfCache.pOpP}});
        }

        bg.addrms = makeBG(kAddRmsB, {
            {0, pfCache.pX}, {1, pfCache.pProj}, {2, pfCache.pNorm},
            {3, lw.postAttnNorm}, {4, pfCache.pRstd}, {5, pfCache.pRmsP}});

        if (usePrequant) {
            bg.gateup = makeBG(tuning.prefillUseWidePrequant ? *kMatWide : kMat, {
                {0, pfCache.pNormQ}, {1, pfCache.pNormQS}, {2, lw.guW},
                {3, lw.guS}, {4, zeroBiasGU}, {5, pfCache.pGU}, {6, pfCache.pGuP}});

            bg.siluq = makeBG(*kSiluQ, {
                {0, pfCache.pGU}, {1, pfCache.pGUQ}, {2, pfCache.pGUQS},
                {3, pfCache.pGUQP}});

            bg.downsilu = makeBG(kDnSilu, {
                {0, pfCache.pGUQ}, {1, pfCache.pGUQS}, {2, lw.dnW},
                {3, lw.dnS}, {4, zeroBiasE}, {5, pfCache.pX}, {6, pfCache.pDnP}});
        } else {
            bg.gateup = makeBG(kMat, {
                {0, pfCache.pNorm}, {1, lw.guW}, {2, lw.guS},
                {3, zeroBiasGU}, {4, pfCache.pGU}, {5, pfCache.pGuP}});

            bg.downsilu = makeBG(kDnSilu, {
                {0, pfCache.pGU}, {1, lw.dnW}, {2, lw.dnS},
                {3, zeroBiasE}, {4, pfCache.pX}, {5, pfCache.pDnP}});
        }
    }

    pfCache.finalRmsBG = makeBG(kRmsB, {
        {0, pfCache.pX}, {1, pfCache.pNorm}, {2, finalNormW},
        {3, pfCache.pRstd}, {4, pfCache.pRmsP}});

    // LM head bind group
    if (lmHeadIsQ8) {
        pfCache.lmBG = makeBG(getKernel("q8_matmul"), {
            {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
            {3, zeroBiasV}, {4, logitsBuf}, {5, pfCache.pLmP}});
    }

    // Argmax bind group
    {
        GPUBuffer argmaxP = gpu->createBuffer("pf_argmax_p", 16);
        uint32_t p[4] = {cfg.nVocab, 0, 0, 0};
        gpu->writeBuffer(argmaxP, p, 16);
        pfCache.argmaxBG = makeBG(getKernel("argmax"), {
            {0, logitsBuf}, {1, argmaxResultBuf}, {2, argmaxP}});
    }

    // ── Build pre-recorded indirect dispatch table ───────────────────
    // Pipelines and bind groups are static; only grid sizes change with T.
    // We store a fixed sequence of (pipeline, bindGroup, indirectOffset) entries
    // and update the indirect buffer contents before each prefill call.
    {
        uint32_t perLayerDispatches = usePrequant ? 12u : 8u;
        uint32_t nDispatches = cfg.nLayer * perLayerDispatches + 1;
        pfCache.indirectTable.clear();
        pfCache.indirectTable.reserve(nDispatches);

        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            auto& bg = pfCache.layerBGs[li];
            auto addEntry = [&](WGPUComputePipeline pl, WGPUBindGroup bg_,
                                const std::string& name) {
                uint64_t off = pfCache.indirectTable.size() * 12;
                pfCache.indirectTable.push_back({pl, bg_, off, name});
            };
            addEntry(kRmsB.pipeline,    bg.rms,      "pf_rms");
            if (usePrequant) addEntry(kQuant->pipeline, bg.qnorm,   "pf_qnorm");
            addEntry(tuning.prefillUseWidePrequant ? kMatWide->pipeline : kMat.pipeline,
                     bg.qkv,      "pf_qkv");
            addEntry(kRopeB.pipeline,   bg.rope,     "pf_rope");
            addEntry(kAttn.pipeline,    bg.attn,     "pf_attn");
            if (usePrequant) addEntry(kQuant->pipeline, bg.attnq,   "pf_attn_quant");
            addEntry(kMat.pipeline,     bg.oproj,    "pf_oproj");
            addEntry(kAddRmsB.pipeline, bg.addrms,   "pf_add_rms");
            if (usePrequant) addEntry(kQuant->pipeline, bg.qnorm,   "pf_qnorm_ffn");
            addEntry(tuning.prefillUseWidePrequant ? kMatWide->pipeline : kMat.pipeline,
                     bg.gateup,   "pf_gateup");
            if (usePrequant) addEntry(kSiluQ->pipeline, bg.siluq,   "pf_silu_quant");
            addEntry((tuning.prefillUseWidePrequantAdd && kDnSiluWide) ? kDnSiluWide->pipeline : kDnSilu.pipeline,
                     bg.downsilu, "pf_down_silu");
        }
        {
            uint64_t off = pfCache.indirectTable.size() * 12;
            pfCache.indirectTable.push_back({kRmsB.pipeline, pfCache.finalRmsBG,
                                              off, "pf_final_rms"});
        }

        // Create the indirect buffer: [gx, gy, gz] × nDispatches
        pfCache.indirectBuf = gpu->createBuffer("pf_indirect",
            nDispatches * 12, BUF_INDIRECT | BUF_COPY_DST);
    }

    pfCache.ready = true;
}

// ─── Pre-record command buffer pool ──────────────────────────────────────────

void ModelRunner::refillCBPool(int slot) {
    auto& ps = pool[slot];

    // Release any remaining CBs
    for (int i = ps.cbIdx; i < (int)ps.cbPool.size(); i++)
        wgpuCommandBufferRelease(ps.cbPool[i]);

    // Each "token" needs nGroups CBs. Pre-record decodeCbPoolBatch tokens worth.
    int totalCBs = decodeCbPoolBatch * nGroups;
    ps.cbPool.resize(totalCBs);
    ps.cbIdx = 0;

    // Compute group boundaries: split per-slot dispatches into nGroups chunks
    auto& dispatches = ps.dispatches;
    int total = (int)dispatches.size();
    std::vector<int> groupStart(nGroups + 1);
    for (int g = 0; g <= nGroups; g++)
        groupStart[g] = g * total / nGroups;

    auto encodeGroup = [&](int gBegin, int gEnd, bool addCopy) -> WGPUCommandBuffer {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

        if (passPerDispatch) {
            for (int d = gBegin; d < gEnd; d++) {
                auto& di = dispatches[d];
                WGPUComputePassDescriptor cpD{};
                auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
                wgpuComputePassEncoderSetPipeline(pass, di.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, di.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, di.gx, di.gy, di.gz);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
            }
        } else {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            for (int d = gBegin; d < gEnd; d++) {
                auto& di = dispatches[d];
                wgpuComputePassEncoderSetPipeline(pass, di.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, di.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, di.gx, di.gy, di.gz);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        if (addCopy)
            wgpuCommandEncoderCopyBufferToBuffer(enc, argmaxResultBuf.handle, 0,
                                                  ps.stagingBuf, 0, 4);

        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuCommandEncoderRelease(enc);
        return cb;
    };

    for (int i = 0; i < decodeCbPoolBatch; i++) {
        for (int g = 0; g < nGroups; g++) {
            bool isLast = (g == nGroups - 1);
            ps.cbPool[i * nGroups + g] = encodeGroup(
                groupStart[g], groupStart[g + 1], isLast);
        }
    }
}

void ModelRunner::autotuneDecodeDepth() {
    if (!gpu || gpu->backendType != WGPUBackendType_D3D12) return;
    if (decodePoolCapacity < 3) return;

    const int originalDepth = decodePoolDepth;
    const int maxDepth = std::min(decodePoolCapacity, 8);
    const int nTokens = 24;
    const double epsilon = 0.015;
    double bestMsPerTok = 1e30;
    int bestDepth = decodePoolDepth;

    for (int depth = 2; depth <= maxDepth; depth++) {
        double msPerTok = benchmarkDecodeConfig(depth, nTokens, 2);
        if (msPerTok < bestMsPerTok * (1.0 - epsilon) ||
            (fabs(msPerTok - bestMsPerTok) <= bestMsPerTok * epsilon && depth < bestDepth)) {
            bestMsPerTok = msPerTok;
            bestDepth = depth;
        }
    }

    decodePoolDepth = bestDepth;
    for (int s = 0; s < decodePoolDepth; s++)
        refillCBPool(s);
    resetKVCache();

    fprintf(stderr, "  Decode depth autotune: %d -> %d (%.2f ms/tok)\n",
           originalDepth, decodePoolDepth, bestMsPerTok);
}

double ModelRunner::benchmarkDecodeConfig(int depth, int nTokens, int repeats) {
    int32_t seedToken = 0;
    double totalMsPerTok = 0.0;
    for (int rep = 0; rep < repeats; rep++) {
        resetKVCache();
        gpu->writeBuffer(argmaxResultBuf, &seedToken, 4);
        for (int s = 0; s < depth; s++)
            refillCBPool(s);

        auto t0 = std::chrono::steady_clock::now();
        int submitted = 0;
        int completed = 0;
        int primeCount = std::min(depth, nTokens);
        for (int i = 0; i < primeCount; i++) {
            submitDecode((uint32_t)i, i);
            submitted++;
        }
        while (completed < submitted) {
            int slot = completed % depth;
            (void)readArgmax(slot);
            completed++;
            if (submitted < nTokens) {
                submitDecode((uint32_t)submitted, slot);
                submitted++;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        totalMsPerTok += std::chrono::duration<double, std::milli>(t1 - t0).count() / nTokens;
    }
    return totalMsPerTok / std::max(repeats, 1);
}

void ModelRunner::applyDecodeKernelSelection(bool useFastQkv, bool useFastOproj,
                                             bool useFastGateup) {
    tuning.decodeUseFastQkv = useFastQkv && decodeFastVariantsAvailable;
    tuning.decodeUseFastOproj = useFastOproj && decodeFastVariantsAvailable;
    tuning.decodeUseFastGateup = useFastGateup && decodeFastVariantsAvailable;

    const auto& plQ8Matmul = getKernel("q8_matmul");
    const auto& plQ8MatmulNorm = getKernel("q8_matmul_norm");
    const auto& plQ8Fast = getKernel("q8_matmul_fast");
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& di = decodeDispatchIndices[i];
        auto& vbg = decodeVariantBGs[i];

        if (di.qkv >= 0) {
            auto& d = allDecodeDispatches[di.qkv];
            d.pipeline = tuning.decodeUseFastQkv && vbg.qkvFast ? plQ8Fast.pipeline : plQ8MatmulNorm.pipeline;
            d.bindGroup = tuning.decodeUseFastQkv && vbg.qkvFast ? vbg.qkvFast : vbg.qkvBase;
            autoDecodeDispatches[di.qkv + autoDecodePrefixCount] = d;
        }
        if (di.oproj >= 0) {
            auto& d = allDecodeDispatches[di.oproj];
            d.pipeline = tuning.decodeUseFastOproj && vbg.oprojFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastOproj && vbg.oprojFast ? vbg.oprojFast : vbg.oprojBase;
            autoDecodeDispatches[di.oproj + autoDecodePrefixCount] = d;
        }
        if (di.gateup >= 0) {
            auto& d = allDecodeDispatches[di.gateup];
            d.pipeline = tuning.decodeUseFastGateup && vbg.gateupFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastGateup && vbg.gateupFast ? vbg.gateupFast : vbg.gateupBase;
            autoDecodeDispatches[di.gateup + autoDecodePrefixCount] = d;
        }
    }

    for (int s = 0; s < decodePoolDepth; s++) {
        auto& ps = pool[s];
        if (!ps.dispatches.empty()) {
            for (uint32_t i = 0; i < cfg.nLayer; i++) {
                auto& di = decodeDispatchIndices[i];
                if (di.qkv >= 0)
                    ps.dispatches[di.qkv + autoDecodePrefixCount] = autoDecodeDispatches[di.qkv + autoDecodePrefixCount];
                if (di.oproj >= 0)
                    ps.dispatches[di.oproj + autoDecodePrefixCount] = autoDecodeDispatches[di.oproj + autoDecodePrefixCount];
                if (di.gateup >= 0)
                    ps.dispatches[di.gateup + autoDecodePrefixCount] = autoDecodeDispatches[di.gateup + autoDecodePrefixCount];
            }
        }
        refillCBPool(s);
    }
}

void ModelRunner::autotuneDecodeKernels() {
    if (!gpu || gpu->backendType != WGPUBackendType_D3D12) return;
    if (!decodeFastVariantsAvailable) return;

    const int nTokens = 12;
    const double epsilon = 0.015;
    double bestMsPerTok = 1e30;
    bool bestQkv = tuning.decodeUseFastQkv;
    bool bestOproj = tuning.decodeUseFastOproj;
    bool bestGateup = tuning.decodeUseFastGateup;

    for (int mask = 0; mask < 8; mask++) {
        bool useFastQkv = (mask & 1) != 0;
        bool useFastOproj = (mask & 2) != 0;
        bool useFastGateup = (mask & 4) != 0;
        applyDecodeKernelSelection(useFastQkv, useFastOproj, useFastGateup);
        double msPerTok = benchmarkDecodeConfig(decodePoolDepth, nTokens, 1);
        int currentFastCount = (int)useFastQkv + (int)useFastOproj + (int)useFastGateup;
        int bestFastCount = (int)bestQkv + (int)bestOproj + (int)bestGateup;
        if (msPerTok < bestMsPerTok * (1.0 - epsilon) ||
            (fabs(msPerTok - bestMsPerTok) <= bestMsPerTok * epsilon && currentFastCount < bestFastCount)) {
            bestMsPerTok = msPerTok;
            bestQkv = useFastQkv;
            bestOproj = useFastOproj;
            bestGateup = useFastGateup;
        }
    }

    applyDecodeKernelSelection(bestQkv, bestOproj, bestGateup);
    resetKVCache();
    fprintf(stderr, "  Decode kernel autotune: qkv=%s oproj=%s gateup=%s (%.2f ms/tok)\n",
           bestQkv ? "fast" : "base",
           bestOproj ? "fast" : "base",
           bestGateup ? "fast" : "base",
           bestMsPerTok);
}

std::string ModelRunner::decodeAutotuneCachePath() const {
    namespace fs = std::filesystem;
    fs::path modelPath(ggufPath);
    fs::path modelDir = modelPath.parent_path();
    std::string modelName = modelDir.filename().string();
    if (modelName == "weights")
        modelName = modelDir.parent_path().filename().string();

    fs::path repoRoot;
    for (auto p = fs::absolute(modelPath); !p.empty() && p != p.parent_path(); p = p.parent_path()) {
        if (fs::exists(p / ".gitignore") && fs::exists(p / "runtimes")) {
            repoRoot = p;
            break;
        }
    }
    if (repoRoot.empty())
        return {};

    fs::path cacheDir = repoRoot / "gitignore" / "models" / modelName;
    fs::create_directories(cacheDir);
    return (cacheDir / ("decode_autotune_" + backendName(gpu->backendType) + ".txt")).string();
}

std::string ModelRunner::decodeAutotuneCacheKey() const {
    std::ostringstream oss;
    oss << "backend=" << backendName(gpu->backendType)
        << ";gpu=" << gpu->adapterName
        << ";desc=" << gpu->adapterDescription
        << ";arch=" << cfg.arch
        << ";layers=" << cfg.nLayer
        << ";embd=" << cfg.nEmbd
        << ";head_dim=" << cfg.headDim
        << ";kv_heads=" << cfg.nKvHeads
        << ";intermediate=" << cfg.intermediateSize
        << ";invocations=" << effectiveLimits(*gpu).maxComputeInvocationsPerWorkgroup
        << ";wgmem=" << effectiveLimits(*gpu).maxComputeWorkgroupStorageSize
        << ";pool_cap=" << decodePoolCapacity
        << ";batch=" << decodeCbPoolBatch;
    return oss.str();
}

bool ModelRunner::loadDecodeAutotuneCache() {
    std::string path = decodeAutotuneCachePath();
    if (path.empty()) return false;

    std::ifstream in(path);
    if (!in) return false;

    std::string key;
    std::string line;
    int cachedDepth = -1;
    int cachedQkv = -1, cachedOproj = -1, cachedGateup = -1;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string name = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        if (name == "key") key = value;
        else if (name == "depth") cachedDepth = atoi(value.c_str());
        else if (name == "qkv_fast") cachedQkv = atoi(value.c_str());
        else if (name == "oproj_fast") cachedOproj = atoi(value.c_str());
        else if (name == "gateup_fast") cachedGateup = atoi(value.c_str());
    }

    if (key != decodeAutotuneCacheKey()) return false;
    if (cachedDepth < 2 || cachedDepth > decodePoolCapacity) return false;
    if (cachedQkv < 0 || cachedOproj < 0 || cachedGateup < 0) return false;

    decodePoolDepth = cachedDepth;
    applyDecodeKernelSelection(cachedQkv != 0, cachedOproj != 0, cachedGateup != 0);
    resetKVCache();
    fprintf(stderr, "  Decode autotune cache: loaded from %s\n", path.c_str());
    return true;
}

void ModelRunner::saveDecodeAutotuneCache() const {
    std::string path = decodeAutotuneCachePath();
    if (path.empty()) return;

    std::ofstream out(path, std::ios::trunc);
    if (!out) return;
    out << "key=" << decodeAutotuneCacheKey() << "\n";
    out << "depth=" << decodePoolDepth << "\n";
    out << "qkv_fast=" << (tuning.decodeUseFastQkv ? 1 : 0) << "\n";
    out << "oproj_fast=" << (tuning.decodeUseFastOproj ? 1 : 0) << "\n";
    out << "gateup_fast=" << (tuning.decodeUseFastGateup ? 1 : 0) << "\n";
}

void ModelRunner::printActiveDecodeTuning(const char* prefix) const {
    fprintf(stderr, "%s: depth=%d/%d qkv=%s oproj=%s gateup=%s lm_head=%s batch=%d\n",
           prefix,
           decodePoolDepth,
           decodePoolCapacity,
           tuning.decodeUseFastQkv ? "fast" : "base",
           tuning.decodeUseFastOproj ? "fast" : "base",
           tuning.decodeUseFastGateup ? "fast" : "base",
           tuning.decodeUseWideFp16 ? "fp16_wide" : "fp16",
           decodeCbPoolBatch);
}

void ModelRunner::destroy() {
    if (!gpu) return;

    std::unordered_set<void*> releasedBindGroups;
    auto releaseBG = [&](WGPUBindGroup& bg) {
        if (!bg) return;
        void* key = reinterpret_cast<void*>(bg);
        if (releasedBindGroups.insert(key).second)
            wgpuBindGroupRelease(bg);
        bg = nullptr;
    };

    for (auto& d : allDecodeDispatches)
        releaseBG(d.bindGroup);
    for (auto& d : autoDecodeDispatches)
        releaseBG(d.bindGroup);

    for (auto& bg : decodeVariantBGs) {
        releaseBG(bg.qkvBase);
        releaseBG(bg.qkvFast);
        releaseBG(bg.oprojBase);
        releaseBG(bg.oprojFast);
        releaseBG(bg.gateupBase);
        releaseBG(bg.gateupFast);
    }

    for (auto& bg : pfCache.layerBGs) {
        releaseBG(bg.rms);
        releaseBG(bg.qnorm);
        releaseBG(bg.qkv);
        releaseBG(bg.rope);
        releaseBG(bg.attn);
        releaseBG(bg.attnq);
        releaseBG(bg.oproj);
        releaseBG(bg.addrms);
        releaseBG(bg.gateup);
        releaseBG(bg.siluq);
        releaseBG(bg.downsilu);
    }
    releaseBG(pfCache.finalRmsBG);
    releaseBG(pfCache.lmBG);
    releaseBG(pfCache.argmaxBG);

    for (auto& slot : pool) {
        for (int i = slot.cbIdx; i < (int)slot.cbPool.size(); i++) {
            if (slot.cbPool[i])
                wgpuCommandBufferRelease(slot.cbPool[i]);
        }
        slot.cbPool.clear();
        slot.cbIdx = 0;
        if (slot.stagingBuf) {
            wgpuBufferRelease(slot.stagingBuf);
            slot.stagingBuf = nullptr;
        }
        if (slot.ropeParamsBuf.handle) {
            wgpuBufferRelease(slot.ropeParamsBuf.handle);
            slot.ropeParamsBuf.handle = nullptr;
        }
        if (slot.attnParamsBuf.handle) {
            wgpuBufferRelease(slot.attnParamsBuf.handle);
            slot.attnParamsBuf.handle = nullptr;
        }
        slot.dispatches.clear();
    }
    pool.clear();

    if (profiler) {
        profiler->destroy();
        delete profiler;
        profiler = nullptr;
    }
}

// ─── Inference ───────────────────────────────────────────────────────────────

void ModelRunner::uploadEmbedding(int32_t tokenId) {
    if (tokenId < 0 || (uint32_t)tokenId >= cfg.nVocab) tokenId = 0;
    const float* emb = embeddingCPU.data() + tokenId * cfg.nEmbd;
    if (cfg.embeddingScale > 0.0f) {
        std::vector<float> scaled(cfg.nEmbd);
        for (uint32_t i = 0; i < cfg.nEmbd; i++)
            scaled[i] = emb[i] * cfg.embeddingScale;
        gpu->writeBuffer(xBuf, scaled.data(), cfg.nEmbd * 4);
    } else {
        gpu->writeBuffer(xBuf, emb, cfg.nEmbd * 4);
    }

    // PLE: compute per-layer embedding slices for this token
    if (cfg.pleSize > 0 && !pleEmbCPU.empty() && !pleSliceBufs.empty()) {
        uint32_t pleDim = cfg.pleSize;
        uint32_t totalPleDim = pleDim * cfg.nLayer;
        float pleScale = sqrtf((float)pleDim);
        float embInvScale = 1.0f / sqrtf((float)cfg.nEmbd);

        // 1. Look up per-layer token embeddings: pleEmbCPU[tokenId * totalPleDim .. +totalPleDim]
        //    Shape: [totalPleDim] = [nLayer * pleDim]
        std::vector<float> pleEmbs(totalPleDim);
        if ((size_t)tokenId * totalPleDim + totalPleDim <= pleEmbCPU.size()) {
            for (uint32_t j = 0; j < totalPleDim; j++)
                pleEmbs[j] = pleEmbCPU[(size_t)tokenId * totalPleDim + j] * pleScale;
        }

        // 2. Upload per-layer slices to GPU
        for (uint32_t li = 0; li < cfg.nLayer && li < (uint32_t)pleSliceBufs.size(); li++) {
            gpu->writeBuffer(pleSliceBufs[li],
                             pleEmbs.data() + li * pleDim, pleDim * 4);
        }
    }
}

void ModelRunner::updateDecodeParams(uint32_t pos, uint32_t cacheLen) {
    auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
    p[3] = pos;
    p[5] = cacheLen * cfg.nKvHeads * cfg.headDim;
    gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);

    uint32_t T_total = cacheLen + 1;
    uint32_t n_chunks = (T_total + gqaChunkSize - 1) / gqaChunkSize;
    auto* cp = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
    cp[2] = T_total;
    cp[4] = n_chunks;
    gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);

    // SWA: clamp T_total to sliding window size
    if (chunkedAttnParamsBufSWA.handle && cfg.slidingWindow > 0) {
        uint32_t T_swa = std::min(T_total, cfg.slidingWindow);
        uint32_t n_chunks_swa = (T_swa + gqaChunkSize - 1) / gqaChunkSize;
        uint8_t swaData[32];
        memcpy(swaData, chunkedAttnParamData.data(), 32);
        auto* sp = reinterpret_cast<uint32_t*>(swaData);
        sp[2] = T_swa;
        sp[4] = n_chunks_swa;
        gpu->writeBuffer(chunkedAttnParamsBufSWA, swaData, 32);
    }
}

void ModelRunner::prepareDecodeParams(uint32_t pos, uint32_t cacheLen, int slot) {
    auto& ps = pool[slot];

    uint8_t localRope[32], localAttn[32];
    memcpy(localRope, ropeParamData.data(), 32);
    auto* p = reinterpret_cast<int32_t*>(localRope);
    p[3] = pos;
    p[5] = cacheLen * cfg.nKvHeads * cfg.headDim;
    gpu->writeBuffer(ps.ropeParamsBuf, localRope, 32);

    uint32_t T_total = cacheLen + 1;
    uint32_t n_chunks = (T_total + gqaChunkSize - 1) / gqaChunkSize;
    memcpy(localAttn, chunkedAttnParamData.data(), 32);
    auto* cp = reinterpret_cast<uint32_t*>(localAttn);
    cp[2] = T_total;
    cp[4] = n_chunks;
    gpu->writeBuffer(ps.attnParamsBuf, localAttn, 32);

    // SWA per-slot param buffers
    if (ps.attnParamsBufSWA.handle && cfg.slidingWindow > 0) {
        uint32_t T_swa = std::min(T_total, cfg.slidingWindow);
        uint32_t n_chunks_swa = (T_swa + gqaChunkSize - 1) / gqaChunkSize;
        uint8_t swaAttn[32];
        memcpy(swaAttn, chunkedAttnParamData.data(), 32);
        auto* sp = reinterpret_cast<uint32_t*>(swaAttn);
        sp[2] = T_swa;
        sp[4] = n_chunks_swa;
        gpu->writeBuffer(ps.attnParamsBufSWA, swaAttn, 32);
    }
}

std::vector<float> ModelRunner::decode(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);

    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    std::vector<uint8_t> result;
    if (profiler && profiler->enabled()) {
        result = gpu->submitAndReadbackProfiled(
            allDecodeDispatches, logitsBuf, cfg.nVocab * 4, *profiler);
    } else {
        result = gpu->submitAndReadback(
            allDecodeDispatches, logitsBuf, cfg.nVocab * 4, passPerDispatch);
    }

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    std::vector<float> logits(cfg.nVocab);
    memcpy(logits.data(), result.data(), cfg.nVocab * 4);

    return logits;
}

int32_t ModelRunner::decodeArgmax(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);

    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    auto result = gpu->submitAndReadback(
        allDecodeDispatches, argmaxResultBuf, 4, passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    int32_t token;
    memcpy(&token, result.data(), 4);
    return token;
}

void ModelRunner::submitDecode(uint32_t posOffset, int slot) {
    auto& ps = pool[slot];
    uint32_t cacheLen = kvCache[0].len;
    prepareDecodeParams(posOffset, cacheLen, slot);

    // Refill pool if exhausted
    if (ps.cbIdx >= (int)ps.cbPool.size())
        refillCBPool(slot);

    using hrc = std::chrono::high_resolution_clock;
    auto t0 = hrc::now();

    // Submit nGroups pre-recorded CBs for this token
    for (int g = 0; g < nGroups; g++) {
        WGPUCommandBuffer cb = ps.cbPool[ps.cbIdx++];
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
    }

    auto t1 = hrc::now();
    gpu->timing.submit_ns += (t1 - t0).count();

    // Start async map (non-blocking until WaitAny is called)
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void*) {};
    ps.pendingFuture = wgpuBufferMapAsync(ps.stagingBuf, 1, 0, 4, mcb);

    auto t2 = hrc::now();
    gpu->timing.map_start_ns += (t2 - t1).count();
    gpu->timing.count++;

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;
}

int32_t ModelRunner::readArgmax(int slot) {
    auto& ps = pool[slot];
    return gpu->completeAsyncMapI32(ps.stagingBuf, ps.pendingFuture);
}

void ModelRunner::prefillStep(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);
    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    gpu->submitOnly(allDecodeDispatches, !passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;
}

std::vector<float> ModelRunner::prefillFinish(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);
    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    // Submit and read back logits (blocking — only for the last prefill token)
    auto result = gpu->submitAndReadback(
        allDecodeDispatches, logitsBuf, cfg.nVocab * 4, passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    std::vector<float> logits(cfg.nVocab);
    memcpy(logits.data(), result.data(), cfg.nVocab * 4);
    return logits;
}

int32_t ModelRunner::prefillBatched(
        const int32_t* tokenIds, uint32_t T, uint32_t posOffset) {
    // For small T or when prefill resources are not available (K-quant), use serial path
    if (T <= 16 || !pfCache.pX.handle) {
        fprintf(stderr, "  [prefillBatched] serial path T=%u\n", T); fflush(stderr);
        if (T == 1) {
            auto logits = decode(tokenIds[0], posOffset);
            return argmax(logits);
        }
        // Use synchronous decode for each token (safer than fire-and-forget prefillStep)
        std::vector<float> logits;
        for (uint32_t t = 0; t < T; t++)
            logits = decode(tokenIds[t], posOffset + t);
        return argmax(logits);
    }

    // ── True batched prefill using pre-allocated resources ───────────
    using hrc = std::chrono::high_resolution_clock;
    auto t_start = hrc::now();

    uint32_t qDimL  = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;
    uint32_t Q8_TILE = 8, TILE_M = 8;
    bool usePrequant = !useMMA && useDP4A;

    // Upload embeddings into pre-allocated pX
    std::vector<float> embData(T * cfg.nEmbd);
    for (uint32_t t = 0; t < T; t++) {
        int32_t tok = (tokenIds[t] >= 0 && (uint32_t)tokenIds[t] < cfg.nVocab)
                      ? tokenIds[t] : 0;
        memcpy(embData.data() + t * cfg.nEmbd,
               embeddingCPU.data() + tok * cfg.nEmbd, cfg.nEmbd * 4);
    }
    if (cfg.embeddingScale > 0.0f) {
        for (size_t i = 0; i < embData.size(); i++)
            embData[i] *= cfg.embeddingScale;
    }
    gpu->writeBuffer(pfCache.pX, embData.data(), T * cfg.nEmbd * 4);

    // Write T-dependent params into pre-allocated param buffers
    {
        uint32_t v[4];
        v[0] = T; v[1] = qkvOutL; v[2] = cfg.nEmbd; v[3] = 0;
        gpu->writeBuffer(pfCache.pQkvP, v, 16);
        v[0] = T; v[1] = cfg.nEmbd; v[2] = qDimL;
        gpu->writeBuffer(pfCache.pOpP, v, 16);
        v[0] = T; v[1] = 2 * cfg.intermediateSize; v[2] = cfg.nEmbd;
        gpu->writeBuffer(pfCache.pGuP, v, 16);
        v[0] = cfg.intermediateSize; v[1] = cfg.nEmbd; v[2] = cfg.intermediateSize; v[3] = T;
        gpu->writeBuffer(pfCache.pDnP, v, 16);

        if (usePrequant) {
            v[0] = T; v[1] = cfg.nEmbd; v[2] = 0; v[3] = 0;
            gpu->writeBuffer(pfCache.pNormQP, v, 16);
            v[0] = T; v[1] = qDimL; v[2] = 0; v[3] = 0;
            gpu->writeBuffer(pfCache.pAttnQP, v, 16);
            v[0] = T; v[1] = cfg.intermediateSize; v[2] = 0; v[3] = 0;
            gpu->writeBuffer(pfCache.pGUQP, v, 16);
        }
    }

    uint32_t cacheLen = kvCache[0].len;

    // Write per-layer RoPE + attention params
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        uint32_t ropeP[8] = {};
        ropeP[0] = cfg.nHead;  ropeP[1] = qDimL;  ropeP[2] = kvDimL;
        ropeP[3] = posOffset;  ropeP[4] = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        ropeP[5] = cacheLen;
        float eps_f = cfg.rmsNormEps; memcpy(&ropeP[6], &eps_f, 4);
        ropeP[7] = cfg.nKvHeads;
        gpu->writeBuffer(pfCache.ropeParams[li], ropeP, 32);

        uint32_t T_total = cacheLen + T;
        uint32_t ap[8] = {};
        ap[0] = cfg.nKvHeads * cfg.headDim;
        ap[1] = cfg.nHead / cfg.nKvHeads;
        ap[2] = T_total;  ap[3] = cacheLen;
        ap[4] = T;  // T_prefill
        float sc = 1.0f / sqrtf((float)cfg.headDim);
        float ni = -1e9f;
        memcpy(&ap[5], &sc, 4);  memcpy(&ap[6], &ni, 4);
        gpu->writeBuffer(pfCache.attnParams[li], ap, 32);
    }

    auto t_params = hrc::now();

    // Write grid sizes into pre-allocated indirect dispatch buffer
    const uint32_t MAT_M = tuning.prefillMatM;
    const uint32_t MAT_N = tuning.prefillMatN;
    const uint32_t MAT_WIDE_M = tuning.prefillWideMatM;
    const uint32_t MAT_WIDE_N = tuning.prefillWideMatN;
    const uint32_t DS_M  = tuning.prefillDnM;
    const uint32_t DS_N  = tuning.prefillDnN;
    const uint32_t ATTN_BQ = tuning.prefillAttnBlockQ;
    const uint32_t QN_NORM = (cfg.nEmbd + 255u) / 256u;
    const uint32_t QN_ATTN = (qDimL + 255u) / 256u;
    const uint32_t QN_GU   = (cfg.intermediateSize + 255u) / 256u;

    uint32_t perLayerDispatches = usePrequant ? 12u : 8u;
    uint32_t nDispatches = cfg.nLayer * perLayerDispatches + 1;
    std::vector<uint32_t> grids(nDispatches * 3);
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        uint32_t base = li * perLayerDispatches * 3;
        uint32_t idx = base;
        auto emit = [&](uint32_t gx, uint32_t gy, uint32_t gz) {
            grids[idx + 0] = gx;
            grids[idx + 1] = gy;
            grids[idx + 2] = gz;
            idx += 3;
        };

        emit(T, 1, 1);
        if (usePrequant) emit(QN_NORM, T, 1);
           emit((qkvOutL + MAT_WIDE_N - 1) / MAT_WIDE_N,
               (T + MAT_WIDE_M - 1) / MAT_WIDE_M, 1);
        emit(cfg.nHead + cfg.nKvHeads, T, 1);
        emit(cfg.nHead, (T + ATTN_BQ - 1) / ATTN_BQ, 1);
        if (usePrequant) emit(QN_ATTN, T, 1);
        emit((cfg.nEmbd + MAT_N - 1) / MAT_N, (T + MAT_M - 1) / MAT_M, 1);
        emit(T, 1, 1);
           if (usePrequant) emit(QN_NORM, T, 1);
           emit((2 * cfg.intermediateSize + MAT_WIDE_N - 1) / MAT_WIDE_N,
               (T + MAT_WIDE_M - 1) / MAT_WIDE_M, 1);
        if (usePrequant) emit(QN_GU, T, 1);
        emit((cfg.nEmbd + DS_N - 1) / DS_N, (T + DS_M - 1) / DS_M, 1);
    }
    uint32_t fb = cfg.nLayer * perLayerDispatches * 3;
    grids[fb + 0] = T; grids[fb + 1] = 1; grids[fb + 2] = 1;

    gpu->writeBuffer(pfCache.indirectBuf, grids.data(), nDispatches * 12);

    // Submit all prefill dispatches in one shot
    auto t_dispatches = hrc::now();

    // Write LM head params before submission (needed in the same command buffer)
    {
        uint32_t v[4] = {cfg.nEmbd, cfg.nVocab, 1, 0};
        gpu->writeBuffer(pfCache.pLmP, v, 16);
    }

    std::vector<uint8_t> result;

    if (profiler && profiler->enabled()) {
        // Profiler path: use direct dispatch (needs per-dispatch compute passes)
        std::vector<Dispatch> allPrefill;
        allPrefill.reserve(nDispatches);
        for (uint32_t i = 0; i < nDispatches; i++) {
            auto& e = pfCache.indirectTable[i];
            allPrefill.push_back({e.pipeline, e.bindGroup,
                grids[i*3], grids[i*3+1], grids[i*3+2], e.name});
        }
        gpu->submitOnlyProfiled(allPrefill, *profiler);

        // Copy last token's norm output → normOutBuf for LM head
        {
            WGPUCommandEncoderDescriptor enD{};
            auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                pfCache.pNorm.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
                normOutBuf.handle, 0, cfg.nEmbd * 4);
            WGPUCommandBufferDescriptor cbD{};
            auto cb = wgpuCommandEncoderFinish(enc, &cbD);
            wgpuQueueSubmit(gpu->queue, 1, &cb);
            wgpuCommandEncoderRelease(enc);
            wgpuCommandBufferRelease(cb);
        }

        // LM head + softcap + argmax
        std::vector<Dispatch> lmArgmax;
        if (lmHeadIsQ8) {
            lmArgmax.push_back({getKernel("q8_matmul").pipeline, pfCache.lmBG,
                1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1, "pf_lm"});
        }
        if (softcapPipeline) {
            lmArgmax.push_back({softcapPipeline, softcapBG,
                softcapDispatchX, 1, 1, "pf_softcap"});
        }
        lmArgmax.push_back({getKernel("argmax").pipeline, pfCache.argmaxBG,
            1, 1, 1, "pf_argmax"});
        result = gpu->submitAndReadbackProfiled(lmArgmax, argmaxResultBuf, 4,
                                                 *profiler);
    } else {
        // Fast path: indirect dispatch — pre-recorded pipeline/BG, only grid size buffer changes
        auto rb = gpu->getOrCreateReadbackBuf(4);

        auto t_enc0 = hrc::now();

        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

        // 1. All prefill dispatches in a single compute pass using indirect dispatch
        {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            for (auto& e : pfCache.indirectTable) {
                wgpuComputePassEncoderSetPipeline(pass, e.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, e.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroupsIndirect(
                    pass, pfCache.indirectBuf.handle, e.indirectOffset);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 2. Copy last token's norm output → normOutBuf
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pfCache.pNorm.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
            normOutBuf.handle, 0, cfg.nEmbd * 4);

        // 3. LM head + argmax in another compute pass
        {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            if (lmHeadIsQ8) {
                wgpuComputePassEncoderSetPipeline(pass, getKernel("q8_matmul").pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, pfCache.lmBG, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, 1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1);
            }
            if (softcapPipeline) {
                wgpuComputePassEncoderSetPipeline(pass, softcapPipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, softcapBG, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, softcapDispatchX, 1, 1);
            }
            wgpuComputePassEncoderSetPipeline(pass, getKernel("argmax").pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, pfCache.argmaxBG, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 4. Copy argmax result for readback
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            argmaxResultBuf.handle, 0, rb, 0, 4);

        // 5. Copy last hidden state → xBuf for decode
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pfCache.pX.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
            xBuf.handle, 0, cfg.nEmbd * 4);

        auto t_enc1 = hrc::now();  // encoding done

        // Single submit
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);

        auto t_finish = hrc::now();  // finish (Dawn barrier computation) done

        wgpuQueueSubmit(gpu->queue, 1, &cb);

        auto t_submitted = hrc::now();  // submit done

        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);

        // Synchronous map of 4 bytes
        struct { bool done; uint32_t status; } ms{false, 0};
        WGPUBufferMapCallbackInfo mcb{};
        mcb.mode = WGPUCallbackMode_WaitAnyOnly;
        mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
            auto* p = static_cast<decltype(&ms)>(u);
            p->done = true; p->status = s;
        };
        mcb.userdata1 = &ms;
        auto mf = wgpuBufferMapAsync(rb, 1 /*READ*/, 0, 4, mcb);
        WGPUFutureWaitInfo mw{mf, 0};
        wgpuInstanceWaitAny(gpu->instance, 1, &mw, UINT64_MAX);

        auto t_gpudone = hrc::now();  // GPU finished + readback mapped

        result.resize(4);
        if (ms.status == 1) {
            auto ptr = wgpuBufferGetConstMappedRange(rb, 0, 4);
            memcpy(result.data(), ptr, 4);
            wgpuBufferUnmap(rb);
        }

        // Detailed CPU timing breakdown
        auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        fprintf(stderr, "    cpu: encode=%.0fus finish=%.0fus submit=%.0fus gpu_wait=%.0fus\n",
               us(t_enc0, t_enc1), us(t_enc1, t_finish),
               us(t_finish, t_submitted), us(t_submitted, t_gpudone));
    }
    auto t_submit = hrc::now();

    // Print prefill timing breakdown
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "  [prefill T=%u] params=%.1fms build=%.1fms gpu+readback=%.1fms total=%.1fms (%zu dispatches)\n",
           T, ms(t_start, t_params), ms(t_params, t_dispatches),
           ms(t_dispatches, t_submit),
           ms(t_start, t_submit), pfCache.indirectTable.size());

    // Update KV cache lengths
    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len += T;

    // Copy last hidden state to xBuf for subsequent decode (profiler path only;
    // non-profiled path already includes this in the consolidated encoder)
    if (profiler && profiler->enabled()) {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pfCache.pX.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
            xBuf.handle, 0, cfg.nEmbd * 4);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);
    }

    int32_t tokId;
    memcpy(&tokId, result.data(), 4);
    return tokId;
}
int32_t ModelRunner::argmax(const std::vector<float>& logits) {
    return (int32_t)std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end()));
}

void ModelRunner::resetKVCache() {
    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len = 0;
}

void ModelRunner::enableProfiling() {
    profiler = new GPUProfiler();
    if (!profiler->init(gpu->device, gpu->instance, gpu->queue)) {
        fprintf(stderr, "WARNING: GPU timestamp queries not available\n");
        delete profiler;
        profiler = nullptr;
        return;
    }

    // Acquire CPU<->GPU clock calibration
    auto cal = acquireClockCalibration(gpu->device, gpu->backendType);
    if (cal.valid) {
        calibration = new ClockCalibration(cal);
        fprintf(stderr, "  Clock calibration: GPU=%llu ns, CPU=%llu ns, deviation=%llu ns\n",
               (unsigned long long)cal.gpuTimestampNs,
               (unsigned long long)cal.cpuTimestampNs,
               (unsigned long long)cal.maxDeviationNs);
    } else {
        fprintf(stderr, "  Clock calibration: not available (GPU-only timing)\n");
    }
}


void ModelRunner::printProfileReport(int nDecodeTokens, int nPrefillTokens,
                                     double prefillMs, double decodeMs,
                                     const std::string& profileOutputPath) {
    if (!profiler || !profiler->enabled() || profiler->nextIndex == 0)
        return;

    // Map the profiler's readback buffer to get timestamp values
    uint32_t count = profiler->nextIndex;
    uint64_t readSize = count * 8;

    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(profiler->readbackBuf, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(gpu->instance, 1, &mw, UINT64_MAX);

    if (ms.status != 1) {
        fprintf(stderr, "Failed to map profiler readback buffer\n");
        return;
    }

    auto ptr = (const uint64_t*)wgpuBufferGetConstMappedRange(
        profiler->readbackBuf, 0, readSize);

    // Compute GPU frequency (nanoseconds per tick)
    // Vulkan timestamps are in nanoseconds on most drivers
    // but on some may use the device's timestamp period.
    // Dawn normalizes to nanoseconds for us.

    // Aggregate by kernel name (strip layer prefix)
    struct AggEntry {
        double totalUs = 0;
        uint32_t count = 0;
    };
    std::unordered_map<std::string, AggEntry> agg;
    double totalGpuUs = 0;

    for (auto& e : profiler->entries) {
        uint64_t begin = ptr[e.beginIdx];
        uint64_t end   = ptr[e.endIdx];
        // Skip invalid timestamps (uninitialized or wrapped)
        if (end <= begin || begin == 0) continue;
        double durNs = (double)(end - begin);
        double durUs = durNs / 1000.0;

        // Strip layer prefix: "L5/q8_qkv" -> "q8_qkv"
        std::string kernel = e.name;
        auto slash = kernel.find('/');
        if (slash != std::string::npos)
            kernel = kernel.substr(slash + 1);

        agg[kernel].totalUs += durUs;
        agg[kernel].count++;
        totalGpuUs += durUs;
    }

    // Sort by total time descending
    std::vector<std::pair<std::string, AggEntry>> sorted_agg(agg.begin(), agg.end());
    std::sort(sorted_agg.begin(), sorted_agg.end(),
              [](auto& a, auto& b) { return a.second.totalUs > b.second.totalUs; });

    // Print report
    fprintf(stderr, "\n--- GPU Profile (hardware timestamps) ---\n");
    fprintf(stderr, "%-20s %10s %6s %10s %6s\n",
           "Kernel", "Total(ms)", "Count", "Avg(us)", "%%");
    fprintf(stderr, "%-20s %10s %6s %10s %6s\n",
           "--------------------", "----------", "------", "----------", "------");
    for (auto& [name, e] : sorted_agg) {
        double totalMs = e.totalUs / 1000.0;
        double avgUs = e.totalUs / e.count;
        double pct = totalGpuUs > 0 ? e.totalUs / totalGpuUs * 100.0 : 0;
        fprintf(stderr, "%-20s %10.2f %6u %10.1f %5.1f%%\n",
               name.c_str(), totalMs, e.count, avgUs, pct);
    }
    fprintf(stderr, "%-20s %10.2f\n", "TOTAL", totalGpuUs / 1000.0);

    // Generate HTML timeline report
    std::string htmlPath = profileOutputPath;
    if (htmlPath.empty()) {
        // Default: place profile.html next to the GGUF file
        auto dir = std::filesystem::path(ggufPath).parent_path();
        htmlPath = (dir / "profile.html").string();
    }
    generateProfileHTML(*gpu, *profiler, calibration, ptr,
                        nDecodeTokens, nPrefillTokens,
                        prefillMs, decodeMs, htmlPath);

    wgpuBufferUnmap(profiler->readbackBuf);

    profiler->destroy();
    delete profiler;
    profiler = nullptr;
    if (calibration) { delete calibration; calibration = nullptr; }
}

// ─── MTP (Multi-Token Prediction) ─────────────────────────────────────────

int32_t ModelRunner::mtpDraft(int32_t lastToken, uint32_t pos,
                               std::vector<int32_t>& draftTokens) {
    if (mtpCfg.type == MTPType::None) return 0;
    if (!mtpWeights.preProjW.handle) return 0;

    draftTokens.clear();
    uint32_t numDraft = mtpCfg.numDraftTokens;

    // Buffers for MTP hidden state (reuse existing intermediate buffers where possible)
    // MTP hidden dim may differ from backbone hidden dim
    uint32_t mtpE = mtpCfg.hiddenSize > 0 ? mtpCfg.hiddenSize : cfg.nEmbd;

    for (uint32_t d = 0; d < numDraft; d++) {
        int32_t inputToken = (d == 0) ? lastToken : draftTokens.back();

        // 1. Embed the input token (reuse backbone embedding)
        if (inputToken < 0 || (uint32_t)inputToken >= cfg.nVocab) break;
        const float* emb = embeddingCPU.data() + inputToken * cfg.nEmbd;

        // 2. For Gemma 4: concat(embed * scale, backbone_hidden) → pre-project
        //    For Qwen 3.6: enorm(embed), hnorm(hidden) → concat → project
        //
        // The backbone's last hidden state is in xBuf after the most recent decode.
        // Read it back from GPU for the concat.
        //
        // NOTE: For full GPU implementation, we'd keep everything on GPU.
        // This CPU-side implementation is a correctness reference; GPU kernels
        // for the MTP head would be needed for production perf.

        // 3. Run MTP decoder layers (Q-only attention for Gemma 4, full for Qwen)
        // 4. Post-project back to backbone dim
        // 5. RMSNorm + LM head → argmax

        // For now: MTP weights must be loaded and pipeline built.
        // The full GPU pipeline for MTP requires:
        //   - Pre-projection matmul (2*E → mtpE)
        //   - Per-layer: RMSNorm, Q-proj, attention (shared KV), FFN
        //   - Post-projection matmul (mtpE → E)
        //   - Final norm + LM head matmul + argmax
        //
        // Each of these can reuse existing Q8 matmul / norm / attention kernels.
        // The Q-only attention (Gemma 4) reads K/V from the backbone's kvCache
        // but only computes Q from the MTP hidden state.

        // Placeholder: without MTP weights loaded, we can't draft
        break;
    }

    return (int32_t)draftTokens.size();
}

int32_t ModelRunner::mtpVerifyAndAccept(const std::vector<int32_t>& draftTokens,
                                         uint32_t pos, uint32_t& acceptedCount) {
    if (draftTokens.empty()) { acceptedCount = 0; return -1; }

    // Verification: run the backbone on all draft tokens in parallel
    // using the batched prefill path. This gives us logits for each position.
    //
    // Build input: [draft_0, draft_1, ..., draft_N-1]
    // Run prefillBatched to get the backbone's prediction at each position.
    // Compare each backbone prediction with the corresponding draft token.

    uint32_t T = (uint32_t)draftTokens.size();
    acceptedCount = 0;

    // Use prefillBatched if available, otherwise serial
    if (T == 1) {
        auto logits = decode(draftTokens[0], pos);
        int32_t backbone_pred = argmax(logits);
        if (backbone_pred == draftTokens[0]) {
            acceptedCount = 1;
            // Need one more decode to get the next token
            auto next_logits = decode(draftTokens[0], pos + 1);
            return argmax(next_logits);
        }
        return backbone_pred;
    }

    // For T > 1: ideally use batched prefill for parallel verification.
    // The backbone processes [draft_0, ..., draft_{T-1}] and returns
    // logits at each position. We accept greedily until mismatch.
    //
    // Since prefillBatched returns only the last token's argmax,
    // we need the serial path for proper verification:
    for (uint32_t i = 0; i < T; i++) {
        auto logits = decode(draftTokens[i], pos + i);
        int32_t backbone_pred = argmax(logits);

        if (i + 1 < T && backbone_pred != draftTokens[i + 1]) {
            // Mismatch at position i+1: accept tokens 0..i, return backbone's prediction
            acceptedCount = i + 1;
            return backbone_pred;
        }

        if (i + 1 == T) {
            // All drafts verified, return the bonus token
            acceptedCount = T;
            return backbone_pred;
        }
    }

    acceptedCount = 0;
    return -1;
}
