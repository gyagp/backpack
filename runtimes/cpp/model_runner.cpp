#include "model_runner.h"
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
        ? std::clamp(depth, 3, 4)
        : std::clamp(depth, 2, 6);
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

    // Parse GGUF (metadata + tensor index)
    if (!gguf.open(ggufPath)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", ggufPath.c_str());
        return false;
    }

    // Extract model config from GGUF metadata
    cfg = extractModelConfig(gguf);

    printf("Model: %s (%u layers, E=%u, HD=%u, V=%u, KV=%u)\n",
           cfg.arch.c_str(), cfg.nLayer, cfg.nEmbd, cfg.headDim,
           cfg.nVocab, cfg.nKvHeads);
    printf("  RoPE theta=%.0f, RMSNorm eps=%.1e, QK-norm=%s\n",
           cfg.ropeTheta, cfg.rmsNormEps,
           cfg.hasQkNorm ? "yes" : "no");

    // Single compute pass for all backends — Dawn handles barriers internally
    passPerDispatch = false;
    printf("  Backend: %s, single-pass dispatch\n",
           gpu->backendType == WGPUBackendType_D3D12 ? "D3D12" : "Vulkan");

    // Read entire GGUF file for tensor data
    FILE* f = fopen(ggufPath.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> fileData(fileSize);
    fread(fileData.data(), 1, fileSize, f);
    fclose(f);

    // Load weights
    loadWeights(gguf, fileData);

    // RoPE tables
    computeRopeTables();

    // Build decode pipeline
    buildDecodePipeline();

    return true;
}

// ─── Load weights ────────────────────────────────────────────────────────────

void ModelRunner::loadWeights(const GGUFFile& gguf,
                               const std::vector<uint8_t>& fileData) {
    auto t0 = std::chrono::steady_clock::now();
    printf("  Loading %llu tensors...\n", (unsigned long long)gguf.n_tensors);

    uint32_t qDim  = cfg.nHead * cfg.headDim;
    uint32_t kvDim = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOut = qDim + 2 * kvDim;

    // Zero bias buffers
    uint32_t maxBias = std::max({cfg.nEmbd, qkvOut,
                                 2 * cfg.intermediateSize, cfg.nVocab});
    std::vector<float> zeros(maxBias, 0.0f);
    zeroBiasE   = gpu->createBuffer("zero_bias_E", cfg.nEmbd * 4);
    zeroBiasQKV = gpu->createBuffer("zero_bias_QKV", qkvOut * 4);
    zeroBiasGU  = gpu->createBuffer("zero_bias_GU", 2 * cfg.intermediateSize * 4);
    zeroBiasV   = gpu->createBuffer("zero_bias_V", cfg.nVocab * 4);
    gpu->writeBuffer(zeroBiasE,   zeros.data(), cfg.nEmbd * 4);
    gpu->writeBuffer(zeroBiasQKV, zeros.data(), qkvOut * 4);
    gpu->writeBuffer(zeroBiasGU,  zeros.data(), 2 * cfg.intermediateSize * 4);
    gpu->writeBuffer(zeroBiasV,   zeros.data(), cfg.nVocab * 4);

    // KV cache (fp16 — halves attention bandwidth)
    kvCache.resize(cfg.nLayer);
    uint64_t kvSize = (uint64_t)maxSeqLen * cfg.nKvHeads * cfg.headDim * 2;  // 2 bytes per f16
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
        kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
        kvCache[i].len = 0;
    }
    printf("  KV cache: %.0f MB (fp16)\n", cfg.nLayer * 2.0 * kvSize / 1048576.0);

    // Helper: load fp32/fp16 norm weight from GGUF tensor
    auto loadNorm = [&](const std::string& ggufName, GPUBuffer& buf) {
        auto it = gguf.tensor_index.find(ggufName);
        if (it == gguf.tensor_index.end()) return;
        auto& ti = gguf.tensors[it->second];
        const uint8_t* data = fileData.data() + gguf.data_offset + ti.offset;
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

    // Per-layer weights
    layerWeights.resize(cfg.nLayer);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto pfx = "blk." + std::to_string(i) + ".";
        auto& lw = layerWeights[i];

        // Fuse Q/K/V into single QKV
        {
            auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
            auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
            auto vi = gguf.tensor_index.find(pfx + "attn_v.weight");
            if (qi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qi->second];
                auto& kt = gguf.tensors[ki->second];
                auto& vt = gguf.tensors[vi->second];
                auto qr = repack_q8_0(fileData.data() + gguf.data_offset + qt.offset, qDim, cfg.nEmbd);
                auto kr = repack_q8_0(fileData.data() + gguf.data_offset + kt.offset, kvDim, cfg.nEmbd);
                auto vr = repack_q8_0(fileData.data() + gguf.data_offset + vt.offset, kvDim, cfg.nEmbd);
                Q8Repacked fused;
                fused.N = qkvOut; fused.K = cfg.nEmbd;
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

        // O projection
        {
            auto it = gguf.tensor_index.find(pfx + "attn_output.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                auto rep = repack_q8_0(fileData.data() + gguf.data_offset + ti.offset,
                                        cfg.nEmbd, qDim);
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".o", rep, lw.oW, lw.oS);
            }
        }

        // Fuse gate + up
        {
            auto gi = gguf.tensor_index.find(pfx + "ffn_gate.weight");
            auto ui = gguf.tensor_index.find(pfx + "ffn_up.weight");
            if (gi != gguf.tensor_index.end() && ui != gguf.tensor_index.end()) {
                auto& gt = gguf.tensors[gi->second];
                auto& ut = gguf.tensors[ui->second];
                auto gr = repack_q8_0(fileData.data() + gguf.data_offset + gt.offset,
                                       cfg.intermediateSize, cfg.nEmbd);
                auto ur = repack_q8_0(fileData.data() + gguf.data_offset + ut.offset,
                                       cfg.intermediateSize, cfg.nEmbd);
                Q8Repacked fused;
                fused.N = 2 * cfg.intermediateSize; fused.K = cfg.nEmbd;
                fused.weights.reserve(gr.weights.size() + ur.weights.size());
                fused.weights.insert(fused.weights.end(), gr.weights.begin(), gr.weights.end());
                fused.weights.insert(fused.weights.end(), ur.weights.begin(), ur.weights.end());
                fused.scales.reserve(gr.scales.size() + ur.scales.size());
                fused.scales.insert(fused.scales.end(), gr.scales.begin(), gr.scales.end());
                fused.scales.insert(fused.scales.end(), ur.scales.begin(), ur.scales.end());
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".gu", fused, lw.guW, lw.guS);
            }
        }

        // Down projection
        {
            auto it = gguf.tensor_index.find(pfx + "ffn_down.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                auto rep = repack_q8_0(fileData.data() + gguf.data_offset + ti.offset,
                                        cfg.nEmbd, cfg.intermediateSize);
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".dn", rep, lw.dnW, lw.dnS);
            }
        }

        // Norm weights
        loadNorm(pfx + "attn_norm.weight", lw.inputNorm);
        loadNorm(pfx + "ffn_norm.weight", lw.postAttnNorm);
        loadNorm(pfx + "attn_q_norm.weight", lw.qNorm);
        loadNorm(pfx + "attn_k_norm.weight", lw.kNorm);

        if (i % 7 == 6 || i == cfg.nLayer - 1)
            printf("  loaded layer %u/%u\n", i + 1, cfg.nLayer);
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
            const uint8_t* data = fileData.data() + gguf.data_offset + ti.offset;
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
            } else {
                memcpy(embeddingCPU.data(), data, nel * 4);
            }
            printf("  Embedding: %u × %u (%s)\n", cfg.nVocab, cfg.nEmbd,
                   ti.type == GGUF_TYPE_Q8_0 ? "Q8_0→f32" :
                   ti.type == GGUF_TYPE_F16 ? "f16→f32" : "f32");

            // LM head: use Q8 format if embedding is Q8_0 (halves bandwidth)
            if (cfg.tieWordEmbeddings) {
                if (ti.type == GGUF_TYPE_Q8_0) {
                    // Keep as Q8 on GPU — no dequant needed
                    auto rep = repack_q8_0(data, cfg.nVocab, cfg.nEmbd);
                    uploadQ8Weight(*gpu, "lm_head_q8", rep,
                                   lmHeadQ8W, lmHeadQ8S);
                    lmHeadIsQ8 = true;
                    uint64_t wBytes = (uint64_t)rep.weights.size() * 4;
                    uint64_t sBytes = (uint64_t)rep.scales.size() * 4;
                    printf("  LM head: tied embeddings (Q8, %llu MB)\n",
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
                    printf("  LM head: tied embeddings (fp16, %llu MB)\n",
                           (unsigned long long)(totalBytes / 1048576));
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("  Weights loaded in %lldms\n", (long long)ms);
}

// ─── RoPE tables ─────────────────────────────────────────────────────────────

void ModelRunner::computeRopeTables() {
    uint32_t half = cfg.headDim / 2;
    std::vector<float> cosTable(maxSeqLen * half), sinTable(maxSeqLen * half);
    for (uint32_t pos = 0; pos < maxSeqLen; pos++) {
        for (uint32_t i = 0; i < half; i++) {
            float freq = 1.0f / powf(cfg.ropeTheta, (float)(2 * i) / cfg.headDim);
            float angle = pos * freq;
            cosTable[pos * half + i] = cosf(angle);
            sinTable[pos * half + i] = sinf(angle);
        }
    }
    ropeCosBuf = gpu->createBuffer("rope_cos", maxSeqLen * half * 4);
    ropeSinBuf = gpu->createBuffer("rope_sin", maxSeqLen * half * 4);
    gpu->writeBuffer(ropeCosBuf, cosTable.data(), maxSeqLen * half * 4);
    gpu->writeBuffer(ropeSinBuf, sinTable.data(), maxSeqLen * half * 4);
}

// ─── Build decode pipeline ───────────────────────────────────────────────────

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

    const bool subgroupMatrixKernelReady =
        gpu->supportsSubgroupMatrix &&
        canUse512ThreadKernels &&
        canCompileEmbeddedKernel(*gpu, "test_subgroup_matrix");
    printf("  Subgroup matrix: %s\n",
           subgroupMatrixKernelReady ? "available (i8×i8→i32 MMA)" : "not available");
    auto& plQ8Fast     = getKernel("q8_matmul_fast");
    auto& plFusedRope  = getKernel("fused_qknorm_rope");
    auto& plChunkP1    = getKernel("gqa_chunked_pass1");
    auto& plChunkP2    = getKernel("gqa_chunked_pass2");
    auto& plFp16Gemm   = getKernel("fp16_gemm");
    auto& plFp16Wide   = getKernel("fp16_gemm_wide");
    auto& plArgmax     = getKernel("argmax");
    auto& plEmbGather  = getKernel("embed_gather");
    auto& plDownSilu   = getKernel("q8_down_silu_add");

    tuning.decodeUseFastQkv = decodeFastQ8Eligible;
    tuning.decodeUseFastGateup = decodeFastQ8Eligible;
    tuning.decodeUseFastOproj = false;
    tuning.decodeUseWideFp16 = decodeWideFp16Eligible;
    decodeFastVariantsAvailable = decodeFastQ8Eligible;

    // Kernel selection per projection:
    auto& plQkv = tuning.decodeUseFastQkv ? plQ8Fast : plQ8Matmul;
    auto& plOp  = tuning.decodeUseFastOproj ? plQ8Fast : plQ8Matmul;
    auto& plGu  = tuning.decodeUseFastGateup ? plQ8Fast : plQ8Matmul;
    auto& plDnSilu = plDownSilu;

    useMMA = (gpu->backendType != WGPUBackendType_D3D12) && subgroupMatrixKernelReady;
    decodePoolCapacity = chooseDecodePoolDepth(*gpu);
    decodePoolDepth = decodePoolCapacity;
    decodeCbPoolBatch = chooseDecodeCbPoolBatch(*gpu, cfg);
        printf("  Initial decode heuristic: qkv=%s oproj=%s gateup=%s lm_head=%s pool=%d batch=%d\n",
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
    auto q8OprojParams = makeQ8Params("p_oproj", qDim, cfg.nEmbd);
    auto q8GuParams    = makeQ8Params("p_gu", cfg.nEmbd, 2 * cfg.intermediateSize);
    // Fused down+silu params: [K=IM, N=E, IM, 0]
    GPUBuffer q8DnSiluParams;
    {
        uint32_t data[4] = {cfg.intermediateSize, cfg.nEmbd, cfg.intermediateSize, 0};
        q8DnSiluParams = gpu->createBuffer("p_dn_silu", 16);
        gpu->writeBuffer(q8DnSiluParams, data, 16);
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
        p[0] = cfg.nHead;  p[1] = qDim;  p[2] = kvDim;
        p[3] = 0;  p[4] = cfg.headDim / 2;  p[5] = 0;
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

    // ─── Single set of intermediate buffers ───────────────────────────────
    // GPU queue executes CBs in submission order — no WAR hazards between
    // consecutive decode steps. Only staging buffers need per-slot copies.
    xBuf          = gpu->createBuffer("x", cfg.nEmbd * 4);
    normOutBuf    = gpu->createBuffer("norm_out", cfg.nEmbd * 4);
    qkvBuf        = gpu->createBuffer("qkv_out", qkvOut * 4);
    qRotBuf       = gpu->createBuffer("q_rot", qDim * 4);
    attnOutBuf    = gpu->createBuffer("attn_out", qDim * 4);
    projOutBuf    = gpu->createBuffer("proj_out", cfg.nEmbd * 4);
    gateUpBuf     = gpu->createBuffer("gate_up", 2 * cfg.intermediateSize * 4);

    rstdBuf       = gpu->createBuffer("rstd", 16);
    logitsBuf     = gpu->createBuffer("logits", cfg.nVocab * 4);
    attnPartialsBuf = gpu->createBuffer("attn_partials",
        cfg.nHead * maxChunks * (cfg.headDim + 2) * 4);

    // Single set of dynamic params (writeBuffer is queue-sequenced)
    fusedRopeParamsBuf = gpu->createBuffer("p_frope", 32);
    gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);
    chunkedAttnParamsBuf = gpu->createBuffer("p_cattn", 32);
    gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);

    // ─── Build dispatch list (single — identical for every token) ─────────
    allDecodeDispatches.reserve(cfg.nLayer * 11 + 2);
    decodeDispatchIndices.assign(cfg.nLayer, {});
    decodeVariantBGs.assign(cfg.nLayer, {});

    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        auto& di = decodeDispatchIndices[i];
        auto& vbg = decodeVariantBGs[i];
        std::string L = "L" + std::to_string(i) + "/";

        // 1. RMSNorm
        if (i == 0) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.inputNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_norm"});
        }

        // 2. QKV matmul
        {
            vbg.qkvBase = makeBG(plQ8Matmul, {
                {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, qkvBuf}, {5, q8QkvParams}});
            if (decodeFastVariantsAvailable) {
                vbg.qkvFast = makeBG(plQ8Fast, {
                    {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                    {3, zeroBiasQKV}, {4, qkvBuf}, {5, q8QkvParams}});
            }
            auto bg = tuning.decodeUseFastQkv && vbg.qkvFast ? vbg.qkvFast : vbg.qkvBase;
            auto pipeline = tuning.decodeUseFastQkv && vbg.qkvFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            di.qkv = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({pipeline, bg,
                1, (qkvOut + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_qkv"});
        }

        {
            auto bg = makeBG(plFusedRope, {
                {0, qkvBuf}, {1, qRotBuf},
                {2, kvCache[i].K}, {3, kvCache[i].V},
                {4, ropeCosBuf}, {5, ropeSinBuf},
                {6, lw.qNorm}, {7, lw.kNorm},
                {8, fusedRopeParamsBuf}});
            allDecodeDispatches.push_back({plFusedRope.pipeline, bg,
                cfg.nHead + cfg.nKvHeads, 1, 1, L+"fused_rope"});
        }

        {
            auto bg = makeBG(plChunkP1, {
                {0, qRotBuf}, {1, kvCache[i].K}, {2, kvCache[i].V},
                {3, attnPartialsBuf}, {4, chunkedAttnParamsBuf}});
            allDecodeDispatches.push_back({plChunkP1.pipeline, bg,
                cfg.nHead, maxChunks, 1, L+"attn_p1"});
        }

        {
            auto bg = makeBG(plChunkP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, chunkedAttnParamsBuf}});
            allDecodeDispatches.push_back({plChunkP2.pipeline, bg,
                cfg.nHead, 1, 1, L+"attn_p2"});
        }

        {
            vbg.oprojBase = makeBG(plQ8Matmul, {
                {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                {3, zeroBiasE}, {4, projOutBuf}, {5, q8OprojParams}});
            if (decodeFastVariantsAvailable) {
                vbg.oprojFast = makeBG(plQ8Fast, {
                    {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                    {3, zeroBiasE}, {4, projOutBuf}, {5, q8OprojParams}});
            }
            auto bg = tuning.decodeUseFastOproj && vbg.oprojFast ? vbg.oprojFast : vbg.oprojBase;
            auto pipeline = tuning.decodeUseFastOproj && vbg.oprojFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            di.oproj = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({pipeline, bg,
                1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_oproj"});
        }

        {
            auto bg = makeBG(plAddRmsNorm, {
                {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                {3, lw.postAttnNorm}, {4, rstdBuf}, {5, rmsParams}});
            allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg, 1, 1, 1, L+"add_rms"});
        }

        {
            vbg.gateupBase = makeBG(plQ8Matmul, {
                {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                {3, zeroBiasGU}, {4, gateUpBuf}, {5, q8GuParams}});
            if (decodeFastVariantsAvailable) {
                vbg.gateupFast = makeBG(plQ8Fast, {
                    {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                    {3, zeroBiasGU}, {4, gateUpBuf}, {5, q8GuParams}});
            }
            auto bg = tuning.decodeUseFastGateup && vbg.gateupFast ? vbg.gateupFast : vbg.gateupBase;
            auto pipeline = tuning.decodeUseFastGateup && vbg.gateupFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            di.gateup = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({pipeline, bg,
                1, (2 * cfg.intermediateSize + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_gateup"});
        }

        // 9. Fused: SiLU·mul + down projection + residual add
        //    Reads gateUpBuf, applies silu(gate)*up on-the-fly, matmul with W_down, adds to xBuf
        {
            auto bg = makeBG(plDnSilu, {
                {0, gateUpBuf}, {1, lw.dnW}, {2, lw.dnS},
                {3, zeroBiasE}, {4, xBuf}, {5, q8DnSiluParams}});
            allDecodeDispatches.push_back({plDnSilu.pipeline, bg,
                1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_down_silu_add"});
        }

        // 10. RMSNorm for next layer
        if (i < cfg.nLayer - 1) {
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
    if (lmHeadIsQ8) {
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
        autoDecodeDispatches.insert(autoDecodeDispatches.end(),
            allDecodeDispatches.begin(), allDecodeDispatches.end());
    }

    printf("  %zu decode dispatches (%u layers)\n",
           allDecodeDispatches.size(), cfg.nLayer);

    // ─── Create staging pool ──────────────────────────────────────────────
    pool.resize(decodePoolCapacity);
    for (int s = 0; s < decodePoolCapacity; s++) {
        WGPUBufferDescriptor bd{};
        bd.usage = BUF_MAP_READ | BUF_COPY_DST;
        bd.size = 4;
        char label[32]; snprintf(label, 32, "staging_%d", s);
        bd.label = {label, (uint32_t)strlen(label)};
        pool[s].stagingBuf = wgpuDeviceCreateBuffer(gpu->device, &bd);
        refillCBPool(s);
    }
    printf("  Pool: %d slots × %d pre-recorded CBs\n",
            decodePoolCapacity, decodeCbPoolBatch);

    // Pre-allocate prefill resources (buffers + bind groups at maxSeqLen)
    initPrefillResources();
}

// ─── Pre-allocate prefill resources ──────────────────────────────────────────

void ModelRunner::initPrefillResources() {
    uint32_t T = maxSeqLen;
    uint32_t qDimL  = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;
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
    auto& kRopeB   = getKernel("rope_batched_simple");

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
        printf("  DP4A (dot4I8Packed): %s\n",
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
    const CompiledPipeline* kSiluQ = usePrequant ? &getKernel("silu_quantize_rows_d3d12") : nullptr;
    const CompiledPipeline* kMatWide = tuning.prefillUseWidePrequant
        ? &getKernel(wideMatKernel)
        : nullptr;
    const CompiledPipeline* kDnSiluWide = tuning.prefillUseWidePrequantAdd
        ? &getKernel(wideAddKernel)
        : nullptr;
    auto& kMat     = getKernel(matKernel);
    auto& kDnSilu  = getKernel(dnSiluKernel);
    auto& kAttn    = getKernel(attnKernel);
    printf("  Prefill tuning: mat=%s qkv/gateup=%s down=%s attn=%s tiles=%ux%u wide=%ux%u pool=%d\n",
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

    // Argmax bind group (reuses existing argmax kernel + result buffer)
    {
        auto argmaxParams = gpu->getBuffer("p_argmax");
        pfCache.argmaxBG = makeBG(getKernel("argmax"), {
            {0, logitsBuf}, {1, argmaxResultBuf}, {2, argmaxParams}});
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

    // Compute group boundaries: split autoDecodeDispatches into nGroups chunks
    int total = (int)autoDecodeDispatches.size();
    std::vector<int> groupStart(nGroups + 1);
    for (int g = 0; g <= nGroups; g++)
        groupStart[g] = g * total / nGroups;

    auto encodeGroup = [&](int gBegin, int gEnd, bool addCopy) -> WGPUCommandBuffer {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

        if (passPerDispatch) {
            for (int d = gBegin; d < gEnd; d++) {
                auto& di = autoDecodeDispatches[d];
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
                auto& di = autoDecodeDispatches[d];
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
    const int maxDepth = std::min(decodePoolCapacity, 4);
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

    printf("  Decode depth autotune: %d -> %d (%.2f ms/tok)\n",
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
    const auto& plQ8Fast = getKernel("q8_matmul_fast");
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& di = decodeDispatchIndices[i];
        auto& vbg = decodeVariantBGs[i];

        if (di.qkv >= 0) {
            auto& d = allDecodeDispatches[di.qkv];
            d.pipeline = tuning.decodeUseFastQkv && vbg.qkvFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastQkv && vbg.qkvFast ? vbg.qkvFast : vbg.qkvBase;
            autoDecodeDispatches[di.qkv + 1] = d;
        }
        if (di.oproj >= 0) {
            auto& d = allDecodeDispatches[di.oproj];
            d.pipeline = tuning.decodeUseFastOproj && vbg.oprojFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastOproj && vbg.oprojFast ? vbg.oprojFast : vbg.oprojBase;
            autoDecodeDispatches[di.oproj + 1] = d;
        }
        if (di.gateup >= 0) {
            auto& d = allDecodeDispatches[di.gateup];
            d.pipeline = tuning.decodeUseFastGateup && vbg.gateupFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastGateup && vbg.gateupFast ? vbg.gateupFast : vbg.gateupBase;
            autoDecodeDispatches[di.gateup + 1] = d;
        }
    }

    for (int s = 0; s < decodePoolDepth; s++)
        refillCBPool(s);
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
    printf("  Decode kernel autotune: qkv=%s oproj=%s gateup=%s (%.2f ms/tok)\n",
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
    printf("  Decode autotune cache: loaded from %s\n", path.c_str());
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
    printf("%s: depth=%d/%d qkv=%s oproj=%s gateup=%s lm_head=%s batch=%d\n",
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
    gpu->writeBuffer(xBuf, emb, cfg.nEmbd * 4);
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
    updateDecodeParams(posOffset, cacheLen);

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

    // Fire-and-forget: submit dispatches, no readback.
    // GPU queue processes in order, so the next step's writeBuffer
    // won't execute until this CB finishes.
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
    // For small T, serial path is faster
    if (T <= 16) {
        if (T == 1) {
            auto logits = decode(tokenIds[0], posOffset);
            return argmax(logits);
        }
        for (uint32_t t = 0; t + 1 < T; t++)
            prefillStep(tokenIds[t], posOffset + t);
        auto logits = prefillFinish(tokenIds[T - 1], posOffset + T - 1);
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
        ropeP[3] = posOffset;  ropeP[4] = cfg.headDim / 2;
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

        // LM head + argmax
        std::vector<Dispatch> lmArgmax;
        if (lmHeadIsQ8) {
            lmArgmax.push_back({getKernel("q8_matmul").pipeline, pfCache.lmBG,
                1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1, "pf_lm"});
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
        printf("    cpu: encode=%.0fus finish=%.0fus submit=%.0fus gpu_wait=%.0fus\n",
               us(t_enc0, t_enc1), us(t_enc1, t_finish),
               us(t_finish, t_submitted), us(t_submitted, t_gpudone));
    }
    auto t_submit = hrc::now();

    // Print prefill timing breakdown
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    printf("  [prefill T=%u] params=%.1fms build=%.1fms gpu+readback=%.1fms total=%.1fms (%zu dispatches)\n",
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
        printf("  Clock calibration: GPU=%llu ns, CPU=%llu ns, deviation=%llu ns\n",
               (unsigned long long)cal.gpuTimestampNs,
               (unsigned long long)cal.cpuTimestampNs,
               (unsigned long long)cal.maxDeviationNs);
    } else {
        printf("  Clock calibration: not available (GPU-only timing)\n");
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
    printf("\n--- GPU Profile (hardware timestamps) ---\n");
    printf("%-20s %10s %6s %10s %6s\n",
           "Kernel", "Total(ms)", "Count", "Avg(us)", "%%");
    printf("%-20s %10s %6s %10s %6s\n",
           "--------------------", "----------", "------", "----------", "------");
    for (auto& [name, e] : sorted_agg) {
        double totalMs = e.totalUs / 1000.0;
        double avgUs = e.totalUs / e.count;
        double pct = totalGpuUs > 0 ? e.totalUs / totalGpuUs * 100.0 : 0;
        printf("%-20s %10.2f %6u %10.1f %5.1f%%\n",
               name.c_str(), totalMs, e.count, avgUs, pct);
    }
    printf("%-20s %10.2f\n", "TOTAL", totalGpuUs / 1000.0);

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
