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

    // KV cache
    kvCache.resize(cfg.nLayer);
    uint64_t kvSize = (uint64_t)maxSeqLen * cfg.nKvHeads * cfg.headDim * 4;
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
        kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
        kvCache[i].len = 0;
    }
    printf("  KV cache: %.0f MB\n", cfg.nLayer * 2.0 * kvSize / 1048576.0);

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
    uint32_t Q8_TILE = 8;
    uint32_t maxChunks = (maxSeqLen + gqaChunkSize - 1) / gqaChunkSize;

    // Load kernels from embedded shaders
    auto& plRmsNorm    = getKernel("rms_norm");
    auto& plAddRmsNorm = getKernel("add_rms_norm");
    auto& plQ8Matmul   = getKernel("q8_matmul");
    auto& plQ8MatAdd   = getKernel("q8_matmul_add");
    auto& plQ8Fast     = getKernel("q8_matmul_fast");
    auto& plQ8AddFast  = getKernel("q8_matmul_add_fast");
    auto& plFusedRope  = getKernel("fused_qknorm_rope");
    auto& plChunkP1    = getKernel("gqa_chunked_pass1");
    auto& plChunkP2    = getKernel("gqa_chunked_pass2");
    auto& plSiluMul    = getKernel("silu_mul_fused");
    auto& plFp16Gemm   = getKernel("fp16_gemm");
    auto& plFp16Wide   = getKernel("fp16_gemm_wide");
    auto& plArgmax     = getKernel("argmax");
    auto& plEmbGather  = getKernel("embed_gather");

    // Kernel selection per projection:
    // dp4a: quantize activations to int8 on-the-fly, use dot4I8Packed
    // Fused: q8_down_silu_add reads gateUpBuf, applies silu·mul, matmul, residual add
    auto& plDp4a = getKernel("q8_matmul_dp4a");
    auto& plQkv = plDp4a;                            // K=2048, N=4096
    auto& plOp  = plDp4a;                            // K=2048, N=2048
    auto& plGu  = plDp4a;                            // K=2048, N=12288
    auto& plDnSilu = getKernel("q8_down_silu_add");  // fused: silu_mul + down_add

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

    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
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
            auto bg = makeBG(plQkv, {
                {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, qkvBuf}, {5, q8QkvParams}});
            allDecodeDispatches.push_back({plQkv.pipeline, bg,
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
            auto bg = makeBG(plOp, {
                {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                {3, zeroBiasE}, {4, projOutBuf}, {5, q8OprojParams}});
            allDecodeDispatches.push_back({plOp.pipeline, bg,
                1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_oproj"});
        }

        {
            auto bg = makeBG(plAddRmsNorm, {
                {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                {3, lw.postAttnNorm}, {4, rstdBuf}, {5, rmsParams}});
            allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg, 1, 1, 1, L+"add_rms"});
        }

        {
            auto bg = makeBG(plGu, {
                {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                {3, zeroBiasGU}, {4, gateUpBuf}, {5, q8GuParams}});
            allDecodeDispatches.push_back({plGu.pipeline, bg,
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
        uint32_t FP16_TILE = 8;
        auto bg = makeBG(plFp16Gemm, {
            {0, normOutBuf}, {1, lmHeadW}, {2, zeroBiasV},
            {3, logitsBuf}, {4, lmheadParams}});
        allDecodeDispatches.push_back({plFp16Gemm.pipeline, bg,
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
    for (int s = 0; s < POOL_DEPTH; s++) {
        WGPUBufferDescriptor bd{};
        bd.usage = BUF_MAP_READ | BUF_COPY_DST;
        bd.size = 4;
        char label[32]; snprintf(label, 32, "staging_%d", s);
        bd.label = {label, (uint32_t)strlen(label)};
        pool[s].stagingBuf = wgpuDeviceCreateBuffer(gpu->device, &bd);
        refillCBPool(s);
    }
    printf("  Pool: %d slots × %d pre-recorded CBs\n", POOL_DEPTH, CB_POOL_BATCH);
}

// ─── Pre-record command buffer pool ──────────────────────────────────────────

void ModelRunner::refillCBPool(int slot) {
    auto& ps = pool[slot];

    // Release any remaining CBs
    for (int i = ps.cbIdx; i < (int)ps.cbPool.size(); i++)
        wgpuCommandBufferRelease(ps.cbPool[i]);

    // Each "token" needs nGroups CBs. Pre-record CB_POOL_BATCH tokens worth.
    int totalCBs = CB_POOL_BATCH * nGroups;
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

    for (int i = 0; i < CB_POOL_BATCH; i++) {
        for (int g = 0; g < nGroups; g++) {
            bool isLast = (g == nGroups - 1);
            ps.cbPool[i * nGroups + g] = encodeGroup(
                groupStart[g], groupStart[g + 1], isLast);
        }
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

std::vector<float> ModelRunner::prefillBatched(
        const int32_t* tokenIds, uint32_t T, uint32_t posOffset) {
    // Batched prefill: matmuls process all T tokens in parallel (one weight read).
    // RoPE and KV scatter are done per-token (reusing the decode kernel).
    // Attention uses the batched causal kernel.

    uint32_t qDimL = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;
    uint32_t Q8_TILE = 8, TILE_M = 8;

    // ── Allocate T-sized buffers ─────────────────────────────
    auto pXBuf       = gpu->createBuffer("pf_x", T * cfg.nEmbd * 4);
    auto pNormBuf    = gpu->createBuffer("pf_norm", T * cfg.nEmbd * 4);
    auto pQkvBuf     = gpu->createBuffer("pf_qkv", T * qkvOutL * 4);
    auto pQRotBuf    = gpu->createBuffer("pf_qrot", T * qDimL * 4);
    auto pAttnOutBuf = gpu->createBuffer("pf_attn", T * qDimL * 4);
    auto pProjBuf    = gpu->createBuffer("pf_proj", T * cfg.nEmbd * 4);
    auto pGateUpBuf  = gpu->createBuffer("pf_gu", T * 2 * cfg.intermediateSize * 4);
    auto pRstdBuf    = gpu->createBuffer("pf_rstd", T * 4);

    // Upload all T embeddings at once
    std::vector<float> embData(T * cfg.nEmbd);
    for (uint32_t t = 0; t < T; t++) {
        int32_t tid_tok = (tokenIds[t] >= 0 && (uint32_t)tokenIds[t] < cfg.nVocab)
                          ? tokenIds[t] : 0;
        memcpy(embData.data() + t * cfg.nEmbd,
               embeddingCPU.data() + tid_tok * cfg.nEmbd,
               cfg.nEmbd * 4);
    }
    gpu->writeBuffer(pXBuf, embData.data(), T * cfg.nEmbd * 4);

    // Get batched kernels
    auto& plRmsB     = getKernel("rms_norm_batched");
    auto& plAddRmsB  = getKernel("add_rms_norm_batched");
    auto& plQ8B      = getKernel("q8_matmul_batched");
    auto& plDnSiluB  = getKernel("q8_down_silu_add_batched");
    auto& plCausalA  = getKernel("causal_attn");
    auto& plFusedRope = getKernel("fused_qknorm_rope");

    // Create batched params
    auto makeP3 = [&](const std::string& n, uint32_t a, uint32_t b, uint32_t c) -> GPUBuffer {
        uint32_t d[4] = {a, b, c, 0};
        auto buf = gpu->createBuffer(n, 16);
        gpu->writeBuffer(buf, d, 16);
        return buf;
    };
    auto makeP4 = [&](const std::string& n, uint32_t a, uint32_t b, uint32_t c, uint32_t d_) -> GPUBuffer {
        uint32_t d[4] = {a, b, c, d_};
        auto buf = gpu->createBuffer(n, 16);
        gpu->writeBuffer(buf, d, 16);
        return buf;
    };

    auto pQkvP = makeP3("pp_qkv", cfg.nEmbd, qkvOutL, T);
    auto pOpP  = makeP3("pp_oproj", qDimL, cfg.nEmbd, T);
    auto pGuP  = makeP3("pp_gu", cfg.nEmbd, 2 * cfg.intermediateSize, T);
    auto pDnP  = makeP4("pp_dn", cfg.intermediateSize, cfg.nEmbd, cfg.intermediateSize, T);

    GPUBuffer pRmsP;
    {
        uint32_t d[4]; d[0] = cfg.nEmbd; d[1] = cfg.nEmbd;
        float eps = cfg.rmsNormEps; memcpy(&d[2], &eps, 4); d[3] = 0;
        pRmsP = gpu->createBuffer("pp_rms", 16);
        gpu->writeBuffer(pRmsP, d, 16);
    }

    // Per-token RoPE params buffer (rewritten per token)
    GPUBuffer pRopeP = gpu->createBuffer("pp_rope", 32);

    uint32_t cacheLen = kvCache[0].len;

    // ── Per-layer processing ─────────────────────────────────
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        std::vector<Dispatch> ld;  // layer dispatches

        // 1. Batched RMSNorm (T rows)
        {
            auto bg = makeBG(plRmsB, {
                {0, pXBuf}, {1, pNormBuf}, {2, lw.inputNorm},
                {3, pRstdBuf}, {4, pRmsP}});
            ld.push_back({plRmsB.pipeline, bg, T, 1, 1, "pf_rms"});
        }

        // 2. Batched QKV matmul (reads weights ONCE for all T tokens)
        {
            auto bg = makeBG(plQ8B, {
                {0, pNormBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, pQkvBuf}, {5, pQkvP}});
            ld.push_back({plQ8B.pipeline, bg,
                (T + TILE_M - 1) / TILE_M,
                (qkvOutL + Q8_TILE - 1) / Q8_TILE, 1, "pf_qkv"});
        }

        // Submit matmul dispatches
        gpu->submitOnly(ld, true);

        // 3. Per-token RoPE + KV scatter (T sequential dispatches)
        // We need per-token position, so submit T individual rope dispatches.
        for (uint32_t t = 0; t < T; t++) {
            uint32_t pos = posOffset + t;
            uint32_t cl = cacheLen + t;  // KV cache length at this point

            // Set up rope params for this token
            auto* rp = reinterpret_cast<int32_t*>(ropeParamData.data());
            rp[0] = cfg.nHead; rp[1] = qDimL; rp[2] = kvDimL;
            rp[3] = pos;
            rp[4] = cfg.headDim / 2;
            rp[5] = cl * cfg.nKvHeads * cfg.headDim;
            float eps = cfg.rmsNormEps; memcpy(&rp[6], &eps, 4);
            gpu->writeBuffer(pRopeP, ropeParamData.data(), 32);

            // Compute offsets into the T-batched QKV and QRot buffers
            // The fused_rope kernel reads from binding[0] (qkvBuf) at offset 0
            // We need to point it at row t of pQkvBuf → use a view/offset
            // But WebGPU bind groups don't support offsets...
            // Solution: create per-token bind groups pointing to sub-regions
            // Actually, we need separate buffers or a kernel that can handle rows.
            // For simplicity: copy row t of pQkvBuf to the T=1 qkvBuf, run rope,
            // copy back. This is expensive but correct for a first version.

            // Copy row t of pQkvBuf → qkvBuf (T=1 buffer)
            {
                WGPUCommandEncoderDescriptor enD{};
                auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
                wgpuCommandEncoderCopyBufferToBuffer(enc,
                    pQkvBuf.handle, t * qkvOutL * 4,
                    qkvBuf.handle, 0, qkvOutL * 4);
                WGPUCommandBufferDescriptor cbD{};
                auto cb = wgpuCommandEncoderFinish(enc, &cbD);
                wgpuQueueSubmit(gpu->queue, 1, &cb);
                wgpuCommandEncoderRelease(enc);
                wgpuCommandBufferRelease(cb);
            }

            // Run T=1 fused RoPE using decode buffers
            auto bg = makeBG(plFusedRope, {
                {0, qkvBuf}, {1, qRotBuf},
                {2, kvCache[i].K}, {3, kvCache[i].V},
                {4, ropeCosBuf}, {5, ropeSinBuf},
                {6, lw.qNorm}, {7, lw.kNorm},
                {8, pRopeP}});
            std::vector<Dispatch> rd;
            rd.push_back({plFusedRope.pipeline, bg,
                cfg.nHead + cfg.nKvHeads, 1, 1, "pf_rope"});
            gpu->submitOnly(rd, true);

            // Copy qRotBuf (T=1) → row t of pQRotBuf
            {
                WGPUCommandEncoderDescriptor enD{};
                auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
                wgpuCommandEncoderCopyBufferToBuffer(enc,
                    qRotBuf.handle, 0,
                    pQRotBuf.handle, t * qDimL * 4, qDimL * 4);
                WGPUCommandBufferDescriptor cbD{};
                auto cb = wgpuCommandEncoderFinish(enc, &cbD);
                wgpuQueueSubmit(gpu->queue, 1, &cb);
                wgpuCommandEncoderRelease(enc);
                wgpuCommandBufferRelease(cb);
            }
        }

        // 4. Batched causal attention
        {
            uint32_t T_total = cacheLen + T;
            uint32_t ap[8] = {};
            ap[0] = cfg.nKvHeads * cfg.headDim;   // kv_stride
            ap[1] = cfg.nHead / cfg.nKvHeads;     // n_rep
            ap[2] = T_total;                       // total KV length
            ap[3] = cacheLen;                      // cache_offset (queries start here)
            float sc = 1.0f / sqrtf((float)cfg.headDim);
            float ni = -1e9f;
            memcpy(&ap[5], &sc, 4);
            memcpy(&ap[6], &ni, 4);
            auto pAP = gpu->createBuffer("pp_attn_L" + std::to_string(i), 32);
            gpu->writeBuffer(pAP, ap, 32);

            auto bg = makeBG(plCausalA, {
                {0, pQRotBuf}, {1, kvCache[i].K}, {2, kvCache[i].V},
                {3, pAttnOutBuf}, {4, pAP}});
            std::vector<Dispatch> ad;
            ad.push_back({plCausalA.pipeline, bg, cfg.nHead, T, 1, "pf_attn"});
            gpu->submitOnly(ad, true);
        }

        // 5-8. Batched oproj → add_rms → gateup → down_silu_add
        {
            std::vector<Dispatch> md;

            // O projection
            auto bg_op = makeBG(plQ8B, {
                {0, pAttnOutBuf}, {1, lw.oW}, {2, lw.oS},
                {3, zeroBiasE}, {4, pProjBuf}, {5, pOpP}});
            md.push_back({plQ8B.pipeline, bg_op,
                (T + TILE_M - 1) / TILE_M,
                (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, "pf_oproj"});

            // Add + RMSNorm
            auto bg_arm = makeBG(plAddRmsB, {
                {0, pXBuf}, {1, pProjBuf}, {2, pNormBuf},
                {3, lw.postAttnNorm}, {4, pRstdBuf}, {5, pRmsP}});
            md.push_back({plAddRmsB.pipeline, bg_arm, T, 1, 1, "pf_add_rms"});

            // Gate+Up
            auto bg_gu = makeBG(plQ8B, {
                {0, pNormBuf}, {1, lw.guW}, {2, lw.guS},
                {3, zeroBiasGU}, {4, pGateUpBuf}, {5, pGuP}});
            md.push_back({plQ8B.pipeline, bg_gu,
                (T + TILE_M - 1) / TILE_M,
                (2 * cfg.intermediateSize + Q8_TILE - 1) / Q8_TILE, 1, "pf_gateup"});

            // Fused SiLU + down + residual add
            auto bg_dn = makeBG(plDnSiluB, {
                {0, pGateUpBuf}, {1, lw.dnW}, {2, lw.dnS},
                {3, zeroBiasE}, {4, pXBuf}, {5, pDnP}});
            md.push_back({plDnSiluB.pipeline, bg_dn,
                (T + TILE_M - 1) / TILE_M,
                (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, "pf_down_silu"});

            gpu->submitOnly(md, true);
        }
    }

    // Final RMSNorm (batched)
    {
        auto bg = makeBG(plRmsB, {
            {0, pXBuf}, {1, pNormBuf}, {2, finalNormW},
            {3, pRstdBuf}, {4, pRmsP}});
        std::vector<Dispatch> fd;
        fd.push_back({plRmsB.pipeline, bg, T, 1, 1, "pf_final_rms"});
        gpu->submitOnly(fd, true);
    }

    // LM head — only compute for the last token (T=1)
    // Copy last row of pNormBuf → normOutBuf (T=1)
    {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pNormBuf.handle, (T - 1) * cfg.nEmbd * 4,
            normOutBuf.handle, 0, cfg.nEmbd * 4);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);
    }

    // Use existing T=1 LM head + argmax
    std::vector<Dispatch> lm;
    if (lmHeadIsQ8) {
        auto lmP = makeP3("pp_lm", cfg.nEmbd, cfg.nVocab, 1);
        auto bg = makeBG(getKernel("q8_matmul"), {
            {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
            {3, zeroBiasV}, {4, logitsBuf}, {5, lmP}});
        lm.push_back({getKernel("q8_matmul").pipeline, bg,
            1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1, "pf_lm"});
    }
    auto result = gpu->submitAndReadback(lm, logitsBuf, cfg.nVocab * 4, passPerDispatch);

    // Update KV cache lengths
    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len += T;

    // Copy last hidden state to xBuf for subsequent decode
    {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pXBuf.handle, (T - 1) * cfg.nEmbd * 4,
            xBuf.handle, 0, cfg.nEmbd * 4);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);
    }

    std::vector<float> logits(cfg.nVocab);
    memcpy(logits.data(), result.data(), cfg.nVocab * 4);
    return logits;
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
