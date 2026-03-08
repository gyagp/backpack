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

    // Intermediate buffers
    xBuf          = gpu->createBuffer("x", cfg.nEmbd * 4);
    normOutBuf    = gpu->createBuffer("norm_out", cfg.nEmbd * 4);
    qkvBuf        = gpu->createBuffer("qkv_out", qkvOut * 4);
    qRotBuf       = gpu->createBuffer("q_rot", qDim * 4);
    attnOutBuf    = gpu->createBuffer("attn_out", qDim * 4);
    projOutBuf    = gpu->createBuffer("proj_out", cfg.nEmbd * 4);
    gateUpBuf     = gpu->createBuffer("gate_up", 2 * cfg.intermediateSize * 4);
    siluOutBuf    = gpu->createBuffer("silu_out", cfg.intermediateSize * 4);
    rstdBuf       = gpu->createBuffer("rstd", 16);
    logitsBuf     = gpu->createBuffer("logits", cfg.nVocab * 4);

    uint32_t maxChunks = (maxSeqLen + gqaChunkSize - 1) / gqaChunkSize;
    attnPartialsBuf = gpu->createBuffer("attn_partials",
        cfg.nHead * maxChunks * (cfg.headDim + 2) * 4);

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

            // LM head: convert to fp16 for GPU GEMM
            if (cfg.tieWordEmbeddings) {
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
    auto& plFusedRope  = getKernel("fused_qknorm_rope");
    auto& plChunkP1    = getKernel("gqa_chunked_pass1");
    auto& plChunkP2    = getKernel("gqa_chunked_pass2");
    auto& plSiluMul    = getKernel("silu_mul_fused");
    auto& plFp16Gemm   = getKernel("fp16_gemm");
    auto& plFp16Wide   = getKernel("fp16_gemm_wide");

    // Params buffers
    auto makeQ8Params = [&](const std::string& name, uint32_t K, uint32_t N) -> GPUBuffer {
        uint32_t data[4] = {K, N, 0, 0};
        auto buf = gpu->createBuffer(name, 16);
        gpu->writeBuffer(buf, data, 16);
        return buf;
    };

    auto q8QkvParams   = makeQ8Params("p_qkv", cfg.nEmbd, qkvOut);
    auto q8OprojParams = makeQ8Params("p_oproj", qDim, cfg.nEmbd);
    auto q8GuParams    = makeQ8Params("p_gu", cfg.nEmbd, 2 * cfg.intermediateSize);
    auto q8DnParams    = makeQ8Params("p_dn", cfg.intermediateSize, cfg.nEmbd);

    GPUBuffer rmsParams;
    {
        uint32_t rn[4];
        rn[0] = cfg.nEmbd; rn[1] = cfg.nEmbd;
        float eps = cfg.rmsNormEps; memcpy(&rn[2], &eps, 4);
        rn[3] = 0;
        rmsParams = gpu->createBuffer("p_rms", 16);
        gpu->writeBuffer(rmsParams, rn, 16);
    }

    GPUBuffer siluParams;
    {
        uint32_t sm[4] = {cfg.intermediateSize, 0, 0, 0};
        siluParams = gpu->createBuffer("p_silu", 16);
        gpu->writeBuffer(siluParams, sm, 16);
    }

    GPUBuffer lmheadParams;
    {
        uint32_t fp[4] = {cfg.nEmbd, cfg.nVocab, 0, 0};
        lmheadParams = gpu->createBuffer("p_lmhead", 16);
        gpu->writeBuffer(lmheadParams, fp, 16);
    }

    // Fused RoPE params
    {
        ropeParamData.resize(32, 0);
        auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
        p[0] = cfg.nHead;
        p[1] = qDim;
        p[2] = kvDim;
        p[3] = 0;               // pos (dynamic)
        p[4] = cfg.headDim / 2;
        p[5] = 0;               // cache_offset (dynamic)
        float eps = cfg.rmsNormEps;
        memcpy(&p[6], &eps, 4);
        fusedRopeParamsBuf = gpu->createBuffer("p_frope", 32);
        gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);
    }

    // Chunked attention params
    {
        chunkedAttnParamData.resize(32, 0);
        auto* p = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
        p[0] = cfg.nKvHeads * cfg.headDim;
        p[1] = cfg.nHead / cfg.nKvHeads;
        p[2] = 0;               // T_total (dynamic)
        p[3] = gqaChunkSize;
        p[4] = 0;               // n_chunks (dynamic)
        float scale = 1.0f / sqrtf((float)cfg.headDim);
        float neg_inf = -1e9f;
        memcpy(&p[5], &scale, 4);
        memcpy(&p[6], &neg_inf, 4);
        p[7] = maxChunks;
        chunkedAttnParamsBuf = gpu->createBuffer("p_cattn", 32);
        gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);
    }

    // Build dispatch sequence
    allDecodeDispatches.reserve(cfg.nLayer * 11 + 2);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];

        if (!lw.qkvW.handle || !lw.oW.handle || !lw.guW.handle || !lw.dnW.handle) {
            fprintf(stderr, "ERROR: layer %u missing weight buffers\n", i);
            exit(1);
        }

        // Models without QK norm need all-ones weight buffers (identity norm)
        if (!lw.qNorm.handle) {
            std::vector<float> ones(cfg.headDim, 1.0f);
            lw.qNorm = gpu->createBuffer("qnorm_id_" + std::to_string(i),
                                          cfg.headDim * 4);
            gpu->writeBuffer(lw.qNorm, ones.data(), cfg.headDim * 4);
        }
        if (!lw.kNorm.handle) {
            std::vector<float> ones(cfg.headDim, 1.0f);
            lw.kNorm = gpu->createBuffer("knorm_id_" + std::to_string(i),
                                          cfg.headDim * 4);
            gpu->writeBuffer(lw.kNorm, ones.data(), cfg.headDim * 4);
        }

        std::string L = "L" + std::to_string(i) + "/";

        // 1. RMSNorm (first layer uses input norm)
        if (i == 0) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.inputNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_norm"});
        }

        // 2. QKV matmul
        {
            auto bg = makeBG(plQ8Matmul, {
                {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, qkvBuf}, {5, q8QkvParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (qkvOut + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_qkv"});
        }

        // 3. Fused QKnorm + RoPE + KV scatter
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

        // 4. Chunked attention pass 1
        {
            auto bg = makeBG(plChunkP1, {
                {0, qRotBuf}, {1, kvCache[i].K}, {2, kvCache[i].V},
                {3, attnPartialsBuf}, {4, chunkedAttnParamsBuf}});
            allDecodeDispatches.push_back({plChunkP1.pipeline, bg,
                cfg.nHead, maxChunks, 1, L+"attn_p1"});
        }

        // 5. Chunked attention pass 2
        {
            auto bg = makeBG(plChunkP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, chunkedAttnParamsBuf}});
            allDecodeDispatches.push_back({plChunkP2.pipeline, bg,
                cfg.nHead, 1, 1, L+"attn_p2"});
        }

        // 6. O projection
        {
            auto bg = makeBG(plQ8Matmul, {
                {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                {3, zeroBiasE}, {4, projOutBuf}, {5, q8OprojParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_oproj"});
        }

        // 7. Add + RMSNorm
        {
            auto bg = makeBG(plAddRmsNorm, {
                {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                {3, lw.postAttnNorm}, {4, rstdBuf}, {5, rmsParams}});
            allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg, 1, 1, 1, L+"add_rms"});
        }

        // 8. Gate+Up matmul
        {
            auto bg = makeBG(plQ8Matmul, {
                {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                {3, zeroBiasGU}, {4, gateUpBuf}, {5, q8GuParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (2 * cfg.intermediateSize + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_gateup"});
        }

        // 9. SiLU x mul
        {
            auto bg = makeBG(plSiluMul, {
                {0, gateUpBuf}, {1, siluOutBuf}, {2, siluParams}});
            allDecodeDispatches.push_back({plSiluMul.pipeline, bg,
                (cfg.intermediateSize + 127) / 128, 1, 1, L+"silu_mul"});
        }

        // 10. Down projection + residual add
        {
            auto bg = makeBG(plQ8MatAdd, {
                {0, siluOutBuf}, {1, lw.dnW}, {2, lw.dnS},
                {3, zeroBiasE}, {4, xBuf}, {5, q8DnParams}});
            allDecodeDispatches.push_back({plQ8MatAdd.pipeline, bg,
                1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_down_add"});
        }

        // 11. RMSNorm for next layer
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

    // LM head (fp16 GEMM)
    {
        uint32_t FP16_TILE = 8;
        auto bg = makeBG(plFp16Gemm, {
            {0, normOutBuf}, {1, lmHeadW}, {2, zeroBiasV},
            {3, logitsBuf}, {4, lmheadParams}});
        allDecodeDispatches.push_back({plFp16Gemm.pipeline, bg,
            1, (cfg.nVocab + FP16_TILE - 1) / FP16_TILE, 1, "lm_head"});
    }

    // GPU argmax on logits
    {
        auto& plArgmax = getKernel("argmax");
        argmaxResultBuf = gpu->createBuffer("argmax_result", 4);

        // Double-buffered MAP_READ staging buffers for async readback
        for (int i = 0; i < 2; i++) {
            WGPUBufferDescriptor bd{};
            bd.usage = BUF_MAP_READ | BUF_COPY_DST;
            bd.size = 4;
            char label[32];
            snprintf(label, sizeof(label), "staging_%d", i);
            bd.label = {label, (uint32_t)strlen(label)};
            stagingBufs[i] = wgpuDeviceCreateBuffer(gpu->device, &bd);
        }
        GPUBuffer argmaxParams;
        {
            uint32_t p[4] = {cfg.nVocab, 0, 0, 0};
            argmaxParams = gpu->createBuffer("p_argmax", 16);
            gpu->writeBuffer(argmaxParams, p, 16);
        }
        auto bg = makeBG(plArgmax, {
            {0, logitsBuf}, {1, argmaxResultBuf}, {2, argmaxParams}});
        allDecodeDispatches.push_back({plArgmax.pipeline, bg, 1, 1, 1, "argmax"});
    }

    // GPU embedding gather (reads argmax result, writes embedding to xBuf)
    // This enables zero-readback autoregressive decode.
    {
        auto& plEmbGather = getKernel("embed_gather");

        // Upload full embedding table to GPU
        uint64_t embBytes = (uint64_t)embeddingCPU.size() * 4;
        embeddingGpuBuf = gpu->createBuffer("embedding_gpu", embBytes);
        const uint64_t CHUNK = 128 * 1024 * 1024;
        for (uint64_t off = 0; off < embBytes; off += CHUNK) {
            uint64_t sz = std::min(CHUNK, embBytes - off);
            wgpuQueueWriteBuffer(gpu->queue, embeddingGpuBuf.handle, off,
                                 (const uint8_t*)embeddingCPU.data() + off, sz);
        }

        GPUBuffer embedParams;
        {
            uint32_t p[4] = {cfg.nEmbd, 0, 0, 0};
            embedParams = gpu->createBuffer("p_embed", 16);
            gpu->writeBuffer(embedParams, p, 16);
        }
        auto bg = makeBG(plEmbGather, {
            {0, embeddingGpuBuf}, {1, argmaxResultBuf},
            {2, xBuf}, {3, embedParams}});

        // autoDecodeDispatches = embed_gather + all decode dispatches
        autoDecodeDispatches.clear();
        autoDecodeDispatches.push_back({plEmbGather.pipeline, bg,
            (cfg.nEmbd + 255) / 256, 1, 1, "embed_gather"});
        autoDecodeDispatches.insert(autoDecodeDispatches.end(),
            allDecodeDispatches.begin(), allDecodeDispatches.end());
    }

    printf("  Pre-recorded %zu decode dispatches (%u layers)\n",
           allDecodeDispatches.size(), cfg.nLayer);
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

    // Submit all dispatches (including GPU argmax) and read back just 4 bytes
    auto result = gpu->submitAndReadback(
        allDecodeDispatches, argmaxResultBuf, 4, passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    int32_t token;
    memcpy(&token, result.data(), 4);
    return token;
}

void ModelRunner::decodeAutoregressive(uint32_t posOffset) {
    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    // Submit dispatches + copy argmax to staging buffer (ONE command buffer)
    // Double-buffered: alternate between staging buffers 0 and 1
    gpu->submitAndCopyAsync(autoDecodeDispatches,
                             argmaxResultBuf, 4,
                             stagingBufs[stagingIdx]);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;
}

int32_t ModelRunner::readLastArgmax() {
    // Wait for the staging buffer map to complete and read the result
    int32_t token = gpu->completeAsyncMapI32(stagingBufs[stagingIdx]);
    // Flip to the other staging buffer for next token
    stagingIdx ^= 1;
    return token;
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
