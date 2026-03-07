#include "model_runner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    return {std::istreambuf_iterator<char>(f), {}};
}

WGPUBindGroup ModelRunner::makeBG(
        const CompiledPipeline& pl,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings) {
    fprintf(stderr, "  makeBG: %zu bindings, layout=%p\n",
            bindings.size(), (void*)pl.bgLayout); fflush(stderr);
    WGPUBindGroupEntry entries[16];
    for (size_t i = 0; i < bindings.size() && i < 16; i++) {
        memset(&entries[i], 0, sizeof(WGPUBindGroupEntry));
        entries[i].binding = bindings[i].first;
        entries[i].buffer  = bindings[i].second.handle;
        entries[i].size    = bindings[i].second.size;
        if (!entries[i].buffer) {
            fprintf(stderr, "  ERROR: binding %u has null buffer!\n", entries[i].binding);
        }
    }
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = pl.bgLayout;
    d.entryCount = (uint32_t)bindings.size();
    d.entries = entries;
    auto result = wgpuDeviceCreateBindGroup(gpu->device, &d);
    fprintf(stderr, "  makeBG result: %p\n", (void*)result); fflush(stderr);
    return result;
}

// ─── Load kernel from bundle ─────────────────────────────────────────────────

const CompiledPipeline& ModelRunner::loadKernel(const std::string& name) {
    auto it = kernelCache_.find(name);
    if (it != kernelCache_.end()) return *it->second;

    // Find kernel in manifest
    const auto& kernels = manifest["kernels"].as_array();
    for (auto& k : kernels) {
        if (k["name"].as_string() == name) {
            auto wgsl = read_file(bundleDir + "/" + k["file"].as_string());
            uint32_t nBindings = (uint32_t)k["bindings"].size();
            // Add params buffer binding if kernel has params
            if (k.has("params") && k["params"].size() > 0)
                nBindings++;
            auto& pl = gpu->getOrCreatePipeline(name, wgsl, nBindings);
            kernelCache_[name] = &pl;
            return pl;
        }
    }
    fprintf(stderr, "Kernel not found in manifest: %s\n", name.c_str());
    exit(1);
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

bool ModelRunner::load(GPUContext& ctx, const std::string& dir,
                       const std::string& ggufPath) {
    gpu = &ctx;
    bundleDir = dir;

    // Parse manifest
    auto manifestStr = read_file(dir + "/manifest.json");
    manifest = json_parse(manifestStr);

    auto& model = manifest["model"];
    nLayer          = model["n_layer"].as_uint();
    nHead           = model["n_head"].as_uint();
    nKvHeads        = model["n_kv_heads"].as_uint();
    nEmbd           = model["n_embd"].as_uint();
    intermediateSize= model["intermediate_size"].as_uint();
    nVocab          = model["n_vocab"].as_uint();
    headDim         = model["head_dim"].as_uint();
    rmsNormEps      = (float)model["rms_norm_eps"].as_number();
    ropeTheta       = (float)model["rope_theta"].as_number();
    tieWordEmbeddings = model.has("tie_word_embeddings") ?
                        model["tie_word_embeddings"].as_bool() : true;

    printf("Model: %s (%u layers, E=%u, HD=%u, V=%u)\n",
           model.has("model_type") ? model["model_type"].as_string().c_str() : "unknown",
           nLayer, nEmbd, headDim, nVocab);

    // Load weights from GGUF
    loadWeights(ggufPath);

    // Compute RoPE tables (needed before decode pipeline)
    computeRopeTables();

    // Build decode pipeline
    buildDecodePipeline();

    return true;
}

// ─── Load weights ────────────────────────────────────────────────────────────

void ModelRunner::loadWeights(const std::string& ggufPath) {
    printf("Loading GGUF: %s\n", ggufPath.c_str());
    GGUFFile gguf;
    if (!gguf.open(ggufPath)) {
        fprintf(stderr, "Failed to open GGUF\n"); exit(1);
    }
    printf("  %llu tensors\n", (unsigned long long)gguf.n_tensors);

    // Memory-map the GGUF file for tensor data
    FILE* f = fopen(ggufPath.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> fileData(fileSize);
    fread(fileData.data(), 1, fileSize, f);
    fclose(f);

    uint32_t qDim  = nHead * headDim;
    uint32_t kvDim  = nKvHeads * headDim;
    uint32_t qkvOut = qDim + 2 * kvDim;

    // Zero bias buffers
    std::vector<float> zeros;
    zeros.resize(std::max({nEmbd, qkvOut, 2 * intermediateSize, nVocab}), 0.0f);
    zeroBiasE   = gpu->createBuffer("zero_bias_E", nEmbd * 4);
    zeroBiasQKV = gpu->createBuffer("zero_bias_QKV", qkvOut * 4);
    zeroBiasGU  = gpu->createBuffer("zero_bias_GU", 2 * intermediateSize * 4);
    zeroBiasV   = gpu->createBuffer("zero_bias_V", nVocab * 4);
    gpu->writeBuffer(zeroBiasE, zeros.data(), nEmbd * 4);
    gpu->writeBuffer(zeroBiasQKV, zeros.data(), qkvOut * 4);
    gpu->writeBuffer(zeroBiasGU, zeros.data(), 2 * intermediateSize * 4);
    gpu->writeBuffer(zeroBiasV, zeros.data(), nVocab * 4);

    // Intermediate buffers
    xBuf          = gpu->createBuffer("x", nEmbd * 4);
    normOutBuf    = gpu->createBuffer("norm_out", nEmbd * 4);
    qkvBuf        = gpu->createBuffer("qkv_out", qkvOut * 4);
    qRotBuf       = gpu->createBuffer("q_rot", qDim * 4);
    attnOutBuf    = gpu->createBuffer("attn_out", qDim * 4);
    projOutBuf    = gpu->createBuffer("proj_out", nEmbd * 4);
    gateUpBuf     = gpu->createBuffer("gate_up", 2 * intermediateSize * 4);
    siluOutBuf    = gpu->createBuffer("silu_out", intermediateSize * 4);
    rstdBuf       = gpu->createBuffer("rstd", 16);
    logitsBuf     = gpu->createBuffer("logits", nVocab * 4);

    uint32_t GQA_CHUNK = 64;
    uint32_t maxSeq = 2048;
    uint32_t maxChunks = (maxSeq + GQA_CHUNK - 1) / GQA_CHUNK;
    attnPartialsBuf = gpu->createBuffer("attn_partials",
        nHead * maxChunks * (headDim + 2) * 4);

    // KV cache
    kvCache.resize(nLayer);
    uint64_t kvSize = maxSeq * nKvHeads * headDim * 4;
    for (uint32_t i = 0; i < nLayer; i++) {
        kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
        kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
        kvCache[i].len = 0;
    }
    printf("  KV cache: %.0f MB\n", nLayer * 2.0 * kvSize / 1048576.0);

    // Load per-layer weights
    layerWeights.resize(nLayer);
    auto t0 = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < nLayer; i++) {
        auto pfx = "blk." + std::to_string(i) + ".";
        auto& lw = layerWeights[i];

        // Q8_0 weight tensors: GGUF stores as (K, N) with Q8_0 blocks
        auto loadQ8 = [&](const std::string& ggufName, uint32_t N, uint32_t K,
                          GPUBuffer& wBuf, GPUBuffer& sBuf) {
            auto it = gguf.tensor_index.find(ggufName);
            if (it == gguf.tensor_index.end()) {
                fprintf(stderr, "  Missing tensor: %s\n", ggufName.c_str());
                return;
            }
            auto& ti = gguf.tensors[it->second];
            const uint8_t* data = fileData.data() + gguf.data_offset + ti.offset;
            auto rep = repack_q8_0(data, N, K);
            uploadQ8Weight(*gpu, pfx + ggufName, rep, wBuf, sBuf);
        };

        // Fuse Q/K/V weights into single QKV
        // GGUF has: blk.{i}.attn_q.weight, blk.{i}.attn_k.weight, blk.{i}.attn_v.weight
        {
            std::string qName = pfx + "attn_q.weight";
            std::string kName = pfx + "attn_k.weight";
            std::string vName = pfx + "attn_v.weight";
            auto qi = gguf.tensor_index.find(qName);
            auto ki = gguf.tensor_index.find(kName);
            auto vi = gguf.tensor_index.find(vName);
            if (qi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qi->second];
                auto& kt = gguf.tensors[ki->second];
                auto& vt = gguf.tensors[vi->second];
                // Each is Q8_0: N rows × K cols, 34 bytes per 32-element block
                uint32_t qN = qDim, kN = kvDim, vN = kvDim;
                uint32_t K = nEmbd;
                auto qr = repack_q8_0(fileData.data() + gguf.data_offset + qt.offset, qN, K);
                auto kr = repack_q8_0(fileData.data() + gguf.data_offset + kt.offset, kN, K);
                auto vr = repack_q8_0(fileData.data() + gguf.data_offset + vt.offset, vN, K);
                // Concatenate into fused QKV
                Q8Repacked fused;
                fused.N = qkvOut; fused.K = K;
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

        loadQ8(pfx + "attn_output.weight", nEmbd, qDim, lw.oW, lw.oS);

        // Fuse gate + up into single gate_up weight
        {
            std::string gateName = pfx + "ffn_gate.weight";
            std::string upName = pfx + "ffn_up.weight";
            auto gi = gguf.tensor_index.find(gateName);
            auto ui = gguf.tensor_index.find(upName);
            if (gi != gguf.tensor_index.end() && ui != gguf.tensor_index.end()) {
                auto& gt = gguf.tensors[gi->second];
                auto& ut = gguf.tensors[ui->second];
                auto gr = repack_q8_0(fileData.data() + gguf.data_offset + gt.offset,
                                       intermediateSize, nEmbd);
                auto ur = repack_q8_0(fileData.data() + gguf.data_offset + ut.offset,
                                       intermediateSize, nEmbd);
                Q8Repacked fused;
                fused.N = 2 * intermediateSize; fused.K = nEmbd;
                fused.weights.reserve(gr.weights.size() + ur.weights.size());
                fused.weights.insert(fused.weights.end(), gr.weights.begin(), gr.weights.end());
                fused.weights.insert(fused.weights.end(), ur.weights.begin(), ur.weights.end());
                fused.scales.reserve(gr.scales.size() + ur.scales.size());
                fused.scales.insert(fused.scales.end(), gr.scales.begin(), gr.scales.end());
                fused.scales.insert(fused.scales.end(), ur.scales.begin(), ur.scales.end());
                uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".gu", fused, lw.guW, lw.guS);
            }
        }

        loadQ8(pfx + "ffn_down.weight", nEmbd, intermediateSize, lw.dnW, lw.dnS);

        // Norm weights (fp32)
        auto loadNorm = [&](const std::string& ggufName, GPUBuffer& buf) {
            auto it = gguf.tensor_index.find(ggufName);
            if (it == gguf.tensor_index.end()) return;
            auto& ti = gguf.tensors[it->second];
            const uint8_t* data = fileData.data() + gguf.data_offset + ti.offset;
            // Dequantize if needed (F16 → F32)
            uint32_t nel = 1;
            for (auto d : ti.shape) nel *= (uint32_t)d;
            std::vector<float> fp32(nel);
            if (ti.type == GGUF_TYPE_F16) {
                const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
                for (uint32_t j = 0; j < nel; j++) {
                    // Simple fp16→fp32 conversion
                    uint32_t h = fp16[j];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f;
                    if (exp == 0) f = (sign << 31) | (mant << 13);
                    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
                    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                    memcpy(&fp32[j], &f, 4);
                }
            } else {
                memcpy(fp32.data(), data, nel * 4);
            }
            buf = gpu->createBuffer(ggufName, nel * 4);
            gpu->writeBuffer(buf, fp32.data(), nel * 4);
        };

        loadNorm(pfx + "attn_norm.weight", lw.inputNorm);
        loadNorm(pfx + "ffn_norm.weight", lw.postAttnNorm);
        loadNorm(pfx + "attn_q_norm.weight", lw.qNorm);
        loadNorm(pfx + "attn_k_norm.weight", lw.kNorm);

        if (i % 7 == 6 || i == nLayer - 1)
            printf("  loaded layer %u/%u\n", i + 1, nLayer);
    }

    // Final norm + embedding + LM head
    auto loadNormGlobal = [&](const std::string& name, GPUBuffer& buf) {
        auto it = gguf.tensor_index.find(name);
        if (it == gguf.tensor_index.end()) return;
        auto& ti = gguf.tensors[it->second];
        const uint8_t* data = fileData.data() + gguf.data_offset + ti.offset;
        uint32_t nel = 1;
        for (auto d : ti.shape) nel *= (uint32_t)d;
        std::vector<float> fp32(nel);
        if (ti.type == GGUF_TYPE_F16) {
            const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
            for (uint32_t j = 0; j < nel; j++) {
                uint32_t h = fp16[j]; uint32_t sign = (h>>15)&1;
                uint32_t exp = (h>>10)&0x1F; uint32_t mant = h&0x3FF;
                uint32_t f;
                if (exp==0) f=(sign<<31)|(mant<<13);
                else if (exp==31) f=(sign<<31)|0x7F800000|(mant<<13);
                else f=(sign<<31)|((exp+112)<<23)|(mant<<13);
                memcpy(&fp32[j], &f, 4);
            }
        } else memcpy(fp32.data(), data, nel * 4);
        buf = gpu->createBuffer(name, nel * 4);
        gpu->writeBuffer(buf, fp32.data(), nel * 4);
    };
    loadNormGlobal("output_norm.weight", finalNormW);

    // Embedding (fp32 or fp16 → fp32, stored CPU-side for per-token lookup)
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
                for (uint32_t j = 0; j < nel; j++) {
                    uint32_t h = fp16[j]; uint32_t sign = (h>>15)&1;
                    uint32_t exp = (h>>10)&0x1F; uint32_t mant = h&0x3FF;
                    uint32_t f;
                    if (exp==0) f=(sign<<31)|(mant<<13);
                    else if (exp==31) f=(sign<<31)|0x7F800000|(mant<<13);
                    else f=(sign<<31)|((exp+112)<<23)|(mant<<13);
                    memcpy(&embeddingCPU[j], &f, 4);
                }
            } else if (ti.type == GGUF_TYPE_Q8_0) {
                // Dequantize Q8_0 blocks to fp32
                uint32_t rows = (uint32_t)ti.shape[1];
                uint32_t cols = (uint32_t)ti.shape[0];
                uint32_t nBlocks = cols / 32;
                const Q8_0Block* blocks = reinterpret_cast<const Q8_0Block*>(data);
                for (uint32_t r = 0; r < rows; r++) {
                    for (uint32_t b = 0; b < nBlocks; b++) {
                        const auto& blk = blocks[r * nBlocks + b];
                        uint32_t h = blk.d; uint32_t sign = (h>>15)&1;
                        uint32_t exp_v = (h>>10)&0x1F; uint32_t mant = h&0x3FF;
                        uint32_t f;
                        if (exp_v==0) f=(sign<<31)|(mant<<13);
                        else if (exp_v==31) f=(sign<<31)|0x7F800000|(mant<<13);
                        else f=(sign<<31)|((exp_v+112)<<23)|(mant<<13);
                        float scale; memcpy(&scale, &f, 4);
                        for (int q = 0; q < 32; q++) {
                            embeddingCPU[r * cols + b * 32 + q] =
                                (float)blk.qs[q] * scale;
                        }
                    }
                }
            } else {
                memcpy(embeddingCPU.data(), data, nel * 4);
            }
            printf("  Embedding: %u tokens × %u dims (%s)\n",
                   nVocab, nEmbd,
                   ti.type == GGUF_TYPE_Q8_0 ? "Q8_0→f32" :
                   ti.type == GGUF_TYPE_F16 ? "f16→f32" : "f32");

            // Also upload as fp16 for LM head (tied embeddings)
            if (tieWordEmbeddings) {
                std::vector<uint16_t> fp16(nel);
                for (uint32_t j = 0; j < nel; j++) {
                    float v = embeddingCPU[j];
                    uint32_t fb; memcpy(&fb, &v, 4);
                    uint32_t s = (fb >> 16) & 0x8000;
                    int32_t e = ((fb >> 23) & 0xFF) - 112;
                    uint32_t m = (fb >> 13) & 0x3FF;
                    if (e <= 0) fp16[j] = (uint16_t)s;
                    else if (e > 30) fp16[j] = (uint16_t)(s | 0x7C00);
                    else fp16[j] = (uint16_t)(s | (e << 10) | m);
                }
                // Upload in chunks (Dawn's writeBuffer can fail on very large writes)
                uint64_t totalBytes = (uint64_t)nel * 2;
                lmHeadW = gpu->createBuffer("lm_head_fp16", totalBytes);
                const uint64_t CHUNK = 128 * 1024 * 1024; // 128MB chunks
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
    uint32_t half = headDim / 2;
    uint32_t maxSeq = 2048;
    std::vector<float> cosTable(maxSeq * half), sinTable(maxSeq * half);
    for (uint32_t pos = 0; pos < maxSeq; pos++) {
        for (uint32_t i = 0; i < half; i++) {
            float freq = 1.0f / powf(ropeTheta, (float)(2 * i) / headDim);
            float angle = pos * freq;
            cosTable[pos * half + i] = cosf(angle);
            sinTable[pos * half + i] = sinf(angle);
        }
    }
    ropeCosBuf = gpu->createBuffer("rope_cos", maxSeq * half * 4);
    ropeSinBuf = gpu->createBuffer("rope_sin", maxSeq * half * 4);
    gpu->writeBuffer(ropeCosBuf, cosTable.data(), maxSeq * half * 4);
    gpu->writeBuffer(ropeSinBuf, sinTable.data(), maxSeq * half * 4);
}

// ─── Build decode pipeline ───────────────────────────────────────────────────

void ModelRunner::buildDecodePipeline() {
    printf("  Building decode pipeline...\n"); fflush(stdout);
    printf("  sizeof(WGPUBindGroupEntry)=%zu sizeof(WGPUBindGroupLayoutEntry)=%zu\n",
           sizeof(WGPUBindGroupEntry), sizeof(WGPUBindGroupLayoutEntry));
    fflush(stdout);

    qDim = nHead * headDim;
    kvDim = nKvHeads * headDim;
    qkvOut = qDim + 2 * kvDim;
    uint32_t Q8_TILE = 8;
    uint32_t maxChunks = (maxSeqLen + gqaChunkSize - 1) / gqaChunkSize;

    // Load all needed kernels
    printf("    Loading kernels...\n"); fflush(stdout);
    auto& plRmsNorm    = loadKernel("rms_norm");
    printf("      rms_norm OK\n"); fflush(stdout);
    auto& plAddRmsNorm = loadKernel("add_rms_norm");
    printf("      add_rms_norm OK\n"); fflush(stdout);
    auto& plQ8Matmul   = loadKernel("q8_matmul");
    printf("      q8_matmul OK\n"); fflush(stdout);
    auto& plQ8MatAdd   = loadKernel("q8_matmul_add");
    printf("      q8_matmul_add OK\n"); fflush(stdout);
    auto& plFusedRope  = loadKernel("fused_qknorm_rope");
    printf("      fused_qknorm_rope OK\n"); fflush(stdout);
    auto& plChunkP1    = loadKernel("gqa_chunked_pass1");
    printf("      gqa_chunked_pass1 OK\n"); fflush(stdout);
    auto& plChunkP2    = loadKernel("gqa_chunked_pass2");
    printf("      gqa_chunked_pass2 OK\n"); fflush(stdout);
    auto& plSiluMul    = loadKernel("silu_mul_fused");
    printf("      silu_mul_fused OK\n"); fflush(stdout);
    auto& plFp16Gemm   = loadKernel("fp16_gemm");
    printf("      fp16_gemm OK\n"); fflush(stdout);

    // Create static params buffers
    printf("    Creating params...\n"); fflush(stdout);
    // Q8 params: [K, N] as 2 u32, padded to 16 bytes
    auto makeQ8Params = [&](const std::string& name, uint32_t K, uint32_t N) -> GPUBuffer {
        uint32_t data[4] = {K, N, 0, 0};
        auto buf = gpu->createBuffer(name, 16);
        gpu->writeBuffer(buf, data, 16);
        return buf;
    };

    auto q8QkvParams   = makeQ8Params("p_qkv", nEmbd, qkvOut);
    printf("    q8_qkv params OK\n"); fflush(stdout);
    auto q8OprojParams = makeQ8Params("p_oproj", qDim, nEmbd);
    auto q8GuParams    = makeQ8Params("p_gu", nEmbd, 2 * intermediateSize);
    auto q8DnParams    = makeQ8Params("p_dn", intermediateSize, nEmbd);
    printf("    All Q8 params OK\n"); fflush(stdout);

    // RMSNorm params: [stride, N, eps] = 3 values, stride=N=nEmbd
    printf("    Creating norm params...\n"); fflush(stdout);
    {
        uint32_t rn[4];
        rn[0] = nEmbd; rn[1] = nEmbd;
        float eps = rmsNormEps; memcpy(&rn[2], &eps, 4);
        rn[3] = 0;
        auto buf = gpu->createBuffer("p_rms", 16);
        gpu->writeBuffer(buf, rn, 16);
        paramsBufs["rms"] = buf;
    }

    // SiluMul params: [N]
    {
        uint32_t sm[4] = {intermediateSize, 0, 0, 0};
        auto buf = gpu->createBuffer("p_silu", 16);
        gpu->writeBuffer(buf, sm, 16);
        paramsBufs["silu"] = buf;
    }

    // FP16 GEMM params: [K, N] for LM head
    printf("    Creating lmhead params...\n"); fflush(stdout);
    {
        uint32_t fp[4] = {nEmbd, nVocab, 0, 0};
        auto buf = gpu->createBuffer("p_lmhead", 16);
        gpu->writeBuffer(buf, fp, 16);
        paramsBufs["lmhead"] = buf;
    }

    // Fused RoPE params
    printf("    Creating rope params...\n"); fflush(stdout);
    {
        ropeParamData.resize(32, 0);
        auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
        p[0] = nHead;      // n_head
        p[1] = qDim;       // q_size
        p[2] = kvDim;      // kv_size
        p[3] = 0;          // pos (dynamic)
        p[4] = headDim/2;  // half_rot
        p[5] = 0;          // cache_offset (dynamic)
        float eps = rmsNormEps;
        memcpy(&p[6], &eps, 4);
        fusedRopeParamsBuf = gpu->createBuffer("p_frope", 32);
        gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);
    }

    // Chunked attention params
    printf("    Creating chunked attn params...\n"); fflush(stdout);
    {
        chunkedAttnParamData.resize(32, 0);
        auto* p = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
        p[0] = nKvHeads * headDim;  // kv_stride
        p[1] = nHead / nKvHeads;    // n_rep
        p[2] = 0;                   // T_total (dynamic)
        p[3] = gqaChunkSize;        // chunk_size
        p[4] = 0;                   // n_chunks (dynamic)
        float scale = 1.0f / sqrtf((float)headDim);
        float neg_inf = -1e9f;
        memcpy(&p[5], &scale, 4);
        memcpy(&p[6], &neg_inf, 4);
        chunkedAttnParamsBuf = gpu->createBuffer("p_cattn", 32);
        gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);
    }

    // Build per-layer dispatches
    allDecodeDispatches.reserve(nLayer * 11 + 2);  // 11 per layer + 2 final
    for (uint32_t i = 0; i < nLayer; i++) {
        auto& lw = layerWeights[i];
        if (i % 7 == 0) { printf("    building layer %u/%u...\r", i, nLayer); fflush(stdout); }

        // Check weight availability
        if (!lw.qkvW.handle || !lw.oW.handle || !lw.guW.handle || !lw.dnW.handle) {
            fprintf(stderr, "\n  ERROR: layer %u missing weight buffers\n", i);
            exit(1);
        }
        if (!lw.inputNorm.handle || !lw.postAttnNorm.handle) {
            fprintf(stderr, "\n  ERROR: layer %u missing norm weights\n", i);
            exit(1);
        }
        if (!lw.qNorm.handle || !lw.kNorm.handle) {
            fprintf(stderr, "\n  WARNING: layer %u missing QK norm weights, using dummy\n", i);
            // Create small zero buffers as placeholders
            if (!lw.qNorm.handle) lw.qNorm = gpu->createBuffer("qnorm_dummy_" + std::to_string(i), headDim * 4);
            if (!lw.kNorm.handle) lw.kNorm = gpu->createBuffer("knorm_dummy_" + std::to_string(i), headDim * 4);
        }

        if (!lw.guW.handle || !lw.guS.handle) {
            fprintf(stderr, "\n  ERROR: layer %u missing gate_up weights!\n", i);
            exit(1);
        }
        if (!lw.dnW.handle || !lw.dnS.handle) {
            fprintf(stderr, "\n  ERROR: layer %u missing down weights!\n", i);
            exit(1);
        }

        // RMSNorm (first layer only)
        if (i == 0) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.inputNorm},
                {3, rstdBuf}, {4, paramsBufs["rms"]}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1});
        }

        // QKV matmul
        {
            auto bg = makeBG(plQ8Matmul, {
                {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, qkvBuf}, {5, q8QkvParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (qkvOut + Q8_TILE - 1) / Q8_TILE, 1});
        }

        // Fused QKnorm + RoPE + KV scatter
        {
            auto bg = makeBG(plFusedRope, {
                {0, qkvBuf}, {1, qRotBuf},
                {2, kvCache[i].K}, {3, kvCache[i].V},
                {4, ropeCosBuf}, {5, ropeSinBuf},
                {6, lw.qNorm}, {7, lw.kNorm},
                {8, fusedRopeParamsBuf}});
            allDecodeDispatches.push_back({plFusedRope.pipeline, bg,
                nHead + nKvHeads, 1, 1});
        }

        // Chunked attention pass 1
        {
            fprintf(stderr, "  about to create chunk1 vector...\n"); fflush(stderr);
            auto* cattn1 = new std::vector<std::pair<uint32_t, GPUBuffer>>();
            cattn1->push_back({0, qRotBuf});
            cattn1->push_back({1, kvCache[i].K});
            cattn1->push_back({2, kvCache[i].V});
            cattn1->push_back({3, attnPartialsBuf});
            cattn1->push_back({4, chunkedAttnParamsBuf});
            fprintf(stderr, "  chunk pass 1: %zu bindings\n", cattn1->size());
            auto bg = makeBG(plChunkP1, *cattn1);
            delete cattn1;
            allDecodeDispatches.push_back({plChunkP1.pipeline, bg,
                nHead, maxChunks, 1});  // n_chunks updated dynamically via params
        }

        // Chunked attention pass 2
        {
            auto bg = makeBG(plChunkP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, chunkedAttnParamsBuf}});
            allDecodeDispatches.push_back({plChunkP2.pipeline, bg,
                nHead, 1, 1});
        }

        // O projection
        {
            auto bg = makeBG(plQ8Matmul, {{0,attnOutBuf},{1,lw.oW},{2,lw.oS},{3,zeroBiasE},{4,projOutBuf},{5,q8OprojParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (nEmbd + Q8_TILE - 1) / Q8_TILE, 1});
        }

        // Add + RMSNorm (fused residual + next norm)
        {
            auto bg = makeBG(plAddRmsNorm, {
                {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                {3, lw.postAttnNorm}, {4, rstdBuf},
                {5, paramsBufs["rms"]}});
            allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg,
                1, 1, 1});
        }

        // Gate+Up matmul
        {
            auto bg = makeBG(plQ8Matmul, {{0,normOutBuf},{1,lw.guW},{2,lw.guS},{3,zeroBiasGU},{4,gateUpBuf},{5,q8GuParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (2 * intermediateSize + Q8_TILE - 1) / Q8_TILE, 1});
        }

        // SiLU * mul (fused)
        {
            auto bg = makeBG(plSiluMul, {
                {0, gateUpBuf}, {1, siluOutBuf}, {2, paramsBufs["silu"]}});
            allDecodeDispatches.push_back({plSiluMul.pipeline, bg,
                (intermediateSize + 127) / 128, 1, 1});
        }

        // Down projection + residual add (fused)
        {
            auto bg = makeBG(plQ8MatAdd, {{0,siluOutBuf},{1,lw.dnW},{2,lw.dnS},{3,zeroBiasE},{4,xBuf},{5,q8DnParams}});
            allDecodeDispatches.push_back({plQ8MatAdd.pipeline, bg,
                1, (nEmbd + Q8_TILE - 1) / Q8_TILE, 1});
        }

        // RMSNorm for next layer (or final)
        if (i < nLayer - 1) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, layerWeights[i+1].inputNorm},
                {3, rstdBuf}, {4, paramsBufs["rms"]}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1});
        }
    }

    // Final RMSNorm + LM head
    {
        auto bg = makeBG(plRmsNorm, {
            {0, xBuf}, {1, normOutBuf}, {2, finalNormW},
            {3, rstdBuf}, {4, paramsBufs["rms"]}});
        allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1});
    }

    // LM head (fp16 GEMM): normOut × lmHeadW → logits
    // FP16 GEMM bindings: X(0), W(1), Bias(2), Y(3), _params_(4)
    {
        // Grid for fp16 GEMM: (T=1, ceil(N/8))
        uint32_t FP16_TILE = 8;
        auto bg = makeBG(plFp16Gemm, {
            {0, normOutBuf}, {1, lmHeadW}, {2, zeroBiasV},
            {3, logitsBuf}, {4, paramsBufs["lmhead"]}});
        allDecodeDispatches.push_back({plFp16Gemm.pipeline, bg,
            1, (nVocab + FP16_TILE - 1) / FP16_TILE, 1});
    }

    printf("  Pre-recorded %zu decode dispatches (%u layers)\n",
           allDecodeDispatches.size(), nLayer);
}

// ─── Inference ───────────────────────────────────────────────────────────────

void ModelRunner::uploadEmbedding(int32_t tokenId) {
    // CPU embedding lookup + GPU upload
    if (tokenId < 0 || (uint32_t)tokenId >= nVocab) tokenId = 0;
    const float* emb = embeddingCPU.data() + tokenId * nEmbd;
    gpu->writeBuffer(xBuf, emb, nEmbd * 4);
}

void ModelRunner::updateDecodeParams(uint32_t pos, uint32_t cacheLen) {
    // Update fused RoPE params: pos at offset 12, cache_offset at offset 20
    auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
    p[3] = pos;                                 // pos
    p[5] = cacheLen * nKvHeads * headDim;       // cache_offset
    gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);

    // Update chunked attention params: T_total at offset 8, n_chunks at offset 16
    uint32_t T_total = cacheLen + 1;
    uint32_t n_chunks = (T_total + gqaChunkSize - 1) / gqaChunkSize;
    auto* cp = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
    cp[2] = T_total;
    cp[4] = n_chunks;
    gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);
}

std::vector<float> ModelRunner::decode(int32_t tokenId, uint32_t posOffset) {
    // 1. Upload embedding
    uploadEmbedding(tokenId);

    // 2. Update dynamic params
    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    // 3. Submit all dispatches + readback logits
    auto result = gpu->submitAndReadback(
        allDecodeDispatches, logitsBuf, nVocab * 4);

    // 4. Update KV cache lengths
    for (uint32_t i = 0; i < nLayer; i++)
        kvCache[i].len++;

    // 5. Return logits as float vector
    std::vector<float> logits(nVocab);
    memcpy(logits.data(), result.data(), nVocab * 4);
    return logits;
}

int32_t ModelRunner::argmax(const std::vector<float>& logits) {
    return (int32_t)std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end()));
}








