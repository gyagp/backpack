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

    // Build decode pipeline
    buildDecodePipeline();

    // Compute RoPE tables
    computeRopeTables();

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
        loadQ8(pfx + "ffn_gate.weight", intermediateSize, nEmbd, lw.guW, lw.guS);
        // TODO: fuse gate+up projection (need to concatenate ffn_gate + ffn_up)
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

    // Embedding (fp32 or fp16 → fp32)
    {
        auto it = gguf.tensor_index.find("token_embd.weight");
        if (it != gguf.tensor_index.end()) {
            auto& ti = gguf.tensors[it->second];
            printf("  Embedding: type=%u\n", ti.type);
            // Store as CPU-side for token lookup
            // (embedding is not a GPU kernel — we do CPU lookup + upload per token)
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
    printf("  Building decode pipeline...\n");
    // TODO: Pre-build all dispatch sequences from manifest decode_plan
    // This is the key performance optimization: all bind groups are pre-created
    // and dispatches are just pipeline + bind_group + grid triples.
    printf("  [TODO] Pre-recorded decode dispatches\n");
}

// ─── Inference ───────────────────────────────────────────────────────────────

std::vector<float> ModelRunner::prefill(const std::vector<int32_t>& tokenIds) {
    // TODO: implement multi-token prefill
    printf("[TODO] prefill %zu tokens\n", tokenIds.size());
    return std::vector<float>(nVocab, 0.0f);
}

std::vector<float> ModelRunner::decode(int32_t tokenId, uint32_t posOffset) {
    // TODO: implement single-token decode via pre-recorded dispatches
    printf("[TODO] decode token %d at pos %u\n", tokenId, posOffset);
    return std::vector<float>(nVocab, 0.0f);
}

int32_t ModelRunner::argmax(const std::vector<float>& logits) {
    return (int32_t)std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end()));
}
