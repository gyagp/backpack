#include "gguf_loader.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// GGUF metadata value types
enum GGUFMetaType : uint32_t {
    META_UINT8 = 0, META_INT8 = 1, META_UINT16 = 2, META_INT16 = 3,
    META_UINT32 = 4, META_INT32 = 5, META_FLOAT32 = 6, META_BOOL = 7,
    META_STRING = 8, META_ARRAY = 9, META_UINT64 = 10, META_INT64 = 11,
    META_FLOAT64 = 12,
};

static uint64_t read_u64(FILE* f) { uint64_t v; fread(&v, 8, 1, f); return v; }
static uint32_t read_u32(FILE* f) { uint32_t v; fread(&v, 4, 1, f); return v; }
static float    read_f32(FILE* f) { float v;    fread(&v, 4, 1, f); return v; }
static double   read_f64(FILE* f) { double v;   fread(&v, 8, 1, f); return v; }
static uint8_t  read_u8 (FILE* f) { uint8_t v;  fread(&v, 1, 1, f); return v; }

static std::string read_string(FILE* f) {
    uint64_t len = read_u64(f);
    std::string s(len, '\0');
    fread(s.data(), 1, len, f);
    return s;
}

// Read a metadata value and store it
static GGUFMetaValue read_meta_value(FILE* f, uint32_t type) {
    switch (type) {
        case META_UINT8:   return (uint8_t)read_u8(f);
        case META_INT8:    return (int8_t)read_u8(f);
        case META_UINT16:  { uint16_t v; fread(&v, 2, 1, f); return v; }
        case META_INT16:   { int16_t v;  fread(&v, 2, 1, f); return v; }
        case META_UINT32:  return read_u32(f);
        case META_INT32:   { int32_t v;  fread(&v, 4, 1, f); return v; }
        case META_FLOAT32: return read_f32(f);
        case META_BOOL:    return (bool)read_u8(f);
        case META_STRING:  return read_string(f);
        case META_UINT64:  return read_u64(f);
        case META_INT64:   { int64_t v;  fread(&v, 8, 1, f); return v; }
        case META_FLOAT64: return read_f64(f);
        case META_ARRAY: {
            // For string arrays, store as vector<string>; others: skip
            uint32_t elem_type = read_u32(f);
            uint64_t count = read_u64(f);
            if (elem_type == META_STRING) {
                std::vector<std::string> arr;
                arr.reserve(count);
                for (uint64_t i = 0; i < count; i++)
                    arr.push_back(read_string(f));
                return arr;
            }
            // Skip non-string arrays
            for (uint64_t i = 0; i < count; i++)
                read_meta_value(f, elem_type);  // discard
            return std::string("");  // placeholder
        }
        default:
            fprintf(stderr, "Unknown GGUF meta type: %u\n", type);
            return std::string("");
    }
}

bool GGUFFile::open(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic = read_u32(f);
    if (magic != 0x46554747) { fclose(f); return false; }  // "GGUF"

    version = read_u32(f);
    n_tensors = read_u64(f);
    uint64_t n_kv = read_u64(f);

    // Parse metadata key-value pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(f);
        uint32_t vtype = read_u32(f);
        GGUFMetaValue val = read_meta_value(f, vtype);
        metadata[key] = std::move(val);
    }

    // Read tensor infos
    tensors.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        auto& t = tensors[i];
        t.name = read_string(f);
        uint32_t n_dims = read_u32(f);
        t.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++)
            t.shape[d] = read_u64(f);
        t.type = static_cast<GGUFType>(read_u32(f));
        t.offset = read_u64(f);
        tensor_index[t.name] = i;
    }

    long pos = ftell(f);
    data_offset = (pos + 31) & ~31ULL;

    fclose(f);
    return true;
}

// ─── Metadata accessors ──────────────────────────────────────────────────────

std::string GGUFFile::getString(const std::string& key,
                                 const std::string& def) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return def;
    if (auto* s = std::get_if<std::string>(&it->second)) return *s;
    return def;
}

uint32_t GGUFFile::getU32(const std::string& key, uint32_t def) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return def;
    if (auto* v = std::get_if<uint32_t>(&it->second)) return *v;
    if (auto* v = std::get_if<int32_t> (&it->second)) return (uint32_t)*v;
    if (auto* v = std::get_if<uint64_t>(&it->second)) return (uint32_t)*v;
    if (auto* v = std::get_if<int64_t> (&it->second)) return (uint32_t)*v;
    if (auto* v = std::get_if<uint16_t>(&it->second)) return *v;
    if (auto* v = std::get_if<int16_t> (&it->second)) return (uint32_t)*v;
    if (auto* v = std::get_if<uint8_t> (&it->second)) return *v;
    if (auto* v = std::get_if<int8_t>  (&it->second)) return (uint32_t)*v;
    return def;
}

float GGUFFile::getFloat(const std::string& key, float def) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return def;
    if (auto* v = std::get_if<float> (&it->second)) return *v;
    if (auto* v = std::get_if<double>(&it->second)) return (float)*v;
    return def;
}

bool GGUFFile::getBool(const std::string& key, bool def) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return def;
    if (auto* v = std::get_if<bool>(&it->second)) return *v;
    return def;
}

bool GGUFFile::hasKey(const std::string& key) const {
    return metadata.count(key) > 0;
}

// ─── Extract model config ────────────────────────────────────────────────────

ModelConfig extractModelConfig(const GGUFFile& gguf) {
    ModelConfig cfg;

    cfg.arch = gguf.getString("general.architecture", "llama");
    std::string a = cfg.arch;

    cfg.nLayer          = gguf.getU32(a + ".block_count");
    cfg.nEmbd           = gguf.getU32(a + ".embedding_length");
    cfg.intermediateSize = gguf.getU32(a + ".feed_forward_length");
    cfg.nHead           = gguf.getU32(a + ".attention.head_count");
    cfg.nKvHeads        = gguf.getU32(a + ".attention.head_count_kv", cfg.nHead);
    cfg.rmsNormEps      = gguf.getFloat(a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.ropeTheta       = gguf.getFloat(a + ".rope.freq_base", 1000000.0f);
    cfg.tieWordEmbeddings = !gguf.hasKey("output.weight");

    // head_dim: either from rope.dimension_count or derived
    uint32_t ropeDim = gguf.getU32(a + ".rope.dimension_count", 0);
    cfg.headDim = ropeDim > 0 ? ropeDim : cfg.nEmbd / cfg.nHead;
    // K/V per-head dim — separate from headDim for some archs (qwen35moe).
    // attention.key_length holds the actual K/V per-head dim when present.
    cfg.kvHeadDim = gguf.getU32(a + ".attention.key_length", cfg.headDim);

    // n_vocab from embedding tensor shape
    auto it = gguf.tensor_index.find("token_embd.weight");
    if (it != gguf.tensor_index.end()) {
        auto& ti = gguf.tensors[it->second];
        // GGUF shape is (cols, rows) for 2D — shape[1] = n_vocab
        if (ti.shape.size() >= 2)
            cfg.nVocab = (uint32_t)ti.shape[1];
        else
            cfg.nVocab = (uint32_t)ti.shape[0];
    }

    // Check for QK norm (Qwen3 has it, LLaMA doesn't)
    cfg.hasQkNorm = gguf.tensor_index.count("blk.0.attn_q_norm.weight") > 0;

    // Gemma family: GELU activation, embedding scaling, logit softcapping
    bool isGemma = (a == "gemma" || a == "gemma2" || a == "gemma3" || a == "gemma4");
    if (isGemma) {
        cfg.activation = ActivationType::GELU;
        cfg.embeddingScale = sqrtf((float)cfg.nEmbd);
    }

    // Logit softcapping (Gemma 2/4)
    cfg.logitSoftcap = gguf.getFloat(a + ".final_logit_softcapping", 0.0f);

    // Sliding window attention
    cfg.slidingWindow = gguf.getU32(a + ".attention.sliding_window", 0);

    // Populate per-layer attention types
    if (cfg.slidingWindow > 0 && cfg.nLayer > 0) {
        cfg.layerAttnTypes.resize(cfg.nLayer, AttnLayerType::Global);
        // Gemma 2/3: alternating pattern (even layers = sliding, odd = global)
        if (a == "gemma2" || a == "gemma3") {
            for (uint32_t i = 0; i < cfg.nLayer; i++)
                cfg.layerAttnTypes[i] = (i % 2 == 0) ? AttnLayerType::SlidingWindow
                                                       : AttnLayerType::Global;
        }
        // Gemma 4: read pattern from metadata if available, else default heuristic
        // llama.cpp stores this as LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN
        if (a == "gemma4") {
            // Default Gemma 4 pattern: groups of 5 sliding + 1 global
            for (uint32_t i = 0; i < cfg.nLayer; i++)
                cfg.layerAttnTypes[i] = ((i + 1) % 6 == 0) ? AttnLayerType::Global
                                                              : AttnLayerType::SlidingWindow;
        }
    }

    // Shared KV layers and PLE (Gemma 4)
    cfg.sharedKvLayers = gguf.getU32(a + ".attention.shared_kv_layers", 0);
    cfg.pleSize = gguf.getU32(a + ".embedding_length_per_layer_input", 0);

    // Sandwich norm detection (Gemma 4 has post_norm / post_ffw_norm tensors)
    cfg.hasSandwichNorm = gguf.tensor_index.count("blk.0.post_norm.weight") > 0 ||
                          gguf.tensor_index.count("blk.0.attn_post_norm.weight") > 0;

    // MTP head detection
    cfg.hasMtp = gguf.tensor_index.count("mtp.0.blk.0.attn_q.weight") > 0 ||
                 gguf.tensor_index.count("mtp.0.enorm.weight") > 0;
    // qwen35moe nextN-predict style: tensors live at blk.<lastLayer>.nextn.*
    if (!cfg.hasMtp) {
        uint32_t nextnLayers = gguf.getU32(a + ".nextn_predict_layers", 0);
        if (nextnLayers > 0) {
            cfg.hasMtp = true;
            cfg.mtpNumLayers = nextnLayers;
        }
    }

    // MoE config (qwen35moe / qwen3moe / qwen2moe / mixtral / deepseek2 / dbrx / grok)
    cfg.numExperts          = gguf.getU32(a + ".expert_count", 0);
    cfg.numExpertsPerTok    = gguf.getU32(a + ".expert_used_count", 0);
    cfg.moeIntermediateSize = gguf.getU32(a + ".expert_feed_forward_length", 0);
    cfg.moeSharedIntermediateSize = gguf.getU32(a + ".expert_shared_feed_forward_length",
                                                cfg.moeIntermediateSize);

    // SSM (Mamba) hyperparameters — hybrid archs like qwen35moe
    cfg.ssmInnerSize           = gguf.getU32(a + ".ssm.inner_size", 0);
    cfg.ssmStateSize           = gguf.getU32(a + ".ssm.state_size", 0);
    cfg.ssmConvKernel          = gguf.getU32(a + ".ssm.conv_kernel", 0);
    cfg.ssmGroupCount          = gguf.getU32(a + ".ssm.group_count", 0);
    cfg.ssmTimeStepRank        = gguf.getU32(a + ".ssm.time_step_rank", 0);
    cfg.fullAttentionInterval  = gguf.getU32(a + ".full_attention_interval", 0);

    return cfg;
}

// ─── Q8_0 repacking ──────────────────────────────────────────────────────────

Q8Repacked repack_q8_0(const void* raw_data, uint32_t N, uint32_t K) {
    Q8Repacked result;
    result.N = N;
    result.K = K;

    uint32_t n_blocks_per_row = K / 32;
    const auto* blocks = reinterpret_cast<const Q8_0Block*>(raw_data);

    result.weights.resize(N * (K / 4));
    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t blk = 0; blk < n_blocks_per_row; blk++) {
            const auto& b = blocks[row * n_blocks_per_row + blk];
            for (uint32_t j = 0; j < 32; j += 4) {
                uint32_t packed = 0;
                packed |= (uint32_t)(uint8_t)b.qs[j];
                packed |= (uint32_t)(uint8_t)b.qs[j+1] << 8;
                packed |= (uint32_t)(uint8_t)b.qs[j+2] << 16;
                packed |= (uint32_t)(uint8_t)b.qs[j+3] << 24;
                result.weights[row * (K/4) + blk * 8 + j/4] = packed;
            }
        }
    }

    uint32_t total_blocks = N * n_blocks_per_row;
    result.scales.resize((total_blocks + 1) / 2);
    for (uint32_t i = 0; i < total_blocks; i++) {
        uint16_t d = blocks[i].d;
        if (i % 2 == 0)
            result.scales[i / 2] = d;
        else
            result.scales[i / 2] |= (uint32_t)d << 16;
    }

    return result;
}

// ─── K-quant packing ────────────────────────────────────────────────────────

KQuantPacked pack_q4k(const void* raw_data, uint32_t N, uint32_t K) {
    KQuantPacked result;
    result.N = N;
    result.K = K;
    const uint32_t QK_K = 256;
    const uint32_t BLOCK_SIZE = 144;  // bytes per Q4_K block
    const uint32_t BLOCK_WORDS = BLOCK_SIZE / 4;  // 36
    result.nBlocks = (K + QK_K - 1) / QK_K;
    result.rowStrideWords = result.nBlocks * BLOCK_WORDS;

    const uint8_t* src = reinterpret_cast<const uint8_t*>(raw_data);
    uint64_t rowBytes = (uint64_t)result.nBlocks * BLOCK_SIZE;

    result.data.resize((uint64_t)N * result.rowStrideWords);
    for (uint32_t row = 0; row < N; row++) {
        const uint8_t* rowSrc = src + row * rowBytes;
        uint32_t* rowDst = result.data.data() + row * result.rowStrideWords;
        memcpy(rowDst, rowSrc, rowBytes);
    }
    return result;
}

KQuantPacked pack_q5k(const void* raw_data, uint32_t N, uint32_t K) {
    KQuantPacked result;
    result.N = N;
    result.K = K;
    const uint32_t QK_K = 256;
    const uint32_t BLOCK_SIZE = 176;  // bytes per Q5_K block
    const uint32_t BLOCK_WORDS = BLOCK_SIZE / 4;  // 44
    result.nBlocks = (K + QK_K - 1) / QK_K;
    result.rowStrideWords = result.nBlocks * BLOCK_WORDS;

    const uint8_t* src = reinterpret_cast<const uint8_t*>(raw_data);
    uint64_t rowBytes = (uint64_t)result.nBlocks * BLOCK_SIZE;

    result.data.resize((uint64_t)N * result.rowStrideWords);
    for (uint32_t row = 0; row < N; row++) {
        const uint8_t* rowSrc = src + row * rowBytes;
        uint32_t* rowDst = result.data.data() + row * result.rowStrideWords;
        memcpy(rowDst, rowSrc, rowBytes);
    }
    return result;
}

KQuantPacked pack_q6k(const void* raw_data, uint32_t N, uint32_t K) {
    KQuantPacked result;
    result.N = N;
    result.K = K;
    const uint32_t QK_K = 256;
    const uint32_t BLOCK_SIZE = 210;  // bytes per Q6_K block (not word-aligned!)
    result.nBlocks = (K + QK_K - 1) / QK_K;
    // Pad row stride to 4-byte alignment
    uint32_t rowBytes = result.nBlocks * BLOCK_SIZE;
    uint32_t rowBytesPadded = (rowBytes + 3) & ~3u;
    result.rowStrideWords = rowBytesPadded / 4;

    const uint8_t* src = reinterpret_cast<const uint8_t*>(raw_data);

    result.data.resize((uint64_t)N * result.rowStrideWords, 0);
    for (uint32_t row = 0; row < N; row++) {
        const uint8_t* rowSrc = src + (uint64_t)row * (uint64_t)result.nBlocks * BLOCK_SIZE;
        uint8_t* rowDst = reinterpret_cast<uint8_t*>(result.data.data() + row * result.rowStrideWords);
        memcpy(rowDst, rowSrc, rowBytes);
    }
    return result;
}

// ─── K-quant dequantization ─────────────────────────────────────────────────

static float kq_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0)       f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else                f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

void dequant_kquant(const void* raw_data, float* out, uint32_t N, uint32_t K, GGUFType type) {
    const uint8_t* src = reinterpret_cast<const uint8_t*>(raw_data);
    const uint32_t QK_K = 256;
    uint32_t nBlocks = K / QK_K;

    if (type == GGUF_TYPE_Q4_K) {
        const uint32_t BLOCK_SIZE = 144;
        for (uint32_t row = 0; row < N; row++) {
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = src + ((uint64_t)row * nBlocks + b) * BLOCK_SIZE;
                uint16_t d_u16; memcpy(&d_u16, blk, 2);
                uint16_t dmin_u16; memcpy(&dmin_u16, blk + 2, 2);
                float d = kq_fp16_to_f32(d_u16);
                float dmin = kq_fp16_to_f32(dmin_u16);
                uint8_t scales[12];
                memcpy(scales, blk + 4, 12);

                for (uint32_t sb = 0; sb < 8; sb++) {
                    uint32_t sc_u, mn_u;
                    if (sb < 4) {
                        sc_u = scales[sb] & 0x3F;
                        mn_u = scales[sb + 4] & 0x3F;
                    } else {
                        uint32_t j = sb - 4;
                        sc_u = (scales[j + 8] & 0x0F) | ((scales[j] >> 2) & 0x30);
                        mn_u = (scales[j + 8] >> 4) | ((scales[j + 4] >> 2) & 0x30);
                    }
                    float sc = d * (float)sc_u;
                    float mn = dmin * (float)mn_u;
                    uint32_t g = sb / 2;
                    bool hi = (sb & 1) == 1;
                    for (uint32_t i = 0; i < 32; i++) {
                        uint32_t kidx = b * QK_K + sb * 32 + i;
                        if (kidx < K) {
                            uint8_t qb = blk[16 + g * 32 + i];
                            uint8_t q = hi ? ((qb >> 4) & 0x0F) : (qb & 0x0F);
                            out[row * K + kidx] = sc * (float)q - mn;
                        }
                    }
                }
            }
        }
    } else if (type == GGUF_TYPE_Q5_K) {
        const uint32_t BLOCK_SIZE = 176;
        for (uint32_t row = 0; row < N; row++) {
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = src + ((uint64_t)row * nBlocks + b) * BLOCK_SIZE;
                uint16_t d_u16; memcpy(&d_u16, blk, 2);
                uint16_t dmin_u16; memcpy(&dmin_u16, blk + 2, 2);
                float d = kq_fp16_to_f32(d_u16);
                float dmin = kq_fp16_to_f32(dmin_u16);
                uint8_t scales[12];
                memcpy(scales, blk + 4, 12);

                for (uint32_t sb = 0; sb < 8; sb++) {
                    uint32_t sc_u, mn_u;
                    if (sb < 4) {
                        sc_u = scales[sb] & 0x3F;
                        mn_u = scales[sb + 4] & 0x3F;
                    } else {
                        uint32_t j = sb - 4;
                        sc_u = (scales[j + 8] & 0x0F) | ((scales[j] >> 2) & 0x30);
                        mn_u = (scales[j + 8] >> 4) | ((scales[j + 4] >> 2) & 0x30);
                    }
                    float sc = d * (float)sc_u;
                    float mn = dmin * (float)mn_u;
                    uint32_t g = sb / 2;
                    bool hi = (sb & 1) == 1;
                    for (uint32_t i = 0; i < 32; i++) {
                        uint32_t kidx = b * QK_K + sb * 32 + i;
                        if (kidx < K) {
                            uint8_t qb = blk[48 + g * 32 + i];
                            uint8_t q_lo = hi ? ((qb >> 4) & 0x0F) : (qb & 0x0F);
                            uint8_t qh_byte = blk[16 + i];
                            uint8_t q_hi = (qh_byte >> sb) & 1;
                            uint8_t q = q_lo | (q_hi << 4);
                            out[row * K + kidx] = sc * (float)q - mn;
                        }
                    }
                }
            }
        }
    } else if (type == GGUF_TYPE_Q6_K) {
        // Q6_K block layout (210 bytes, QK_K=256):
        //   ql[128]   (offset 0):   low 4 bits, packed 2 per byte
        //   qh[64]    (offset 128): high 2 bits, packed 4 per byte
        //   scales[16](offset 192): int8 scale per 16-element sub-block
        //   d (fp16)  (offset 208): super-block scale
        const uint32_t BLOCK_SIZE = 210;
        for (uint32_t row = 0; row < N; row++) {
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = src + ((uint64_t)row * nBlocks + b) * BLOCK_SIZE;
                uint16_t d_u16; memcpy(&d_u16, blk + 208, 2);
                float d = kq_fp16_to_f32(d_u16);

                for (uint32_t sb = 0; sb < 16; sb++) {
                    int8_t sc_i8 = (int8_t)blk[192 + sb];
                    float sc = (float)sc_i8;
                    for (uint32_t i = 0; i < 16; i++) {
                        uint32_t kidx = b * QK_K + sb * 16 + i;
                        if (kidx >= K) continue;
                        uint32_t eidx = sb * 16 + i;
                        uint8_t ql_byte = blk[eidx / 2];
                        uint8_t ql = (eidx & 1) ? ((ql_byte >> 4) & 0x0F) : (ql_byte & 0x0F);
                        uint8_t qh_byte = blk[128 + eidx / 4];
                        uint32_t qh_shift = (eidx % 4) * 2;
                        uint8_t qh = (qh_byte >> qh_shift) & 0x03;
                        int q6 = (int)(ql | (qh << 4)) - 32;
                        out[row * K + kidx] = d * sc * (float)q6;
                    }
                }
            }
        }
    }
}

// ─── IQ-quant + Q2_K/Q3_K packers ──────────────────────────────────────────
// Each packer copies raw blocks into a u32-aligned per-row buffer, padding
// non-aligned block sizes so that block N starts at a fixed word stride
// regardless of u32 alignment of the raw stride. WGSL kernels then use
// `b * BLOCK_WORDS` to index into each block.

#include "quants_iq_tables.h"

static KQuantPacked pack_blocked(const void* raw_data, uint32_t N, uint32_t K,
                                  uint32_t blockBytes, uint32_t blockWordsPadded) {
    KQuantPacked result;
    result.N = N;
    result.K = K;
    const uint32_t QK_K_LOCAL = 256;
    result.nBlocks = (K + QK_K_LOCAL - 1) / QK_K_LOCAL;
    result.rowStrideWords = result.nBlocks * blockWordsPadded;

    const uint8_t* src = reinterpret_cast<const uint8_t*>(raw_data);
    result.data.resize((uint64_t)N * result.rowStrideWords, 0);
    for (uint32_t row = 0; row < N; row++) {
        const uint8_t* rowSrc = src + (uint64_t)row * result.nBlocks * blockBytes;
        uint8_t* rowDst = reinterpret_cast<uint8_t*>(result.data.data() + row * result.rowStrideWords);
        for (uint32_t b = 0; b < result.nBlocks; b++) {
            memcpy(rowDst + b * blockWordsPadded * 4, rowSrc + b * blockBytes, blockBytes);
        }
    }
    return result;
}

KQuantPacked pack_q2k(const void* raw_data, uint32_t N, uint32_t K) {
    return pack_blocked(raw_data, N, K, /*blockBytes=*/84, /*blockWordsPadded=*/21);
}
KQuantPacked pack_q3k(const void* raw_data, uint32_t N, uint32_t K) {
    return pack_blocked(raw_data, N, K, /*blockBytes=*/110, /*blockWordsPadded=*/28);
}
KQuantPacked pack_iq2s(const void* raw_data, uint32_t N, uint32_t K) {
    return pack_blocked(raw_data, N, K, /*blockBytes=*/82, /*blockWordsPadded=*/21);
}
KQuantPacked pack_iq3s(const void* raw_data, uint32_t N, uint32_t K) {
    return pack_blocked(raw_data, N, K, /*blockBytes=*/110, /*blockWordsPadded=*/28);
}
KQuantPacked pack_iq4xs(const void* raw_data, uint32_t N, uint32_t K) {
    return pack_blocked(raw_data, N, K, /*blockBytes=*/136, /*blockWordsPadded=*/34);
}

// Codebook accessors (uploaded once to GPU for IQ matmul kernels).
const uint32_t* getIq3sGrid(uint32_t* out_count) {
    if (out_count) *out_count = 512u;
    return reinterpret_cast<const uint32_t*>(iq3s_grid);
}
const uint32_t* getIq2sGridU32(uint32_t* out_count) {
    if (out_count) *out_count = 2048u;  // 1024 uint64 entries = 2048 u32
    return reinterpret_cast<const uint32_t*>(iq2s_grid);
}

// ─── IQ-quant + Q2_K/Q3_K dequantizers ──────────────────────────────────────
// Ported from llama.cpp/ggml-quants.c (MIT). Each block covers QK_K=256 elements.

static float fp16_val(uint16_t h);  // fwd decl — defined below

static constexpr int QK_K = 256;

// Block sizes (bytes per QK_K=256 elements unless noted)
static constexpr int BSZ_Q2_K   =  84;  // 2*fp16 + 16 + 64
static constexpr int BSZ_Q3_K   = 110;  // fp16 + 32 + 64 + 12
static constexpr int BSZ_IQ2_S  =  82;  // fp16 + 64 + 8 + 8
static constexpr int BSZ_IQ3_S  = 110;  // fp16 + 64 + 8 + 32 + 4
static constexpr int BSZ_IQ3_XXS=  98;  // fp16 + 96
static constexpr int BSZ_IQ4_XS = 136;  // fp16 + 2 + 4 + 128
static constexpr int BSZ_IQ4_NL =  18;  // fp16 + 16  (per 32-elem block, not QK_K)

static void dq_q2_K(const uint8_t* data, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_Q2_K;
        float d   = fp16_val(*(const uint16_t*)(x + 0));
        float mn  = fp16_val(*(const uint16_t*)(x + 2));
        const uint8_t* scales = x + 4;
        const uint8_t* q = x + 4 + 16;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (int8_t)((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (int8_t)((q[l+16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dq_q3_K(const uint8_t* data, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_Q3_K;
        const uint8_t* hm = x + 2;
        const uint8_t* q  = x + 2 + 32;
        const uint8_t* sc = x + 2 + 32 + 64;
        float d_all = fp16_val(*(const uint16_t*)(x + 2 + 32 + 64 + 12 - 2));  // fp16 d at end? No — at start of block
        d_all = fp16_val(*(const uint16_t*)(x));
        hm = x + 2;
        q  = x + 2 + 32;
        sc = x + 2 + 32 + 64;
        uint8_t m = 1;
        memcpy(aux, sc, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static void dq_iq2_s(const uint8_t* data, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    float db[2];
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_IQ2_S;
        float d = fp16_val(*(const uint16_t*)(x + 0));
        const uint8_t* qs = x + 2;           // 64 bytes
        const uint8_t* qh = x + 2 + 64;      // 8 bytes
        const uint8_t* scales = x + 2 + 64 + 8;  // 8 bytes
        const uint8_t* signs = qs + QK_K/8;  // signs are in last 32 bytes of qs region: per llama.cpp signs = qs + QK_K/8 = qs+32
        for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
            db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                float dl = db[l/2];
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300)));
                for (int j = 0; j < 8; j++)
                    y[j] = dl * grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.f : 1.f);
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

static void dq_iq3_xxs(const uint8_t* data, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    uint32_t aux32;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_IQ3_XXS;
        float d = fp16_val(*(const uint16_t*)(x + 0));
        const uint8_t* qs = x + 2;             // QK_K/4 = 64 bytes
        const uint8_t* scales_and_signs = qs + QK_K/4;  // 32 bytes
        for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
            memcpy(&aux32, scales_and_signs + 4*ib32, 4);
            float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                const uint8_t* g1 = (const uint8_t*)(iq3xxs_grid + qs[2*l + 0]);
                const uint8_t* g2 = (const uint8_t*)(iq3xxs_grid + qs[2*l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db * g1[j] * ((signs & kmask_iq2xs[j+0]) ? -1.f : 1.f);
                    y[j + 4] = db * g2[j] * ((signs & kmask_iq2xs[j+4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

static void dq_iq3_s(const uint8_t* data, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_IQ3_S;
        float d = fp16_val(*(const uint16_t*)(x + 0));
        const uint8_t* qs    = x + 2;                 // QK_K/4 = 64
        const uint8_t* qh    = x + 2 + 64;            // QK_K/32 = 8
        const uint8_t* signs = x + 2 + 64 + 8;        // QK_K/8 = 32
        const uint8_t* scales= x + 2 + 64 + 8 + 32;   // QK_K/64 = 4
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            float db1 = d * (1 + 2*(scales[ib32/2] & 0xf));
            float db2 = d * (1 + 2*(scales[ib32/2] >>  4));
            for (int l = 0; l < 4; l++) {
                const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j+0] = db1 * g1[j] * ((signs[l] & kmask_iq2xs[j+0]) ? -1.f : 1.f);
                    y[j+4] = db1 * g2[j] * ((signs[l] & kmask_iq2xs[j+4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; l++) {
                const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j+0] = db2 * g1[j] * ((signs[l] & kmask_iq2xs[j+0]) ? -1.f : 1.f);
                    y[j+4] = db2 * g2[j] * ((signs[l] & kmask_iq2xs[j+4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

static void dq_iq4_xs(const uint8_t* data, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_IQ4_XS;
        float d = fp16_val(*(const uint16_t*)(x + 0));
        uint16_t scales_h = *(const uint16_t*)(x + 2);
        const uint8_t* scales_l = x + 4;             // QK_K/64 = 4
        const uint8_t* qs = x + 4 + 4;               // QK_K/2 = 128
        for (int ib = 0; ib < QK_K/32; ib++) {
            int ls = ((scales_l[ib/2] >> (4*(ib%2))) & 0xf) | (((scales_h >> (2*ib)) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y += 32;
            qs += 16;
        }
    }
}

static void dq_iq4_nl(const uint8_t* data, float* y, int64_t k) {
    // 32-element blocks (not QK_K=256). Block = fp16 d + 16 packed nibbles.
    int64_t nb = k / 32;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* x = data + i * BSZ_IQ4_NL;
        float d = fp16_val(*(const uint16_t*)(x + 0));
        const uint8_t* qs = x + 2;
        for (int j = 0; j < 16; j++) {
            y[j+ 0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+16] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y += 32;
    }
}

// ─── Universal dequantize ───────────────────────────────────────────────────

static float fp16_val(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

static float bf16_val(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float r; memcpy(&r, &f, 4); return r;
}

void dequant_tensor(const void* raw_data, float* out, uint32_t N, uint32_t K, GGUFType type) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(raw_data);

    if (type == GGUF_TYPE_F32) {
        memcpy(out, data, (size_t)N * K * 4);
        return;
    }
    if (type == GGUF_TYPE_F16) {
        const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
        for (size_t i = 0; i < (size_t)N * K; i++) out[i] = fp16_val(fp16[i]);
        return;
    }
    if (type == GGUF_TYPE_BF16) {
        const uint16_t* bf = reinterpret_cast<const uint16_t*>(data);
        for (size_t i = 0; i < (size_t)N * K; i++) out[i] = bf16_val(bf[i]);
        return;
    }
    if (type == GGUF_TYPE_Q8_0) {
        uint32_t nBlocks = K / 32;
        struct Q8B { uint16_t d; int8_t qs[32]; };
        const Q8B* blocks = reinterpret_cast<const Q8B*>(data);
        for (uint32_t r = 0; r < N; r++)
            for (uint32_t b = 0; b < nBlocks; b++) {
                float scale = fp16_val(blocks[r * nBlocks + b].d);
                for (int q = 0; q < 32; q++)
                    out[r * K + b * 32 + q] = scale * (float)blocks[r * nBlocks + b].qs[q];
            }
        return;
    }
    if (type == GGUF_TYPE_Q4_0) {
        uint32_t nBlocks = K / 32;
        const size_t blockSize = 18;
        for (uint32_t r = 0; r < N; r++)
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = data + ((size_t)r * nBlocks + b) * blockSize;
                float d = fp16_val(*(const uint16_t*)blk);
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = blk[2 + j];
                    out[r * K + b * 32 + j]      = d * ((float)(int)(byte & 0xF) - 8.0f);
                    out[r * K + b * 32 + j + 16]  = d * ((float)(int)(byte >> 4) - 8.0f);
                }
            }
        return;
    }
    if (type == GGUF_TYPE_Q4_1) {
        uint32_t nBlocks = K / 32;
        const size_t blockSize = 20;
        for (uint32_t r = 0; r < N; r++)
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = data + ((size_t)r * nBlocks + b) * blockSize;
                float d = fp16_val(*(const uint16_t*)blk);
                float m = fp16_val(*(const uint16_t*)(blk + 2));
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = blk[4 + j];
                    out[r * K + b * 32 + j]      = d * (float)(byte & 0xF) + m;
                    out[r * K + b * 32 + j + 16]  = d * (float)(byte >> 4) + m;
                }
            }
        return;
    }
    if (type == GGUF_TYPE_Q5_0) {
        uint32_t nBlocks = K / 32;
        const size_t blockSize = 22;
        for (uint32_t r = 0; r < N; r++)
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = data + ((size_t)r * nBlocks + b) * blockSize;
                float d = fp16_val(*(const uint16_t*)blk);
                uint32_t qh; memcpy(&qh, blk + 2, 4);
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = blk[6 + j];
                    int lo = (byte & 0xF) | (((qh >> j) & 1) << 4);
                    int hi = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4);
                    out[r * K + b * 32 + j]      = d * ((float)lo - 16.0f);
                    out[r * K + b * 32 + j + 16]  = d * ((float)hi - 16.0f);
                }
            }
        return;
    }
    if (type == GGUF_TYPE_Q5_1) {
        uint32_t nBlocks = K / 32;
        const size_t blockSize = 24;
        for (uint32_t r = 0; r < N; r++)
            for (uint32_t b = 0; b < nBlocks; b++) {
                const uint8_t* blk = data + ((size_t)r * nBlocks + b) * blockSize;
                float d = fp16_val(*(const uint16_t*)blk);
                float m = fp16_val(*(const uint16_t*)(blk + 2));
                uint32_t qh; memcpy(&qh, blk + 4, 4);
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = blk[8 + j];
                    int lo = (byte & 0xF) | (((qh >> j) & 1) << 4);
                    int hi = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4);
                    out[r * K + b * 32 + j]      = d * (float)lo + m;
                    out[r * K + b * 32 + j + 16]  = d * (float)hi + m;
                }
            }
        return;
    }
    if (type == GGUF_TYPE_Q4_K || type == GGUF_TYPE_Q5_K || type == GGUF_TYPE_Q6_K) {
        dequant_kquant(raw_data, out, N, K, type);
        return;
    }
    if (type == GGUF_TYPE_Q2_K)   { dq_q2_K   (data, out, (int64_t)N * K); return; }
    if (type == GGUF_TYPE_Q3_K)   { dq_q3_K   (data, out, (int64_t)N * K); return; }
    if (type == GGUF_TYPE_IQ2_S)  { dq_iq2_s  (data, out, (int64_t)N * K); return; }
    if (type == GGUF_TYPE_IQ3_S)  { dq_iq3_s  (data, out, (int64_t)N * K); return; }
    if (type == GGUF_TYPE_IQ3_XXS){ dq_iq3_xxs(data, out, (int64_t)N * K); return; }
    if (type == GGUF_TYPE_IQ4_XS) { dq_iq4_xs (data, out, (int64_t)N * K); return; }
    if (type == GGUF_TYPE_IQ4_NL) { dq_iq4_nl (data, out, (int64_t)N * K); return; }
    fprintf(stderr, "dequant_tensor: unsupported type %d\n", (int)type);
}