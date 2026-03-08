#include "gguf_loader.h"
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
