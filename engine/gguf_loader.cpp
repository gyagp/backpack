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

static std::string read_string(FILE* f) {
    uint64_t len = read_u64(f);
    std::string s(len, '\0');
    fread(s.data(), 1, len, f);
    return s;
}

static void skip_meta_value(FILE* f, uint32_t type) {
    switch (type) {
        case META_UINT8: case META_INT8: case META_BOOL: fseek(f, 1, SEEK_CUR); break;
        case META_UINT16: case META_INT16: fseek(f, 2, SEEK_CUR); break;
        case META_UINT32: case META_INT32: case META_FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case META_UINT64: case META_INT64: case META_FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case META_STRING: { uint64_t len = read_u64(f); fseek(f, (long)len, SEEK_CUR); break; }
        case META_ARRAY: {
            uint32_t elem_type = read_u32(f);
            uint64_t count = read_u64(f);
            for (uint64_t i = 0; i < count; i++) skip_meta_value(f, elem_type);
            break;
        }
    }
}

bool GGUFFile::open(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Header: magic (4) + version (4) + n_tensors (8) + n_kv (8)
    uint32_t magic = read_u32(f);
    if (magic != 0x46554747) { fclose(f); return false; }  // "GGUF"

    version = read_u32(f);
    n_tensors = read_u64(f);
    uint64_t n_kv = read_u64(f);

    // Skip metadata key-value pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        read_string(f);  // key
        uint32_t vtype = read_u32(f);
        skip_meta_value(f, vtype);
    }

    // Read tensor infos
    tensors.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        auto& t = tensors[i];
        t.name = read_string(f);
        uint32_t n_dims = read_u32(f);
        t.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            t.shape[d] = read_u64(f);
        }
        t.type = static_cast<GGUFType>(read_u32(f));
        t.offset = read_u64(f);
        tensor_index[t.name] = i;
    }

    // Data starts at aligned position after header
    long pos = ftell(f);
    data_offset = (pos + 31) & ~31ULL;  // align to 32 bytes

    fclose(f);
    return true;
}

Q8Repacked repack_q8_0(const void* raw_data, uint32_t N, uint32_t K) {
    Q8Repacked result;
    result.N = N;
    result.K = K;

    uint32_t n_blocks_per_row = K / 32;
    const auto* blocks = reinterpret_cast<const Q8_0Block*>(raw_data);

    // Repack weights: (N, K) int8 → (N, K/4) uint32
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

    // Repack scales: fp16 per block → (N, n_blocks) fp16, packed 2 per u32
    uint32_t total_blocks = N * n_blocks_per_row;
    result.scales.resize((total_blocks + 1) / 2);
    for (uint32_t i = 0; i < total_blocks; i++) {
        uint16_t d = blocks[i].d;
        if (i % 2 == 0) {
            result.scales[i / 2] = d;
        } else {
            result.scales[i / 2] |= (uint32_t)d << 16;
        }
    }

    return result;
}
