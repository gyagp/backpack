#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

/// Minimal GGUF parser — loads Q8_0 tensors from a GGUF file.
/// Matches the Python GGUFFile fast parser in gguf_utils.py.

enum GGUFType : uint32_t {
    GGUF_TYPE_F32 = 0, GGUF_TYPE_F16 = 1,
    GGUF_TYPE_Q4_0 = 2, GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q5_0 = 6, GGUF_TYPE_Q5_1 = 7,
    GGUF_TYPE_Q8_0 = 8, GGUF_TYPE_Q8_1 = 9,
    GGUF_TYPE_Q2_K = 10, GGUF_TYPE_Q3_K = 11,
    GGUF_TYPE_Q4_K = 12, GGUF_TYPE_Q5_K = 13,
    GGUF_TYPE_Q6_K = 14, GGUF_TYPE_IQ2_XXS = 16,
    GGUF_TYPE_BF16 = 30,
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;  // [dim0, dim1, ...]
    GGUFType type;
    uint64_t offset;  // offset from data start in file
};

struct GGUFFile {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t data_offset;
    std::vector<GGUFTensorInfo> tensors;
    std::unordered_map<std::string, size_t> tensor_index;  // name → index

    bool open(const std::string& path);
};

/// Q8_0 block: 32 int8 quantized values + 1 fp16 scale
struct Q8_0Block {
    uint16_t d;      // fp16 scale
    int8_t qs[32];   // quantized values
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

/// Repack Q8_0 blocks into the format expected by the WGSL kernels:
/// - weights: (N, K/4) uint32 packed int8
/// - scales: (N, K/32) float16 packed in uint32 pairs
struct Q8Repacked {
    std::vector<uint32_t> weights;  // (N, K/4)
    std::vector<uint32_t> scales;   // ceil(N * K/32 / 2) u32 pairs
    uint32_t N, K;
};

Q8Repacked repack_q8_0(const void* raw_data, uint32_t N, uint32_t K);
