#pragma once
/**
 * gguf_loader.h — GGUF parser with metadata extraction.
 *
 * Parses both metadata key-value pairs and tensor info from GGUF files.
 * Metadata is used to extract model architecture and hyperparameters,
 * eliminating the need for a separate manifest.json.
 *
 * Follows the llama.cpp GGUF naming conventions:
 *   general.architecture       → model type (e.g. "qwen3", "llama")
 *   {arch}.block_count         → number of layers
 *   {arch}.embedding_length    → hidden size
 *   {arch}.attention.head_count → number of attention heads
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

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

using GGUFMetaValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, float,
    bool, std::string,
    uint64_t, int64_t, double,
    std::vector<std::string>
>;

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    GGUFType type;
    uint64_t offset;
};

struct GGUFFile {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t data_offset = 0;

    std::unordered_map<std::string, GGUFMetaValue> metadata;
    std::vector<GGUFTensorInfo> tensors;
    std::unordered_map<std::string, size_t> tensor_index;

    bool open(const std::string& path);

    std::string getString(const std::string& key, const std::string& def = "") const;
    uint32_t    getU32   (const std::string& key, uint32_t def = 0) const;
    float       getFloat (const std::string& key, float def = 0.0f) const;
    bool        getBool  (const std::string& key, bool def = false) const;
    bool        hasKey   (const std::string& key) const;

    const std::vector<std::string>* getStringArray(const std::string& key) const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return nullptr;
        return std::get_if<std::vector<std::string>>(&it->second);
    }
};

struct Q8_0Block {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

struct Q8Repacked {
    std::vector<uint32_t> weights;
    std::vector<uint32_t> scales;
    uint32_t N, K;
};

Q8Repacked repack_q8_0(const void* raw_data, uint32_t N, uint32_t K);

struct ModelConfig {
    std::string arch;
    uint32_t nLayer = 0;
    uint32_t nHead = 0;
    uint32_t nKvHeads = 0;
    uint32_t nEmbd = 0;
    uint32_t intermediateSize = 0;
    uint32_t nVocab = 0;
    uint32_t headDim = 0;
    float rmsNormEps = 1e-6f;
    float ropeTheta = 1e6f;
    bool tieWordEmbeddings = true;
    bool hasQkNorm = false;
};

ModelConfig extractModelConfig(const GGUFFile& gguf);
