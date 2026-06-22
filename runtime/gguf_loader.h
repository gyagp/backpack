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
    GGUF_TYPE_IQ2_XS = 17, GGUF_TYPE_IQ3_XXS = 18,
    GGUF_TYPE_IQ1_S = 19, GGUF_TYPE_IQ4_NL = 20,
    GGUF_TYPE_IQ3_S = 21, GGUF_TYPE_IQ2_S = 22,
    GGUF_TYPE_IQ4_XS = 23,
    GGUF_TYPE_BF16 = 30,
};

using GGUFMetaValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, float,
    bool, std::string,
    uint64_t, int64_t, double,
    std::vector<std::string>,
    std::vector<int32_t>
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

// ─── K-quant raw block data ─────────────────────────────────────────────────
// Q4_K/Q5_K/Q6_K weights are uploaded as raw u32 words (no repacking needed).
// The WGSL kernels read block data in-place.

struct KQuantPacked {
    std::vector<uint32_t> data;      // raw block data as u32 words
    uint32_t N, K;                   // output rows, input cols
    uint32_t rowStrideWords;         // words per row (padded to alignment)
    uint32_t nBlocks;                // blocks per row
};

/// Pack Q4_K raw data (144-byte blocks) for GPU upload.
/// raw_data: pointer to N rows of Q4_K blocks; K = elements per row.
KQuantPacked pack_q4k(const void* raw_data, uint32_t N, uint32_t K);

/// Pack Q5_K raw data (176-byte blocks) for GPU upload.
KQuantPacked pack_q5k(const void* raw_data, uint32_t N, uint32_t K);

/// Pack Q6_K raw data (210-byte blocks) for GPU upload.
/// Pads row stride to 4-byte alignment since 210 is not word-aligned.
KQuantPacked pack_q6k(const void* raw_data, uint32_t N, uint32_t K);

/// Pack Q2_K (84 bytes / 256 elements) — already u32-aligned, 21 words/block.
KQuantPacked pack_q2k(const void* raw_data, uint32_t N, uint32_t K);

/// Pack Q3_K (110 bytes / 256 elements) — padded per-block to 112 bytes (28 words).
KQuantPacked pack_q3k(const void* raw_data, uint32_t N, uint32_t K);

/// Pack IQ2_S (82 bytes / 256 elements) — padded per-block to 84 bytes (21 words).
KQuantPacked pack_iq2s(const void* raw_data, uint32_t N, uint32_t K);

/// Pack IQ3_S (110 bytes / 256 elements) — padded per-block to 112 bytes (28 words).
KQuantPacked pack_iq3s(const void* raw_data, uint32_t N, uint32_t K);

/// Pack IQ4_XS (136 bytes / 256 elements) — already u32-aligned, 34 words/block.
KQuantPacked pack_iq4xs(const void* raw_data, uint32_t N, uint32_t K);

/// Accessors for IQ codebook arrays (used to upload to GPU).
const uint32_t* getIq3sGrid(uint32_t* out_count);  // returns array of 512 u32
const uint32_t* getIq2sGridU32(uint32_t* out_count);  // returns array of 2048 u32 (1024 u64 entries as u32 pairs)

/// Dequantize K-quant block data to fp32 for CPU-side use (e.g., embedding lookup).
/// Supports Q4_K, Q5_K, Q6_K. Returns N*K floats.
void dequant_kquant(const void* raw_data, float* out, uint32_t N, uint32_t K, GGUFType type);

/// Dequantize any supported quantized type to fp32.
/// Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16, F32.
void dequant_tensor(const void* raw_data, float* out, uint32_t N, uint32_t K, GGUFType type);

enum class ActivationType { SiLU, GELU };
enum class AttnLayerType : uint8_t { Global, SlidingWindow };

struct PerLayerConfig {
    uint32_t qDim = 0;            // Q output dim (nHead * headDim_layer)
    uint32_t kvDim = 0;           // KV dim (nKvHeads * headDim_layer)
    uint32_t headDim = 0;         // head dimension for this layer
    uint32_t intermediateSize = 0;
    int kvSourceLayer = -1;       // -1 = own KV cache, >=0 = reuse from this layer
};

struct ModelConfig {
    std::string arch;
    uint32_t nLayer = 0;
    uint32_t nHead = 0;
    uint32_t nKvHeads = 0;
    uint32_t nEmbd = 0;
    uint32_t intermediateSize = 0;
    uint32_t nVocab = 0;
    uint32_t headDim = 0;
    uint32_t kvHeadDim = 0;       // K/V per-head dim — may differ from headDim
                                  // for archs that decouple it (e.g. qwen35moe
                                  // sets attention.key_length=256 vs headDim=64).
                                  // Defaults to headDim if not specified.
    float rmsNormEps = 1e-6f;
    float ropeTheta = 1e6f;
    bool tieWordEmbeddings = true;
    bool hasQkNorm = false;

    // Gemma family extensions
    ActivationType activation = ActivationType::SiLU;
    float embeddingScale = 0.0f;
    float logitSoftcap = 0.0f;
    uint32_t slidingWindow = 0;
    std::vector<AttnLayerType> layerAttnTypes;

    // Per-layer dimensions (populated during weight loading for variable-dim models)
    std::vector<PerLayerConfig> perLayer;
    bool hasPerLayerDims = false;  // true if perLayer is populated

    // Sandwich norms (Gemma 4)
    bool hasSandwichNorm = false;

    // Per-Layer Embedding (Gemma 4)
    uint32_t pleSize = 0;         // 0 = disabled, >0 = PLE embedding dim
    uint32_t sharedKvLayers = 0;  // number of layers sharing KV from earlier layers

    // MTP (Multi-Token Prediction) head
    bool hasMtp = false;
    uint32_t mtpNumLayers = 0;

    // Mixture-of-Experts (qwen35moe, qwen3moe, deepseek, mixtral, …)
    uint32_t numExperts = 0;           // total experts (e.g. 128)
    uint32_t numExpertsPerTok = 0;     // active per token (e.g. 8)
    uint32_t moeIntermediateSize = 0;  // per-expert FFN dim (routed experts)
    uint32_t moeSharedIntermediateSize = 0;  // shared expert FFN dim (may equal moeIntermediateSize)

    // State-space model (Mamba) — used by hybrid SSM+attention archs (qwen35moe).
    // When ssmInnerSize > 0, some/all layers use SSM instead of attention.
    uint32_t ssmInnerSize = 0;          // d_inner — internal SSM state dim
    uint32_t ssmStateSize = 0;          // d_state — SSM state vector size
    uint32_t ssmConvKernel = 0;         // conv1d kernel size (typically 4)
    uint32_t ssmGroupCount = 0;         // number of SSM groups
    uint32_t ssmTimeStepRank = 0;       // dt projection rank
    uint32_t fullAttentionInterval = 0; // every N-th layer is attention, others SSM (0 = all attention)

    /// For hybrid SSM+attention archs (qwen35moe): returns true if layer `i`
    /// uses the attention path, false if it's an SSM-only layer.
    /// With fullAttentionInterval=4: layers 3, 7, 11, 15, … are attention,
    /// all others are SSM. With interval=0: all layers are attention.
    bool isAttentionLayer(uint32_t i) const {
        if (fullAttentionInterval == 0) return true;
        return ((i + 1) % fullAttentionInterval) == 0;
    }
};

ModelConfig extractModelConfig(const GGUFFile& gguf);
