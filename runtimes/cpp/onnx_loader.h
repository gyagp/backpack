#pragma once
/**
 * onnx_loader.h — ONNX model loader for the C++ runtime.
 *
 * Loads ONNX GenAI models (like phi-4-mini) by parsing the protobuf format
 * directly (no external onnx/protobuf dependency). Extracts model config
 * from genai_config.json, weights from model.onnx + model.onnx.data,
 * and pre-computed RoPE tables from ONNX initializers.
 *
 * Weights are dequantized from Q4/Q8 and repacked into Q8Repacked format
 * compatible with the existing GPU kernels.
 */

#include "gguf_loader.h"   // ModelConfig, Q8Repacked, repack_q8_0
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

/// Result of loading an ONNX model — all data needed to build the GPU pipeline.
struct OnnxLoadResult {
    ModelConfig cfg;

    /// Partial RoPE: rotary_dim may be < head_dim (e.g. Phi-4).
    /// If 0, full RoPE is used (rotary_dim == head_dim).
    uint32_t rotaryDim = 0;

    /// Per-layer weights — either Q8 repacked or fp16.
    struct LayerData {
        Q8Repacked qkv;         // fused Q/K/V projection (Q8 format)
        Q8Repacked o;           // output projection
        Q8Repacked gateup;      // fused gate + up projection
        Q8Repacked down;        // down projection
        // fp16 alternative (used when Q4 source — avoids double-quantization)
        std::vector<uint16_t> qkvFp16;
        std::vector<uint16_t> oFp16;
        std::vector<uint16_t> gateupFp16;
        std::vector<uint16_t> downFp16;
        uint32_t qkvN = 0, qkvK = 0;
        uint32_t oN = 0, oK = 0;
        uint32_t gateupN = 0, gateupK = 0;
        uint32_t downN = 0, downK = 0;
        bool useFp16 = false;   // true = use fp16 weights, false = use Q8
        std::vector<float> inputNorm;       // input layer norm weights
        std::vector<float> postAttnNorm;    // post-attention layer norm weights
        std::vector<float> qNorm;           // Q-norm weights (empty if not present)
        std::vector<float> kNorm;           // K-norm weights (empty if not present)
    };
    std::vector<LayerData> layers;

    /// Embedding table (fp32 for CPU lookup).
    std::vector<float> embeddingCPU;

    /// LM head weights.
    Q8Repacked lmHeadQ8;
    bool hasLmHeadQ8 = false;           // true if we have separate Q8 LM head
    bool tieWordEmbeddings = true;

    /// Final layer norm weights.
    std::vector<float> finalNorm;

    /// Pre-computed RoPE tables from ONNX cos_cache/sin_cache.
    /// Shape: [maxPositions × halfDim] stored row-major as flat arrays.
    std::vector<float> ropeCos;
    std::vector<float> ropeSin;
    bool hasPrecomputedRope = false;
    uint32_t ropeMaxPositions = 0;      // number of rows in cos/sin tables
    uint32_t ropeHalfDim = 0;           // number of columns (half of rotary_dim)
};

/// Load an ONNX model from a directory containing model.onnx + genai_config.json.
/// Populates result with all weights, config, and RoPE tables.
bool loadOnnxModel(const std::string& modelDir, OnnxLoadResult& result);
