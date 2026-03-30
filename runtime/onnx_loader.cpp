#include "onnx_loader.h"
#include "json_parser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// ─── fp16 conversion ─────────────────────────────────────────────────────────

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

// ─── Minimal ONNX protobuf reader ───────────────────────────────────────────
//
// ONNX uses protobuf wire format. We only need to parse specific structures:
//   ModelProto -> GraphProto -> NodeProto, TensorProto
//
// Protobuf wire types:
//   0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit
//
// ONNX field numbers (from onnx.proto3):
//   ModelProto:  7 = graph (GraphProto)
//   GraphProto:  1 = node (NodeProto), 5 = initializer (TensorProto)
//   NodeProto:   1 = input, 2 = output, 3 = name, 4 = op_type, 5 = attribute
//   TensorProto: 1 = dims, 2 = data_type, 3 = segment(unused), 4 = float_data,
//                5 = int32_data, 6 = string_data, 7 = int64_data,
//                8 = name, 9 = doc_string, 13 = raw_data,
//                14 = external_data (StringStringEntryProto)
//   AttributeProto: 1 = name, 2 = ref_attr_name, 3 = doc_string,
//                   4 = type, 5 = f, 6 = i, 7 = s
//   StringStringEntryProto: 1 = key, 2 = value

namespace {

struct PBReader {
    const uint8_t* data;
    const uint8_t* end;

    PBReader(const uint8_t* d, size_t len) : data(d), end(d + len) {}
    bool eof() const { return data >= end; }
    size_t remaining() const { return (size_t)(end - data); }

    uint64_t readVarint() {
        uint64_t v = 0;
        int shift = 0;
        while (data < end) {
            uint8_t b = *data++;
            v |= (uint64_t)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return v;
    }

    uint32_t readFixed32() {
        uint32_t v;
        memcpy(&v, data, 4);
        data += 4;
        return v;
    }

    uint64_t readFixed64() {
        uint64_t v;
        memcpy(&v, data, 8);
        data += 8;
        return v;
    }

    PBReader readLengthDelimited() {
        uint64_t len = readVarint();
        PBReader sub(data, (size_t)len);
        data += len;
        return sub;
    }

    std::string readString() {
        uint64_t len = readVarint();
        std::string s((const char*)data, (size_t)len);
        data += len;
        return s;
    }

    void skip(int wireType) {
        switch (wireType) {
            case 0: readVarint(); break;
            case 1: data += 8; break;
            case 2: { uint64_t len = readVarint(); data += len; break; }
            case 5: data += 4; break;
            default: break;
        }
    }

    // Read field tag: returns (field_number, wire_type)
    std::pair<uint32_t, int> readTag() {
        uint64_t tag = readVarint();
        return {(uint32_t)(tag >> 3), (int)(tag & 7)};
    }
};

// ─── ONNX structures ────────────────────────────────────────────────────────

// ONNX data types (TensorProto.DataType)
enum OnnxDataType : int {
    ONNX_FLOAT = 1,
    ONNX_UINT8 = 2,
    ONNX_INT8 = 3,
    ONNX_UINT16 = 4,
    ONNX_INT16 = 5,
    ONNX_INT32 = 6,
    ONNX_INT64 = 7,
    ONNX_FLOAT16 = 10,
    ONNX_DOUBLE = 11,
    ONNX_UINT32 = 12,
    ONNX_UINT64 = 13,
};

struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int dataType = 0;
    // Raw data: either inline (raw_data in proto) or external (from .data file)
    const uint8_t* rawData = nullptr;
    size_t rawSize = 0;
    // For external data
    std::string externalLocation;
    int64_t externalOffset = -1;
    int64_t externalLength = -1;
    // Inline float data (if stored as repeated float)
    std::vector<float> floatData;
    // Inline int64 data (if stored as repeated int64)
    std::vector<int64_t> int64Data;
    // Inline int32 data (if stored as repeated int32)
    std::vector<int32_t> int32Data;
};

struct OnnxAttribute {
    std::string name;
    int type = 0;  // 1=float, 2=int, 3=string, ...
    float f = 0;
    int64_t i = 0;
    std::string s;
    // For graph attributes (type=5 = GRAPH): raw subgraph data for deferred parsing
    const uint8_t* graphData = nullptr;
    size_t graphSize = 0;
};

struct OnnxNode {
    std::string opType;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<OnnxAttribute> attributes;

    int64_t getAttrInt(const std::string& attrName, int64_t def = 0) const {
        for (auto& a : attributes)
            if (a.name == attrName) return a.i;
        return def;
    }
};

// Parse a TensorProto message
OnnxTensor parseTensorProto(PBReader& r) {
    OnnxTensor t;
    int dataLocation = 0;  // 0=DEFAULT (inline), 1=EXTERNAL
    while (!r.eof()) {
        auto [field, wire] = r.readTag();
        switch (field) {
            case 1: // dims (repeated int64, packed or individual)
                if (wire == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) t.dims.push_back((int64_t)sub.readVarint());
                } else {
                    t.dims.push_back((int64_t)r.readVarint());
                }
                break;
            case 2: t.dataType = (int)r.readVarint(); break;  // data_type
            case 4: // float_data (packed)
                if (wire == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) {
                        float v;
                        uint32_t bits = sub.readFixed32();
                        memcpy(&v, &bits, 4);
                        t.floatData.push_back(v);
                    }
                } else {
                    float v;
                    uint32_t bits = r.readFixed32();
                    memcpy(&v, &bits, 4);
                    t.floatData.push_back(v);
                }
                break;
            case 8: t.name = r.readString(); break;  // name
            case 5: // int32_data (repeated int32, packed or individual)
                if (wire == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) t.int32Data.push_back((int32_t)sub.readVarint());
                } else {
                    t.int32Data.push_back((int32_t)r.readVarint());
                }
                break;
            case 7: // int64_data (repeated int64, packed or individual)
                if (wire == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) t.int64Data.push_back((int64_t)sub.readVarint());
                } else {
                    t.int64Data.push_back((int64_t)r.readVarint());
                }
                break;
            case 15: dataLocation = (int)r.readVarint(); break;  // data_location (compat)
            case 9: { // raw_data (field 9 per ONNX spec)
                uint64_t len = r.readVarint();
                const uint8_t* rawStart = r.data;
                t.rawData = rawStart;
                t.rawSize = (size_t)len;
                r.data += len;
                break;
            }
            case 13: { // external_data (StringStringEntryProto, repeated)
                uint64_t len = r.readVarint();
                const uint8_t* rawStart = r.data;

                // Check if this is actually a StringStringEntryProto
                // (ONNX GenAI WebGPU models encode external_data as field 13)
                // Format: 0x0A <len> <key_string> 0x12 <len> <value_string>
                if (len > 4 && len < 200 && rawStart[0] == 0x0A) {
                    // Try parsing as StringStringEntryProto
                    PBReader sub(rawStart, (size_t)len);
                    std::string key, value;
                    bool isStringEntry = true;
                    while (!sub.eof()) {
                        auto [f2, w2] = sub.readTag();
                        if (f2 == 1 && w2 == 2) key = sub.readString();
                        else if (f2 == 2 && w2 == 2) value = sub.readString();
                        else { isStringEntry = false; break; }
                    }
                    if (isStringEntry && !key.empty()) {
                        if (key == "location") t.externalLocation = value;
                        else if (key == "offset") t.externalOffset = std::stoll(value);
                        else if (key == "length") t.externalLength = std::stoll(value);
                        r.data += len;
                        break;
                    }
                }

                // Normal raw_data
                t.rawData = rawStart;
                t.rawSize = (size_t)len;
                r.data += len;
                break;
            }
            case 14: { // external_data (StringStringEntryProto, repeated) or data_location
                if (wire == 0) {
                    // Some exporters encode data_location here as varint
                    dataLocation = (int)r.readVarint();
                } else {
                    auto sub = r.readLengthDelimited();
                    std::string key, value;
                    while (!sub.eof()) {
                        auto [f2, w2] = sub.readTag();
                        if (f2 == 1) key = sub.readString();
                        else if (f2 == 2) value = sub.readString();
                        else sub.skip(w2);
                    }
                    if (key == "location") t.externalLocation = value;
                    else if (key == "offset") t.externalOffset = std::stoll(value);
                    else if (key == "length") t.externalLength = std::stoll(value);
                }
                break;
            }
            default: r.skip(wire); break;
        }
    }
    // If data_location is EXTERNAL, clear any inline raw_data
    // and set default external location if not specified
    if (dataLocation == 1) {
        t.rawData = nullptr;
        t.rawSize = 0;
        if (t.externalLocation.empty())
            t.externalLocation = "model.onnx.data";  // default ONNX external data file
    }
    return t;
}

// Parse an AttributeProto message
OnnxAttribute parseAttributeProto(PBReader& r) {
    OnnxAttribute a;
    while (!r.eof()) {
        auto [field, wire] = r.readTag();
        switch (field) {
            case 1: a.name = r.readString(); break;
            case 2: {  // f (float)
                uint32_t bits = r.readFixed32();
                memcpy(&a.f, &bits, 4);
                break;
            }
            case 3: a.i = (int64_t)r.readVarint(); break;  // i (int64)
            case 4: a.s = r.readString(); break;  // s (bytes/string)
            case 6: {  // g (GraphProto) - store raw bytes for deferred parsing
                uint64_t len = r.readVarint();
                a.graphData = r.data;
                a.graphSize = (size_t)len;
                r.data += len;
                break;
            }
            case 20: a.type = (int)r.readVarint(); break;  // type
            default: r.skip(wire); break;
        }
    }
    return a;
}

// Parse a NodeProto message
OnnxNode parseNodeProto(PBReader& r) {
    OnnxNode n;
    while (!r.eof()) {
        auto [field, wire] = r.readTag();
        switch (field) {
            case 1: n.inputs.push_back(r.readString()); break;
            case 2: n.outputs.push_back(r.readString()); break;
            case 3: n.name = r.readString(); break;
            case 4: n.opType = r.readString(); break;
            case 5: {
                auto sub = r.readLengthDelimited();
                n.attributes.push_back(parseAttributeProto(sub));
                break;
            }
            default: r.skip(wire); break;
        }
    }
    return n;
}

// ─── ONNX weight name mapping (mirrors Python _onnx_name_to_backpack) ─────

struct WeightMapping {
    std::string backpackName;
    int layerIdx;  // -1 for non-layer weights
};

WeightMapping mapOnnxName(const std::string& onnxName) {
    // Strip common prefixes/suffixes
    std::string name = onnxName;
    auto stripAll = [&](const std::string& pat) {
        size_t pos;
        while ((pos = name.find(pat)) != std::string::npos)
            name.erase(pos, pat.size());
    };
    stripAll("MatMul.");
    stripAll(".weight_Q4G32");
    stripAll(".weight_Q8G32");
    stripAll(".weight_scale");

    // Remove "model." prefix
    if (name.substr(0, 6) == "model.") name = name.substr(6);

    // Map names
    struct NameEntry { std::string onnxSuffix; std::string backpackSuffix; };
    static const NameEntry nameMap[] = {
        {"attn.qkv_proj", "self_attn.qkv_proj.weight"},
        {"attn.q_proj",   "self_attn.q_proj.weight"},
        {"attn.k_proj",   "self_attn.k_proj.weight"},
        {"attn.v_proj",   "self_attn.v_proj.weight"},
        {"attn.o_proj",   "self_attn.o_proj.weight"},
        {"mlp.gate_up_proj", "mlp.gate_up_proj.weight"},
        {"mlp.gate_proj", "mlp.gate_proj.weight"},
        {"mlp.up_proj",   "mlp.up_proj.weight"},
        {"mlp.down_proj", "mlp.down_proj.weight"},
        {"lm_head",       "lm_head.weight"},
    };

    for (auto& entry : nameMap) {
        if (name.find(entry.onnxSuffix) != std::string::npos) {
            // Extract layer number
            int layerIdx = -1;
            // Look for "layers.N." or just ".N."
            size_t dotPos = 0;
            while ((dotPos = name.find('.', dotPos)) != std::string::npos) {
                dotPos++;
                if (dotPos < name.size() && isdigit(name[dotPos])) {
                    size_t numEnd = dotPos;
                    while (numEnd < name.size() && isdigit(name[numEnd])) numEnd++;
                    layerIdx = std::stoi(name.substr(dotPos, numEnd - dotPos));
                    break;
                }
            }
            if (layerIdx >= 0)
                return {"layers." + std::to_string(layerIdx) + "." + entry.backpackSuffix, layerIdx};
            if (entry.onnxSuffix == "lm_head")
                return {entry.backpackSuffix, -1};
        }
    }

    return {"", -1};  // unmapped
}

// Map ONNX norm weight name to backpack convention
std::string mapOnnxNormName(const std::string& onnxName) {
    std::string name = onnxName;
    if (name.substr(0, 6) == "model.") name = name.substr(6);

    if (name.find("final_norm") != std::string::npos)
        return "norm.weight";

    if (name.find("input_layernorm") != std::string::npos) {
        // Extract layer number
        for (size_t i = 0; i < name.size(); i++) {
            if (name.substr(i, 7) == "layers." && i + 7 < name.size() && isdigit(name[i + 7])) {
                size_t numStart = i + 7;
                size_t numEnd = numStart;
                while (numEnd < name.size() && isdigit(name[numEnd])) numEnd++;
                return "layers." + name.substr(numStart, numEnd - numStart) + ".input_layernorm.weight";
            }
        }
    }

    if (name.find("post_attention_layernorm") != std::string::npos) {
        for (size_t i = 0; i < name.size(); i++) {
            if (name.substr(i, 7) == "layers." && i + 7 < name.size() && isdigit(name[i + 7])) {
                size_t numStart = i + 7;
                size_t numEnd = numStart;
                while (numEnd < name.size() && isdigit(name[numEnd])) numEnd++;
                return "layers." + name.substr(numStart, numEnd - numStart) + ".post_attention_layernorm.weight";
            }
        }
    }

    // QK-norm weights (Qwen-3 etc.)
    // model.layers.N.attn.q_norm.layernorm.weight → layers.N.q_norm.weight
    // model.layers.N.attn.k_norm.layernorm.weight → layers.N.k_norm.weight
    if (name.find("q_norm") != std::string::npos) {
        for (size_t i = 0; i < name.size(); i++) {
            if (name.substr(i, 7) == "layers." && i + 7 < name.size() && isdigit(name[i + 7])) {
                size_t numStart = i + 7;
                size_t numEnd = numStart;
                while (numEnd < name.size() && isdigit(name[numEnd])) numEnd++;
                return "layers." + name.substr(numStart, numEnd - numStart) + ".q_norm.weight";
            }
        }
    }

    if (name.find("k_norm") != std::string::npos) {
        for (size_t i = 0; i < name.size(); i++) {
            if (name.substr(i, 7) == "layers." && i + 7 < name.size() && isdigit(name[i + 7])) {
                size_t numStart = i + 7;
                size_t numEnd = numStart;
                while (numEnd < name.size() && isdigit(name[numEnd])) numEnd++;
                return "layers." + name.substr(numStart, numEnd - numStart) + ".k_norm.weight";
            }
        }
    }

    if (name.find("norm.weight") != std::string::npos && name.find("layers") == std::string::npos)
        return "norm.weight";

    return "";
}

// ─── Q4 dequantization ──────────────────────────────────────────────────────

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

/// Dequantize ONNX Q4 weights directly to fp16 (no double-quantization).
/// Returns [N × K] fp16 row-major weight matrix.
std::vector<uint16_t> dequantQ4ToFp16(const uint8_t* wData, const uint8_t* sData,
                                       uint32_t N, uint32_t K, uint32_t blockSize,
                                       uint32_t nGroups, uint32_t blockHalf) {
    std::vector<uint16_t> fp16(N * K);
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(sData);

    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t g = 0; g < nGroups; g++) {
            float scale = fp16_to_f32(scales[row * nGroups + g]);
            const uint8_t* blockData = wData + (row * nGroups + g) * blockHalf;
            for (uint32_t j = 0; j < blockHalf; j++) {
                uint8_t byte = blockData[j];
                int8_t low  = (int8_t)(byte & 0x0F) - 8;
                int8_t high = (int8_t)(byte >> 4) - 8;
                uint32_t col = g * blockSize + j * 2;
                if (col < K)     fp16[row * K + col] = f32_to_fp16((float)low * scale);
                if (col + 1 < K) fp16[row * K + col + 1] = f32_to_fp16((float)high * scale);
            }
        }
    }
    return fp16;
}

/// Repack ONNX Q4 weights directly into Q8_0 format (no dequant roundtrip).
/// Q4 and Q8_0 both use 32-element blocks with fp16 scale.
/// Q4 nibbles (-8..+7) fit directly into Q8_0 int8 values.
Q8Repacked dequantQ4ToQ8(const uint8_t* wData, const uint8_t* sData,
                          uint32_t N, uint32_t K, uint32_t blockSize,
                          uint32_t nGroups, uint32_t blockHalf) {
    // Q4G32 and Q8_0 share the same block_size=32, so we can directly pack
    // Q4 nibbles into Q8_0 blocks with the original Q4 scale.
    uint32_t nBlocksPerRow = nGroups;  // nGroups = K / blockSize
    size_t totalBlocks = (size_t)N * nBlocksPerRow;
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(sData);

    struct Q8Block { uint16_t d; int8_t qs[32]; };
    std::vector<Q8Block> blocks(totalBlocks);

    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t g = 0; g < nGroups; g++) {
            auto& b = blocks[row * nBlocksPerRow + g];
            b.d = scales[row * nGroups + g];  // keep original fp16 scale
            const uint8_t* blockData = wData + (row * nGroups + g) * blockHalf;
            for (uint32_t j = 0; j < blockHalf; j++) {
                uint8_t byte = blockData[j];
                b.qs[j * 2]     = (int8_t)(byte & 0x0F) - 8;  // -8..+7
                b.qs[j * 2 + 1] = (int8_t)(byte >> 4) - 8;
            }
        }
    }

    return repack_q8_0(blocks.data(), N, nBlocksPerRow * 32);
}

/// Dequantize ONNX Q8 weights to fp32, then repack as Q8_0.
/// ONNX Q8 layout: weights [N, n_groups, block_size] uint8 with zero_point=128
Q8Repacked dequantQ8ToQ8(const uint8_t* wData, const uint8_t* sData,
                          uint32_t N, uint32_t K, uint32_t blockSize,
                          uint32_t nGroups) {
    // Similar to Q4 but simpler: each byte is one value
    std::vector<float> fp32(N * K);
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(sData);

    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t g = 0; g < nGroups; g++) {
            float scale = fp16_to_f32(scales[row * nGroups + g]);
            const uint8_t* blockData = wData + (row * nGroups + g) * blockSize;
            for (uint32_t j = 0; j < blockSize; j++) {
                int16_t centered = (int16_t)blockData[j] - 128;
                uint32_t col = g * blockSize + j;
                if (col < K) fp32[row * K + col] = (float)centered * scale;
            }
        }
    }

    // Repack same as Q4 path
    uint32_t nBlocksPerRow = (K + 31) / 32;
    size_t totalBlocks = (size_t)N * nBlocksPerRow;
    struct Q8Block { uint16_t d; int8_t qs[32]; };
    std::vector<Q8Block> blocks(totalBlocks);

    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t blk = 0; blk < nBlocksPerRow; blk++) {
            auto& b = blocks[row * nBlocksPerRow + blk];
            float maxAbs = 0.0f;
            for (int q = 0; q < 32; q++) {
                uint32_t col = blk * 32 + q;
                float val = (col < K) ? fp32[row * K + col] : 0.0f;
                maxAbs = std::max(maxAbs, std::abs(val));
            }
            float scale = maxAbs / 127.0f;
            float invScale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

            uint32_t fb;
            memcpy(&fb, &scale, 4);
            uint32_t s16 = (fb >> 16) & 0x8000;
            int32_t e = ((fb >> 23) & 0xFF) - 112;
            uint32_t m = (fb >> 13) & 0x3FF;
            if (e <= 0) b.d = (uint16_t)s16;
            else if (e > 30) b.d = (uint16_t)(s16 | 0x7C00);
            else b.d = (uint16_t)(s16 | (e << 10) | m);

            for (int q = 0; q < 32; q++) {
                uint32_t col = blk * 32 + q;
                float val = (col < K) ? fp32[row * K + col] : 0.0f;
                int v = (int)roundf(val * invScale);
                v = std::max(-128, std::min(127, v));
                b.qs[q] = (int8_t)v;
            }
        }
    }

    return repack_q8_0(blocks.data(), N, nBlocksPerRow * 32);
}

/// Dequantize GatherBlockQuantized embedding to fp32.
/// Raw: [n_vocab, n_groups, block_size] uint8 with zero_point=128
/// Scales: [n_vocab, n_groups] fp16
std::vector<float> dequantEmbedding(const uint8_t* rawData, const uint8_t* scaleData,
                                     uint32_t nVocab, uint32_t nGroups,
                                     uint32_t blockSize, uint32_t K) {
    std::vector<float> fp32(nVocab * K);
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(scaleData);

    for (uint32_t row = 0; row < nVocab; row++) {
        for (uint32_t g = 0; g < nGroups; g++) {
            float scale = fp16_to_f32(scales[row * nGroups + g]);
            const uint8_t* blockData = rawData + (row * nGroups + g) * blockSize;
            for (uint32_t j = 0; j < blockSize; j++) {
                int16_t centered = (int16_t)blockData[j] - 128;
                uint32_t col = g * blockSize + j;
                if (col < K) fp32[row * K + col] = (float)centered * scale;
            }
        }
    }
    return fp32;
}

/// Extract Constant node tensor values from a GraphProto subgraph.
/// Used to find cos_cache/sin_cache embedded in If node then_branch.
/// Returns a map of output_name → OnnxTensor for Constant nodes with tensor data.
std::unordered_map<std::string, OnnxTensor> extractConstantsFromGraph(
    const uint8_t* graphData, size_t graphSize) {
    std::unordered_map<std::string, OnnxTensor> result;
    PBReader gr(graphData, graphSize);
    while (!gr.eof()) {
        auto [gField, gWire] = gr.readTag();
        if (gField == 1 && gWire == 2) {
            // node (NodeProto)
            auto nodeR = gr.readLengthDelimited();
            std::string opType;
            std::vector<std::string> outputs;
            OnnxTensor tensor;
            bool hasTensor = false;
            while (!nodeR.eof()) {
                auto [nf, nw] = nodeR.readTag();
                switch (nf) {
                    case 2: outputs.push_back(nodeR.readString()); break; // output
                    case 4: opType = nodeR.readString(); break; // op_type
                    case 5: { // attribute
                        auto attrR = nodeR.readLengthDelimited();
                        std::string attrName;
                        while (!attrR.eof()) {
                            auto [af, aw] = attrR.readTag();
                            switch (af) {
                                case 1: attrName = attrR.readString(); break;
                                case 5: { // t (TensorProto) - field 5 in AttributeProto
                                    auto tR = attrR.readLengthDelimited();
                                    tensor = parseTensorProto(tR);
                                    hasTensor = true;
                                    break;
                                }
                                default: attrR.skip(aw); break;
                            }
                        }
                        break;
                    }
                    default: nodeR.skip(nw); break;
                }
            }
            if (opType == "Constant" && hasTensor && !outputs.empty()) {
                result[outputs[0]] = std::move(tensor);
            }
        } else {
            gr.skip(gWire);
        }
    }
    return result;
}

}  // anonymous namespace

// ─── Public API ──────────────────────────────────────────────────────────────

bool loadOnnxModel(const std::string& modelDir, OnnxLoadResult& result) {
    auto t0 = std::chrono::steady_clock::now();

    // ─── 1. Parse model config ────────────────────────────────────────
    std::string configPath = (fs::path(modelDir) / "config.json").string();
    bool useGenaiFormat = false;
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) {
        fprintf(stderr, "Failed to open config: %s\n", configPath.c_str());
        return false;
    }
    std::string configStr((std::istreambuf_iterator<char>(configFile)),
                           std::istreambuf_iterator<char>());
    configFile.close();

    auto configJson = json_parse(configStr);

    auto& cfg = result.cfg;
    if (useGenaiFormat) {
        // GenAI format: model.decoder.num_attention_heads, etc.
        auto& model = configJson["model"];
        auto& dec = model["decoder"];
        cfg.nHead = dec["num_attention_heads"].as_uint();
        cfg.nKvHeads = dec.has("num_key_value_heads")
            ? dec["num_key_value_heads"].as_uint() : cfg.nHead;
        cfg.headDim = dec["head_size"].as_uint();
        cfg.nEmbd = dec["hidden_size"].as_uint();
        cfg.nLayer = dec["num_hidden_layers"].as_uint();
        cfg.nVocab = model["vocab_size"].as_uint();
        cfg.arch = model.has("type") ? model["type"].as_string() : "phi3";
    } else {
        // HuggingFace config.json format: flat keys
        cfg.nHead = configJson.has("num_attention_heads")
            ? configJson["num_attention_heads"].as_uint() : 32;
        cfg.nKvHeads = configJson.has("num_key_value_heads")
            ? configJson["num_key_value_heads"].as_uint() : cfg.nHead;
        cfg.nEmbd = configJson.has("hidden_size")
            ? configJson["hidden_size"].as_uint() : 2048;
        cfg.headDim = cfg.nEmbd / cfg.nHead;
        cfg.nLayer = configJson.has("num_hidden_layers")
            ? configJson["num_hidden_layers"].as_uint() : 24;
        cfg.nVocab = configJson.has("vocab_size")
            ? configJson["vocab_size"].as_uint() : 32000;
        cfg.arch = configJson.has("model_type")
            ? configJson["model_type"].as_string() : "llama";
    }
    cfg.rmsNormEps = 1e-5f;
    cfg.ropeTheta = 10000.0f;
    cfg.tieWordEmbeddings = true;
    cfg.hasQkNorm = false;
    cfg.intermediateSize = 0;  // inferred from weights

    printf("ONNX Config: %s (%u layers, E=%u, HD=%u, V=%u, KV=%u)\n",
           cfg.arch.c_str(), cfg.nLayer, cfg.nEmbd, cfg.headDim,
           cfg.nVocab, cfg.nKvHeads);

    // ─── 2. Load ONNX protobuf ──────────────────────────────────────────
    std::string onnxPath = (fs::path(modelDir) / "model.onnx").string();
    if (!fs::exists(onnxPath)) {
        // Find any .onnx file in the directory
        for (auto& e : fs::directory_iterator(modelDir)) {
            if (e.is_regular_file() && e.path().extension() == ".onnx") {
                onnxPath = e.path().string();
                break;
            }
        }
    }
    auto onnxFileSize = fs::file_size(onnxPath);
    FILE* f = fopen(onnxPath.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", onnxPath.c_str());
        return false;
    }
    std::vector<uint8_t> onnxData(onnxFileSize);
    fread(onnxData.data(), 1, onnxFileSize, f);
    fclose(f);

    printf("  ONNX file: %llu bytes\n", (unsigned long long)onnxFileSize);
    fflush(stdout);

    // Load external data file if present
    std::vector<uint8_t> externalData;
    std::string extDataPath = (fs::path(modelDir) / "model.onnx.data").string();
    if (fs::exists(extDataPath)) {
        // Use filesystem to get file size (handles > 2GB on Windows)
        auto extFileSize = fs::file_size(extDataPath);
        printf("  Loading external data: %llu bytes (%.0f MB)...\n",
               (unsigned long long)extFileSize, extFileSize / 1048576.0);
        fflush(stdout);
        FILE* ef = fopen(extDataPath.c_str(), "rb");
        if (ef) {
            externalData.resize(extFileSize);
            size_t totalRead = 0;
            while (totalRead < extFileSize) {
                size_t chunk = std::min((size_t)(256 * 1024 * 1024),
                                        (size_t)(extFileSize - totalRead));
                size_t n = fread(externalData.data() + totalRead, 1, chunk, ef);
                if (n == 0) break;
                totalRead += n;
            }
            fclose(ef);
            printf("  External data loaded: %zu bytes\n", totalRead);
            fflush(stdout);
        }
    }

    // ─── 3. Parse graph: extract initializers and nodes ──────────────────
    std::unordered_map<std::string, OnnxTensor> initializers;
    std::vector<OnnxNode> nodes;

    PBReader modelReader(onnxData.data(), onnxData.size());
    int64_t externalDataCursor = 0;  // running offset for tensors without explicit offset
    while (!modelReader.eof()) {
        auto [field, wire] = modelReader.readTag();
        if (field == 7 && wire == 2) {
            // graph (GraphProto)
            auto graphReader = modelReader.readLengthDelimited();
            while (!graphReader.eof()) {
                auto [gField, gWire] = graphReader.readTag();
                if (gField == 1 && gWire == 2) {
                    // node (NodeProto)
                    auto nodeReader = graphReader.readLengthDelimited();
                    nodes.push_back(parseNodeProto(nodeReader));
                } else if (gField == 5 && gWire == 2) {
                    // initializer (TensorProto)
                    auto tensorReader = graphReader.readLengthDelimited();
                    auto tensor = parseTensorProto(tensorReader);

                    // Resolve external data pointers
                    if (!tensor.externalLocation.empty() && !externalData.empty()) {
                        int64_t offset = (tensor.externalOffset >= 0)
                            ? tensor.externalOffset : externalDataCursor;
                        int64_t length = tensor.externalLength;
                        if (length < 0) {
                            // Compute from dims and data type
                            int64_t nel = 1;
                            for (auto d : tensor.dims) nel *= d;
                            int bytesPerElem = 1;
                            switch (tensor.dataType) {
                                case ONNX_FLOAT: bytesPerElem = 4; break;
                                case ONNX_FLOAT16: bytesPerElem = 2; break;
                                case ONNX_UINT8: case ONNX_INT8: bytesPerElem = 1; break;
                                case ONNX_INT32: case ONNX_UINT32: bytesPerElem = 4; break;
                                case ONNX_INT64: case ONNX_UINT64: bytesPerElem = 8; break;
                                default: bytesPerElem = 1; break;
                            }
                            length = nel * bytesPerElem;
                        }
                        if (offset + length <= (int64_t)externalData.size()) {
                            tensor.rawData = externalData.data() + offset;
                            tensor.rawSize = (size_t)length;
                            // Advance cursor past this tensor (aligned to 64 bytes)
                            externalDataCursor = offset + length;
                            externalDataCursor = (externalDataCursor + 63) & ~63LL;
                        } else {
                            fprintf(stderr, "  [onnx] WARN: external tensor '%s' offset+length (%lld+%lld=%lld) > file size (%lld)\n",
                                    tensor.name.c_str(), (long long)offset, (long long)length,
                                    (long long)(offset + length), (long long)externalData.size());
                        }
                    }

                    if (!tensor.name.empty())
                        initializers[tensor.name] = std::move(tensor);
                } else {
                    graphReader.skip(gWire);
                }
            }
        } else {
            modelReader.skip(wire);
        }
    }

    printf("  Parsed: %zu initializers, %zu nodes\n",
           initializers.size(), nodes.size());
    fflush(stdout);

    // Detect QK-norm from initializer names
    for (auto& [name, t] : initializers) {
        if (name.find("q_norm") != std::string::npos) {
            cfg.hasQkNorm = true;
            break;
        }
    }

    // Debug: count external vs inline data
    {
        int nExternal = 0, nInline = 0, nNoData = 0;
        size_t totalInlineBytes = 0;
        for (auto& [name, t] : initializers) {
            if (t.rawData && t.rawSize > 0) { nInline++; totalInlineBytes += t.rawSize; }
            else if (!t.externalLocation.empty()) nExternal++;
            else nNoData++;
        }
        fprintf(stderr, "  [onnx] initializers: %d inline (%.1f MB), %d external, %d no-data\n",
                nInline, totalInlineBytes / 1048576.0, nExternal, nNoData);
        fflush(stderr);
    }

    // ─── 4. Extract weights from graph nodes ─────────────────────────────

    uint32_t qDim = cfg.nHead * cfg.headDim;
    uint32_t kvDim = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOut = qDim + 2 * kvDim;

    // Maps from backpack weight name → Q8Repacked
    std::unordered_map<std::string, Q8Repacked> q8Weights;
    // Maps from backpack norm name → float vector
    std::unordered_map<std::string, std::vector<float>> normWeights;

    // ─── 4a. Extract weights directly from initializers by name ──────────
    // ONNX GenAI WebGPU models store Q4G32 weights as initializers with
    // naming pattern: model.layers.N.attn.qkv_proj.MatMul.weight_Q4G32
    // and scales: model.layers.N.attn.qkv_proj.MatMul.weight_scale

    for (auto& [initName, tensor] : initializers) {
        if (!tensor.rawData || tensor.rawSize == 0) continue;

        // Q4G32 weights: name ends with ".weight_Q4G32"
        if (initName.find("weight_Q4G32") != std::string::npos) {
            // Find matching scale tensor
            std::string scaleKey = initName;
            auto pos = scaleKey.find("weight_Q4G32");
            scaleKey.replace(pos, 12, "weight_scale");

            auto scaleIt = initializers.find(scaleKey);
            if (scaleIt == initializers.end() || !scaleIt->second.rawData) continue;

            auto mapping = mapOnnxName(initName);
            if (mapping.backpackName.empty()) continue;

            auto& wT = tensor;
            auto& sT = scaleIt->second;

            // Q4G32 layout: weight [N, nGroups, blockSize/2] uint8
            //               scale  [N, nGroups] fp16
            uint32_t N = (uint32_t)wT.dims[0];
            uint32_t nGroups = (wT.dims.size() >= 2) ? (uint32_t)wT.dims[1] : 1;
            uint32_t blockHalf = (wT.dims.size() >= 3) ? (uint32_t)wT.dims[2] : 16;
            uint32_t blockSize = blockHalf * 2;
            uint32_t K = nGroups * blockSize;

            q8Weights[mapping.backpackName] = dequantQ4ToQ8(
                wT.rawData, sT.rawData,
                N, K, blockSize, nGroups, blockHalf);

            // Infer intermediate_size from gate/up projections
            if (mapping.backpackName.find("gate_up_proj") != std::string::npos ||
                mapping.backpackName.find("gate_proj") != std::string::npos) {
                if (cfg.intermediateSize == 0)
                    cfg.intermediateSize = N;
            }
        }
        // Q8G32 weights: name ends with ".weight_Q8G32"
        else if (initName.find("weight_Q8G32") != std::string::npos) {
            std::string scaleKey = initName;
            auto pos = scaleKey.find("weight_Q8G32");
            scaleKey.replace(pos, 12, "weight_scale");

            auto scaleIt = initializers.find(scaleKey);
            if (scaleIt == initializers.end() || !scaleIt->second.rawData) continue;

            auto mapping = mapOnnxName(initName);
            if (mapping.backpackName.empty()) continue;

            auto& wT = tensor;
            auto& sT = scaleIt->second;

            uint32_t N = (uint32_t)wT.dims[0];
            uint32_t nGroups = (wT.dims.size() >= 2) ? (uint32_t)wT.dims[1] : 1;
            uint32_t blockSize = (wT.dims.size() >= 3) ? (uint32_t)wT.dims[2] : 32;
            uint32_t K = nGroups * blockSize;

            q8Weights[mapping.backpackName] = dequantQ8ToQ8(
                wT.rawData, sT.rawData,
                N, K, blockSize, nGroups);

            if (mapping.backpackName.find("gate_up_proj") != std::string::npos ||
                mapping.backpackName.find("gate_proj") != std::string::npos) {
                if (cfg.intermediateSize == 0)
                    cfg.intermediateSize = N;
            }
        }
        // Norm weights: name ends with "layernorm.weight" or "norm.weight"
        else if (initName.find("layernorm.weight") != std::string::npos ||
                 (initName.find("norm.weight") != std::string::npos &&
                  initName.find("layernorm") == std::string::npos)) {
            std::string bpName = mapOnnxNormName(initName);
            if (bpName.empty()) continue;

            uint32_t nel = 1;
            for (auto d : tensor.dims) nel *= (uint32_t)d;
            std::vector<float> fp32(nel);
            if (tensor.dataType == ONNX_FLOAT16) {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(tensor.rawData);
                for (uint32_t j = 0; j < nel; j++) fp32[j] = fp16_to_f32(src[j]);
            } else if (tensor.dataType == ONNX_FLOAT) {
                memcpy(fp32.data(), tensor.rawData, nel * 4);
            }
            normWeights[bpName] = std::move(fp32);
        }
    }

    // ─── 4b. Extract weights from graph nodes (MatMulNBits, etc.) ────────
    // This handles models that use MatMulNBits custom ops (non-GenAI format)
    // AND handles embedding extraction via GatherBlockQuantized nodes

    for (size_t nodeIdx = 0; nodeIdx < nodes.size(); nodeIdx++) {
        auto& node = nodes[nodeIdx];
        if (nodeIdx < 10 || nodeIdx % 50 == 0) {
            fprintf(stderr, "  [onnx] node %zu/%zu: op=%s inputs=%zu outputs=%zu\n",
                    nodeIdx, nodes.size(), node.opType.c_str(),
                    node.inputs.size(), node.outputs.size());
            for (size_t ii = 0; ii < node.inputs.size(); ii++)
                fprintf(stderr, "    in[%zu]=%s\n", ii, node.inputs[ii].c_str());
            fflush(stderr);
        }
        if (node.opType == "MatMulNBits") {
            // Quantized matmul (Q4 or Q8)
            if (node.inputs.size() < 3) continue;
            auto& wName = node.inputs[1];
            auto& sName = node.inputs[2];

            uint32_t K = (uint32_t)node.getAttrInt("K");
            uint32_t N = (uint32_t)node.getAttrInt("N");
            uint32_t bits = (uint32_t)node.getAttrInt("bits");
            uint32_t blockSize = (uint32_t)node.getAttrInt("block_size");

            auto wIt = initializers.find(wName);
            auto sIt = initializers.find(sName);
            if (wIt == initializers.end() || sIt == initializers.end()) continue;

            auto& wTensor = wIt->second;
            auto& sTensor = sIt->second;
            if (!wTensor.rawData || !sTensor.rawData) continue;

            auto mapping = mapOnnxName(wName);
            fprintf(stderr, "  [MatMulNBits] %s → %s (K=%u N=%u bits=%u bs=%u)\n",
                    wName.c_str(), mapping.backpackName.c_str(), K, N, bits, blockSize);
            if (mapping.backpackName.empty()) continue;

            uint32_t nGroups = (wTensor.dims.size() >= 2)
                ? (uint32_t)wTensor.dims[1] : (K + blockSize - 1) / blockSize;

            if (bits == 4) {
                uint32_t blockHalf = blockSize / 2;
                q8Weights[mapping.backpackName] = dequantQ4ToQ8(
                    wTensor.rawData, sTensor.rawData,
                    N, K, blockSize, nGroups, blockHalf);
            } else if (bits == 8) {
                q8Weights[mapping.backpackName] = dequantQ8ToQ8(
                    wTensor.rawData, sTensor.rawData,
                    N, K, blockSize, nGroups);
            }

            // Infer intermediate_size from gate/up projections
            if (mapping.backpackName.find("gate_up_proj") != std::string::npos ||
                mapping.backpackName.find("gate_proj") != std::string::npos) {
                if (cfg.intermediateSize == 0)
                    cfg.intermediateSize = N;
            }

        } else if (node.opType == "SimplifiedLayerNormalization" ||
                   node.opType == "SkipSimplifiedLayerNormalization") {
            // Norm weight extraction
            size_t wIdx = (node.opType == "SkipSimplifiedLayerNormalization") ? 2 : 1;
            if (node.inputs.size() <= wIdx) continue;
            auto& wName = node.inputs[wIdx];
            auto it = initializers.find(wName);
            if (it == initializers.end() || !it->second.rawData) continue;

            std::string bpName = mapOnnxNormName(wName);
            if (bpName.empty()) continue;

            auto& t = it->second;
            uint32_t nel = 1;
            for (auto d : t.dims) nel *= (uint32_t)d;
            std::vector<float> fp32(nel);
            if (t.dataType == ONNX_FLOAT16) {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(t.rawData);
                for (uint32_t j = 0; j < nel; j++) fp32[j] = fp16_to_f32(src[j]);
            } else if (t.dataType == ONNX_FLOAT) {
                memcpy(fp32.data(), t.rawData, nel * 4);
            }
            normWeights[bpName] = std::move(fp32);

        } else if (node.opType == "Gather") {
            // Embedding (plain fp16/fp32 or int8 quantized)
            if (node.inputs.empty()) continue;
            auto& embName = node.inputs[0];
            auto it = initializers.find(embName);
            if (it == initializers.end() || !it->second.rawData) continue;
            auto& t = it->second;
            if (embName.find("embed") == std::string::npos) continue;

            if (t.dataType == ONNX_INT8) {
                // Int8 quantized embedding (e.g. phi-4-mini eq8)
                // Look for matching scales initializer: replace "_quantized" with "_scales"
                std::string scaleName = embName;
                auto qpos = scaleName.find("_quantized");
                if (qpos != std::string::npos)
                    scaleName.replace(qpos, 10, "_scales");
                else
                    scaleName += "_scales";

                auto scaleIt = initializers.find(scaleName);
                if (scaleIt == initializers.end() || !scaleIt->second.rawData) {
                    fprintf(stderr, "  [onnx] int8 embedding '%s' but scales '%s' not found\n",
                            embName.c_str(), scaleName.c_str());
                    continue;
                }

                auto& st = scaleIt->second;
                uint32_t nVocab = (uint32_t)t.dims[0];
                uint32_t embK = (t.dims.size() >= 2) ? (uint32_t)t.dims[1] : 1;
                uint32_t nGroups = (st.dims.size() >= 2) ? (uint32_t)st.dims[1] : 1;
                uint32_t blockSize = (nGroups > 0) ? embK / nGroups : embK;

                fprintf(stderr, "  [onnx] int8 embedding: %s V=%u K=%u nGroups=%u bs=%u\n",
                        embName.c_str(), nVocab, embK, nGroups, blockSize);

                result.embeddingCPU.resize((size_t)nVocab * embK);
                const int8_t* src = reinterpret_cast<const int8_t*>(t.rawData);
                const uint16_t* scales = reinterpret_cast<const uint16_t*>(st.rawData);

                for (uint32_t row = 0; row < nVocab; row++) {
                    for (uint32_t g = 0; g < nGroups; g++) {
                        float scale = fp16_to_f32(scales[row * nGroups + g]);
                        for (uint32_t j = 0; j < blockSize; j++) {
                            uint32_t col = g * blockSize + j;
                            if (col < embK) {
                                result.embeddingCPU[row * embK + col] =
                                    (float)src[row * embK + col] * scale;
                            }
                        }
                    }
                }
            } else {
                // Plain fp16/fp32 embedding
                uint32_t nel = 1;
                for (auto d : t.dims) nel *= (uint32_t)d;
                result.embeddingCPU.resize(nel);
                if (t.dataType == ONNX_FLOAT16) {
                    const uint16_t* src = reinterpret_cast<const uint16_t*>(t.rawData);
                    for (uint32_t j = 0; j < nel; j++)
                        result.embeddingCPU[j] = fp16_to_f32(src[j]);
                } else if (t.dataType == ONNX_FLOAT) {
                    memcpy(result.embeddingCPU.data(), t.rawData, nel * 4);
                }
            }

        } else if (node.opType == "GatherBlockQuantized") {
            // Quantized embedding
            if (node.inputs.size() < 3) continue;
            auto& embInput = node.inputs[0];
            auto& scaleName = node.inputs[2];

            // Trace through Reshape if needed
            std::string actualEmbName = embInput;
            for (auto& n2 : nodes) {
                if (n2.opType == "Reshape" && !n2.outputs.empty() &&
                    n2.outputs[0] == embInput) {
                    actualEmbName = n2.inputs[0];
                    break;
                }
            }

            fprintf(stderr, "  [onnx] GatherBlockQuantized: emb=%s scale=%s\n",
                    actualEmbName.c_str(), scaleName.c_str());

            auto embIt = initializers.find(actualEmbName);
            auto scaleIt = initializers.find(scaleName);
            if (embIt == initializers.end()) {
                fprintf(stderr, "  [onnx] embedding '%s' not found, skipping\n",
                        actualEmbName.c_str());
                continue;
            }
            if (!embIt->second.rawData || embIt->second.rawSize < 1024) {
                fprintf(stderr, "  [onnx] embedding '%s' has no/insufficient data (rawSize=%zu), skipping\n",
                        actualEmbName.c_str(), embIt->second.rawSize);
                continue;
            }

            auto& embT = embIt->second;
            fprintf(stderr, "  [onnx] emb tensor: type=%d dims=[", embT.dataType);
            for (size_t d = 0; d < embT.dims.size(); d++)
                fprintf(stderr, "%s%lld", d ? "," : "", (long long)embT.dims[d]);
            fprintf(stderr, "] rawSize=%zu\n", embT.rawSize);
            fflush(stderr);

            uint32_t nVocab = (uint32_t)embT.dims[0];
            uint32_t nGroups = (embT.dims.size() >= 2) ? (uint32_t)embT.dims[1] : 1;
            uint32_t bs = (embT.dims.size() >= 3) ? (uint32_t)embT.dims[2] : 32;
            uint32_t embK = nGroups * bs;

            fprintf(stderr, "  [onnx] V=%u nGroups=%u bs=%u embK=%u\n",
                    nVocab, nGroups, bs, embK);
            fprintf(stderr, "  [onnx] dequant output: %llu floats (%.0f MB)\n",
                    (unsigned long long)nVocab * embK,
                    (double)nVocab * embK * 4.0 / 1048576.0);
            fflush(stderr);

            if (scaleIt != initializers.end() && scaleIt->second.rawData) {
                result.embeddingCPU = dequantEmbedding(
                    embT.rawData, scaleIt->second.rawData,
                    nVocab, nGroups, bs, embK);
            } else {
                // No scales, treat as fp16
                uint32_t nel = 1;
                for (auto d : embT.dims) nel *= (uint32_t)d;
                result.embeddingCPU.resize(nel);
                if (embT.dataType == ONNX_FLOAT16) {
                    const uint16_t* src = reinterpret_cast<const uint16_t*>(embT.rawData);
                    for (uint32_t j = 0; j < nel; j++)
                        result.embeddingCPU[j] = fp16_to_f32(src[j]);
                }
            }
        }
    }

    // Also check initializers directly for final norm (may not have a node)
    for (auto& [name, tensor] : initializers) {
        if ((name.find("final_norm") != std::string::npos ||
             name == "model.norm.weight" || name == "norm.weight") &&
            normWeights.find("norm.weight") == normWeights.end()) {
            if (!tensor.rawData) continue;
            uint32_t nel = 1;
            for (auto d : tensor.dims) nel *= (uint32_t)d;
            std::vector<float> fp32(nel);
            if (tensor.dataType == ONNX_FLOAT16) {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(tensor.rawData);
                for (uint32_t j = 0; j < nel; j++) fp32[j] = fp16_to_f32(src[j]);
            } else if (tensor.dataType == ONNX_FLOAT) {
                memcpy(fp32.data(), tensor.rawData, nel * 4);
            }
            normWeights["norm.weight"] = std::move(fp32);
        }
    }

    printf("  Extracted: %zu Q8 weights, %zu norm weights\n",
           q8Weights.size(), normWeights.size());

    // ─── 5. Fuse Q/K/V → QKV and gate/up per layer ──────────────────────

    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        std::string qKey = "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
        std::string kKey = "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
        std::string vKey = "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
        std::string qkvKey = "layers." + std::to_string(i) + ".self_attn.qkv_proj.weight";

        if (q8Weights.count(qKey) && q8Weights.count(kKey) && q8Weights.count(vKey)) {
            auto& qr = q8Weights[qKey];
            auto& kr = q8Weights[kKey];
            auto& vr = q8Weights[vKey];
            Q8Repacked fused;
            fused.N = qr.N + kr.N + vr.N;
            fused.K = qr.K;
            fused.weights.reserve(qr.weights.size() + kr.weights.size() + vr.weights.size());
            fused.weights.insert(fused.weights.end(), qr.weights.begin(), qr.weights.end());
            fused.weights.insert(fused.weights.end(), kr.weights.begin(), kr.weights.end());
            fused.weights.insert(fused.weights.end(), vr.weights.begin(), vr.weights.end());
            fused.scales.reserve(qr.scales.size() + kr.scales.size() + vr.scales.size());
            fused.scales.insert(fused.scales.end(), qr.scales.begin(), qr.scales.end());
            fused.scales.insert(fused.scales.end(), kr.scales.begin(), kr.scales.end());
            fused.scales.insert(fused.scales.end(), vr.scales.begin(), vr.scales.end());
            q8Weights[qkvKey] = std::move(fused);
            q8Weights.erase(qKey);
            q8Weights.erase(kKey);
            q8Weights.erase(vKey);
        }

        std::string gateKey = "layers." + std::to_string(i) + ".mlp.gate_proj.weight";
        std::string upKey = "layers." + std::to_string(i) + ".mlp.up_proj.weight";
        std::string guKey = "layers." + std::to_string(i) + ".mlp.gate_up_proj.weight";

        if (q8Weights.count(gateKey) && q8Weights.count(upKey)) {
            auto& gr = q8Weights[gateKey];
            auto& ur = q8Weights[upKey];
            Q8Repacked fused;
            fused.N = gr.N + ur.N;
            fused.K = gr.K;
            fused.weights.reserve(gr.weights.size() + ur.weights.size());
            fused.weights.insert(fused.weights.end(), gr.weights.begin(), gr.weights.end());
            fused.weights.insert(fused.weights.end(), ur.weights.begin(), ur.weights.end());
            fused.scales.reserve(gr.scales.size() + ur.scales.size());
            fused.scales.insert(fused.scales.end(), gr.scales.begin(), gr.scales.end());
            fused.scales.insert(fused.scales.end(), ur.scales.begin(), ur.scales.end());

            if (cfg.intermediateSize == 0)
                cfg.intermediateSize = gr.N;

            q8Weights[guKey] = std::move(fused);
            q8Weights.erase(gateKey);
            q8Weights.erase(upKey);
        }
    }

    // ─── 6. Organize into per-layer structure ────────────────────────────

    result.layers.resize(cfg.nLayer);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& ld = result.layers[i];
        std::string pfx = "layers." + std::to_string(i) + ".";

        auto moveQ8 = [&](const std::string& key, Q8Repacked& dst) {
            auto it = q8Weights.find(key);
            if (it != q8Weights.end()) {
                dst = std::move(it->second);
                q8Weights.erase(it);
            }
        };
        auto moveNorm = [&](const std::string& key, std::vector<float>& dst) {
            auto it = normWeights.find(key);
            if (it != normWeights.end()) {
                dst = std::move(it->second);
                normWeights.erase(it);
            }
        };

        moveQ8(pfx + "self_attn.qkv_proj.weight", ld.qkv);
        moveQ8(pfx + "self_attn.o_proj.weight", ld.o);
        moveQ8(pfx + "mlp.gate_up_proj.weight", ld.gateup);
        moveQ8(pfx + "mlp.down_proj.weight", ld.down);
        moveNorm(pfx + "input_layernorm.weight", ld.inputNorm);
        moveNorm(pfx + "post_attention_layernorm.weight", ld.postAttnNorm);
        moveNorm(pfx + "q_norm.weight", ld.qNorm);
        moveNorm(pfx + "k_norm.weight", ld.kNorm);

        if (i % 7 == 6 || i == cfg.nLayer - 1)
            printf("  processed layer %u/%u\n", i + 1, cfg.nLayer);
    }

    // Final norm
    {
        auto it = normWeights.find("norm.weight");
        if (it != normWeights.end()) {
            result.finalNorm = std::move(it->second);
            normWeights.erase(it);
        }
    }

    // LM head
    {
        auto it = q8Weights.find("lm_head.weight");
        if (it != q8Weights.end()) {
            result.lmHeadQ8 = std::move(it->second);
            result.hasLmHeadQ8 = true;
            result.tieWordEmbeddings = false;
            cfg.tieWordEmbeddings = false;
            q8Weights.erase(it);
        }
    }

    // ─── 7. Extract pre-computed RoPE tables ─────────────────────────────

    auto cosIt = initializers.find("cos_cache");
    auto sinIt = initializers.find("sin_cache");
    if (cosIt != initializers.end() && sinIt != initializers.end() &&
        cosIt->second.rawData && sinIt->second.rawData) {
        auto& cosT = cosIt->second;
        auto& sinT = sinIt->second;

        uint32_t nel = 1;
        for (auto d : cosT.dims) nel *= (uint32_t)d;

        result.ropeCos.resize(nel);
        result.ropeSin.resize(nel);

        if (cosT.dataType == ONNX_FLOAT) {
            memcpy(result.ropeCos.data(), cosT.rawData, nel * 4);
        } else if (cosT.dataType == ONNX_FLOAT16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(cosT.rawData);
            for (uint32_t j = 0; j < nel; j++) result.ropeCos[j] = fp16_to_f32(src[j]);
        }

        if (sinT.dataType == ONNX_FLOAT) {
            memcpy(result.ropeSin.data(), sinT.rawData, nel * 4);
        } else if (sinT.dataType == ONNX_FLOAT16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(sinT.rawData);
            for (uint32_t j = 0; j < nel; j++) result.ropeSin[j] = fp16_to_f32(src[j]);
        }

        result.hasPrecomputedRope = true;
        result.ropeMaxPositions = (cosT.dims.size() >= 1) ? (uint32_t)cosT.dims[0] : 0;
        result.ropeHalfDim = (cosT.dims.size() >= 2) ? (uint32_t)cosT.dims[1] : nel / result.ropeMaxPositions;

        // Infer rotary_dim from RoPE table dimensions
        result.rotaryDim = 2 * result.ropeHalfDim;

        printf("  RoPE cache: %u positions x %u half-dim (rotary_dim=%u)\n",
               result.ropeMaxPositions, result.ropeHalfDim, result.rotaryDim);
    }

    // Fallback: cos/sin caches may be inside If node subgraphs (phi-4-mini, LongRoPE)
    if (!result.hasPrecomputedRope) {
        for (auto& node : nodes) {
            if (node.opType != "If") continue;
            // Check if this If node outputs cos_cache or sin_cache
            bool outputsCos = false, outputsSin = false;
            for (auto& out : node.outputs) {
                if (out == "cos_cache") outputsCos = true;
                if (out == "sin_cache") outputsSin = true;
            }
            if (!outputsCos && !outputsSin) continue;

            // Search all branches for cos/sin cache tensors, keep the largest
            OnnxTensor bestCos, bestSin;
            uint32_t bestPositions = 0;

            for (auto& attr : node.attributes) {
                if ((attr.name != "then_branch" && attr.name != "else_branch") ||
                    !attr.graphData || attr.graphSize == 0) continue;

                auto constants = extractConstantsFromGraph(
                    attr.graphData, attr.graphSize);

                OnnxTensor* cosTP = nullptr;
                OnnxTensor* sinTP = nullptr;
                for (auto& [name, tensor] : constants) {
                    if (name.find("cos_cache") != std::string::npos && tensor.rawData)
                        cosTP = &tensor;
                    if (name.find("sin_cache") != std::string::npos && tensor.rawData)
                        sinTP = &tensor;
                }
                if (!cosTP || !sinTP) continue;

                uint32_t nPos = cosTP->dims.empty() ? 0 : (uint32_t)cosTP->dims[0];
                if (nPos > bestPositions) {
                    bestPositions = nPos;
                    bestCos = std::move(*cosTP);
                    bestSin = std::move(*sinTP);
                }
            }

            if (bestPositions > 0 && bestCos.rawData && bestSin.rawData) {
                auto loadTable = [](const OnnxTensor& t) -> std::vector<float> {
                    uint32_t nel = 1;
                    for (auto d : t.dims) nel *= (uint32_t)d;
                    std::vector<float> out(nel);
                    if (t.dataType == ONNX_FLOAT) {
                        memcpy(out.data(), t.rawData, nel * 4);
                    } else if (t.dataType == ONNX_FLOAT16) {
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(t.rawData);
                        for (uint32_t j = 0; j < nel; j++) out[j] = fp16_to_f32(src[j]);
                    }
                    return out;
                };

                result.ropeCos = loadTable(bestCos);
                result.ropeSin = loadTable(bestSin);
                result.hasPrecomputedRope = true;
                result.ropeMaxPositions = bestPositions;
                result.ropeHalfDim = (bestCos.dims.size() >= 2)
                    ? (uint32_t)bestCos.dims[1]
                    : (uint32_t)(result.ropeCos.size() / bestPositions);
                result.rotaryDim = 2 * result.ropeHalfDim;

                printf("  RoPE cache (from If subgraph): %u positions x %u half-dim (rotary_dim=%u)\n",
                       result.ropeMaxPositions, result.ropeHalfDim, result.rotaryDim);
            }
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("  ONNX model parsed in %lldms\n", (long long)ms);

    printf("  Model: E=%u, IM=%u, HD=%u, V=%u\n",
           cfg.nEmbd, cfg.intermediateSize, cfg.headDim, cfg.nVocab);

    return true;
}
