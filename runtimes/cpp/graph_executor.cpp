/**
 * graph_executor.cpp — ONNX graph executor on WebGPU.
 *
 * Parses ONNX protobuf, uploads initializers to GPU, then executes
 * nodes in topological order by dispatching registered op kernels.
 */

#include "graph_executor.h"
#include "wgsl_shaders.h"
#include "clock_calibration.h"
#include "profile_html.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <unordered_set>

namespace fs = std::filesystem;

static constexpr bool kDebugExecTrace = false;

namespace {

struct PipelineWarmupSpec {
    const char* name;
    const char* wgsl;
    uint32_t numBindings;
};

static const PipelineWarmupSpec* warmupSpecForOp(const std::string& opType) {
    static const PipelineWarmupSpec kBinaryElementwise{"binary_elementwise", WGSL_BINARY_ELEMENTWISE, 4};
    static const PipelineWarmupSpec kUnaryElementwise{"unary_elementwise", WGSL_UNARY_ELEMENTWISE, 3};
    static const PipelineWarmupSpec kWhereSelect{"where_select", WGSL_WHERE_SELECT, 5};
    static const PipelineWarmupSpec kEqualOp{"equal_op", WGSL_EQUAL_OP, 4};
    static const PipelineWarmupSpec kSoftmax{"softmax", WGSL_SOFTMAX, 3};
    static const PipelineWarmupSpec kMatmulF32{"matmul_f32", WGSL_MATMUL_F32, 4};
    static const PipelineWarmupSpec kMatmulQ4{"matmul_q4", WGSL_MATMUL_Q4, 5};
    static const PipelineWarmupSpec kGemm{"gemm", WGSL_GEMM, 5};
    static const PipelineWarmupSpec kConv2d{"conv2d", WGSL_CONV2D, 5};
    static const PipelineWarmupSpec kConvTranspose2d{"conv_transpose2d", WGSL_CONV_TRANSPOSE2D, 5};
    static const PipelineWarmupSpec kResizeNearest{"resize_nearest", WGSL_RESIZE_NEAREST, 3};
    static const PipelineWarmupSpec kLayerNorm{"layer_norm", WGSL_LAYER_NORM, 5};
    static const PipelineWarmupSpec kInstanceNorm{"instance_norm", WGSL_INSTANCE_NORM, 5};
    static const PipelineWarmupSpec kGroupNorm{"group_norm", WGSL_GROUP_NORM, 5};
    static const PipelineWarmupSpec kGather{"gather", WGSL_GATHER, 4};
    static const PipelineWarmupSpec kTranspose{"transpose", WGSL_TRANSPOSE, 3};
    static const PipelineWarmupSpec kSlice{"slice", WGSL_SLICE, 3};
    static const PipelineWarmupSpec kExpand{"expand", WGSL_EXPAND, 3};
    static const PipelineWarmupSpec kBidirectionalAttn{"bidirectional_attn", WGSL_BIDIRECTIONAL_ATTN, 5};
    static const PipelineWarmupSpec kRotaryEmbedding{"rotary_embedding", WGSL_ROTARY_EMBEDDING, 6};

    if (opType == "Add" || opType == "Sub" || opType == "Mul" || opType == "Div") return &kBinaryElementwise;
    if (opType == "Sigmoid" || opType == "Tanh" || opType == "Neg" || opType == "Sqrt" ||
        opType == "Sin" || opType == "Cos" || opType == "Gelu" || opType == "Silu" ||
        opType == "Erf" || opType == "Relu" || opType == "Exp" || opType == "Log" ||
        opType == "Abs" || opType == "Floor" || opType == "Ceil" || opType == "Round") return &kUnaryElementwise;
    if (opType == "Where") return &kWhereSelect;
    if (opType == "Equal") return &kEqualOp;
    if (opType == "Softmax") return &kSoftmax;
    if (opType == "MatMul") return &kMatmulF32;
    if (opType == "MatMulNBits") return &kMatmulQ4;
    if (opType == "Gemm") return &kGemm;
    if (opType == "Conv") return &kConv2d;
    if (opType == "ConvTranspose") return &kConvTranspose2d;
    if (opType == "Resize") return &kResizeNearest;
    if (opType == "LayerNormalization") return &kLayerNorm;
    if (opType == "InstanceNormalization") return &kInstanceNorm;
    if (opType == "GroupNorm") return &kGroupNorm;
    if (opType == "Gather") return &kGather;
    if (opType == "Transpose") return &kTranspose;
    if (opType == "Slice") return &kSlice;
    if (opType == "Expand") return &kExpand;
    if (opType == "MultiHeadAttention") return &kBidirectionalAttn;
    if (opType == "RotaryEmbedding") return &kRotaryEmbedding;
    return nullptr;
}

static size_t warmupGraphPipelines(GraphExecutor& ex) {
    std::set<std::string> warmed;
    for (const auto& node : ex.GetGraph().nodes) {
        if (node.opType == "Gemm" && ex.gpu->supportsSubgroups) {
            if (warmed.insert("fp16_gemm").second)
                ex.GetPipeline("fp16_gemm", WGSL_FP16_GEMM, 5);
            if (warmed.insert("fp16_gemm_wide").second)
                ex.GetPipeline("fp16_gemm_wide", WGSL_FP16_GEMM_WIDE, 5);
        }
        if (ex.gpu->supportsShaderF16) {
            if (node.opType == "MatMul" && warmed.insert("matmul_f16").second)
                ex.GetPipeline("matmul_f16", WGSL_MATMUL_F16, 4);
            if (node.opType == "Cast") {
                if (warmed.insert("cast_f32_to_f16").second)
                    ex.GetPipeline("cast_f32_to_f16", WGSL_CAST_F32_TO_F16, 3);
                if (warmed.insert("cast_f16_to_f32").second)
                    ex.GetPipeline("cast_f16_to_f32", WGSL_CAST_F16_TO_F32, 3);
            }
            if ((node.opType == "Add" || node.opType == "Sub" || node.opType == "Mul" || node.opType == "Div") &&
                warmed.insert("binary_elementwise_f16").second) {
                ex.GetPipeline("binary_elementwise_f16", WGSL_BINARY_ELEMENTWISE_F16, 4);
            }
            if ((node.opType == "Sigmoid" || node.opType == "Tanh" || node.opType == "Neg" || node.opType == "Sqrt" ||
                 node.opType == "Sin" || node.opType == "Cos" || node.opType == "Gelu" || node.opType == "Silu" ||
                 node.opType == "Erf" || node.opType == "Relu" || node.opType == "Exp" || node.opType == "Log" ||
                 node.opType == "Abs" || node.opType == "Floor" || node.opType == "Ceil" || node.opType == "Round") &&
                warmed.insert("unary_elementwise_f16").second) {
                ex.GetPipeline("unary_elementwise_f16", WGSL_UNARY_ELEMENTWISE_F16, 3);
            }
            if (node.opType == "Conv" && warmed.insert("conv2d_f16").second) {
                ex.GetPipeline("conv2d_f16", WGSL_CONV2D_F16, 5);
            }
            if (node.opType == "ConvTranspose" && warmed.insert("conv_transpose2d_f16").second) {
                ex.GetPipeline("conv_transpose2d_f16", WGSL_CONV_TRANSPOSE2D_F16, 5);
            }
        }
        const PipelineWarmupSpec* spec = warmupSpecForOp(node.opType);
        if (!spec) continue;
        if (!warmed.insert(spec->name).second) continue;
        ex.GetPipeline(spec->name, spec->wgsl, spec->numBindings);
    }
    return warmed.size();
}

}  // anonymous namespace

// ─── Op Registry (static) ───────────────────────────────────────────────────

std::unordered_map<std::string, OpDispatchFn>& GraphExecutor::GetOpRegistry() {
    static std::unordered_map<std::string, OpDispatchFn> registry;
    return registry;
}

void GraphExecutor::RegisterOp(const std::string& opType, OpDispatchFn fn) {
    GetOpRegistry()[opType] = std::move(fn);
}

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

// ─── Minimal ONNX Protobuf Reader ───────────────────────────────────────────
// Reuses the same wire-format parsing approach as onnx_loader.cpp.

namespace {

struct PBReader {
    const uint8_t* data;
    const uint8_t* end;
    PBReader(const uint8_t* d, size_t len) : data(d), end(d + len) {}
    bool eof() const { return data >= end; }

    uint64_t readVarint() {
        uint64_t v = 0; int shift = 0;
        while (data < end) {
            uint8_t b = *data++; v |= (uint64_t)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break; shift += 7;
        }
        return v;
    }
    uint32_t readFixed32() { uint32_t v; memcpy(&v, data, 4); data += 4; return v; }
    PBReader readLengthDelimited() {
        uint64_t len = readVarint();
        PBReader sub(data, (size_t)len); data += len; return sub;
    }
    std::string readString() {
        uint64_t len = readVarint();
        std::string s((const char*)data, (size_t)len); data += len; return s;
    }
    void skip(int wire) {
        switch (wire) {
            case 0: readVarint(); break;
            case 1: data += 8; break;
            case 2: { uint64_t len = readVarint(); data += len; break; }
            case 5: data += 4; break;
        }
    }
    std::pair<uint32_t, int> readTag() {
        uint64_t tag = readVarint();
        return {(uint32_t)(tag >> 3), (int)(tag & 7)};
    }
};

// ONNX data types
enum OnnxDT { DT_FLOAT=1, DT_UINT8=2, DT_INT8=3, DT_UINT16=4, DT_INT16=5,
              DT_INT32=6, DT_INT64=7, DT_FLOAT16=10, DT_BOOL=9 };

static TensorDtype fromOnnxDtype(int dt) {
    switch (dt) {
        case DT_FLOAT: return TensorDtype::Float32;
        case DT_FLOAT16: return TensorDtype::Float16;
        case DT_INT32: return TensorDtype::Int32;
        case DT_INT64: return TensorDtype::Int64;
        case DT_UINT8: return TensorDtype::UInt8;
        case DT_INT8: return TensorDtype::Int8;
        case DT_BOOL: return TensorDtype::Bool;
        default: return TensorDtype::Float32;
    }
}

struct PBTensor {
    std::string name;
    std::vector<int64_t> dims;
    int dataType = 0;
    const uint8_t* rawData = nullptr;
    size_t rawSize = 0;
    std::string extLocation;
    int64_t extOffset = -1, extLength = -1;
    // Inline typed data (protobuf fields 4/5/7)
    std::vector<float> floatData;
    std::vector<int32_t> int32Data;
    std::vector<int64_t> int64Data;
};

static PBTensor parseTensor(PBReader& r) {
    PBTensor t;
    int dataLoc = 0;
    while (!r.eof()) {
        auto [f, w] = r.readTag();
        switch (f) {
            case 1: // dims
                if (w == 2) { auto sub = r.readLengthDelimited(); while (!sub.eof()) t.dims.push_back((int64_t)sub.readVarint()); }
                else t.dims.push_back((int64_t)r.readVarint());
                break;
            case 2: t.dataType = (int)r.readVarint(); break;
            case 8: t.name = r.readString(); break;
            case 14: // data_location
                if (w == 0) { dataLoc = (int)r.readVarint(); }
                else { r.skip(w); }
                break;
            case 4: // float_data (repeated float, packed)
                if (w == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) { uint32_t bits = sub.readFixed32(); float fv; memcpy(&fv, &bits, 4); t.floatData.push_back(fv); }
                } else { uint32_t bits = r.readFixed32(); float fv; memcpy(&fv, &bits, 4); t.floatData.push_back(fv); }
                break;
            case 5: // int32_data (repeated int32, packed)
                if (w == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) t.int32Data.push_back((int32_t)sub.readVarint());
                } else t.int32Data.push_back((int32_t)r.readVarint());
                break;
            case 7: // int64_data (repeated int64, packed)
                if (w == 2) {
                    auto sub = r.readLengthDelimited();
                    while (!sub.eof()) t.int64Data.push_back((int64_t)sub.readVarint());
                } else t.int64Data.push_back((int64_t)r.readVarint());
                break;
            case 9: { // raw_data (bytes) — field 9 in ONNX TensorProto
                uint64_t len = r.readVarint();
                t.rawData = r.data;
                t.rawSize = (size_t)len;
                r.data += len;
                break;
            }
            case 13: // external_data (StringStringEntryProto, repeated) or data_location
                if (w == 0) { dataLoc = (int)r.readVarint(); }
                else {
                    auto sub = r.readLengthDelimited();
                    std::string key, value;
                    while (!sub.eof()) {
                        auto [f2, w2] = sub.readTag();
                        if (f2 == 1) key = sub.readString();
                        else if (f2 == 2) value = sub.readString();
                        else sub.skip(w2);
                    }
                    if (key == "location") t.extLocation = value;
                    else if (key == "offset") t.extOffset = std::stoll(value);
                    else if (key == "length") t.extLength = std::stoll(value);
                }
                break;
            default: r.skip(w); break;
        }
    }
    if (dataLoc == 1) {
        t.rawData = nullptr; t.rawSize = 0;
        if (t.extLocation.empty()) t.extLocation = "model.onnx.data";
    }
    return t;
}

struct PBAttribute {
    std::string name;
    int type = 0;
    float f = 0;
    int64_t i = 0;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<float> floats;
    // Tensor-valued attribute (for Constant node's 'value')
    PBTensor tensor;
    bool hasTensor = false;
};

static PBAttribute parseAttr(PBReader& r) {
    PBAttribute a;
    while (!r.eof()) {
        auto [f, w] = r.readTag();
        switch (f) {
            case 1: a.name = r.readString(); break;
            case 2: { uint32_t bits = r.readFixed32(); memcpy(&a.f, &bits, 4); break; } // f (float)
            case 3: a.i = (int64_t)r.readVarint(); break; // i (int64)
            case 4: a.s = r.readString(); break; // s (bytes)
            case 5: { // t (TensorProto) — tensor-valued attribute
                auto sub = r.readLengthDelimited();
                a.tensor = parseTensor(sub);
                a.hasTensor = true;
                break;
            }
            case 7: // floats (repeated)
                if (w == 2) { auto sub = r.readLengthDelimited(); while (!sub.eof()) { uint32_t bits = sub.readFixed32(); float fv; memcpy(&fv, &bits, 4); a.floats.push_back(fv); } }
                else { uint32_t bits = r.readFixed32(); float fv; memcpy(&fv, &bits, 4); a.floats.push_back(fv); }
                break;
            case 8: // ints (repeated)
                if (w == 2) { auto sub = r.readLengthDelimited(); while (!sub.eof()) a.ints.push_back((int64_t)sub.readVarint()); }
                else a.ints.push_back((int64_t)r.readVarint());
                break;
            case 20: a.type = (int)r.readVarint(); break; // type (AttributeType)
            default: r.skip(w); break;
        }
    }
    return a;
}

struct PBNode {
    std::string opType, name;
    std::vector<std::string> inputs, outputs;
    std::vector<PBAttribute> attrs;
};

static PBNode parseNode(PBReader& r) {
    PBNode n;
    while (!r.eof()) {
        auto [f, w] = r.readTag();
        switch (f) {
            case 1: n.inputs.push_back(r.readString()); break;
            case 2: n.outputs.push_back(r.readString()); break;
            case 3: n.name = r.readString(); break;
            case 4: n.opType = r.readString(); break;
            case 5: { auto sub = r.readLengthDelimited(); n.attrs.push_back(parseAttr(sub)); break; }
            default: r.skip(w); break;
        }
    }
    return n;
}

// Parse ValueInfoProto (graph input/output with type info)
struct PBValueInfo {
    std::string name;
    int elemType = 1;
    std::vector<int64_t> shape;
};

static PBValueInfo parseValueInfo(PBReader& r) {
    PBValueInfo v;
    while (!r.eof()) {
        auto [f, w] = r.readTag();
        if (f == 1 && w == 2) { v.name = r.readString(); }
        else if (f == 2 && w == 2) {
            // TypeProto
            auto tpr = r.readLengthDelimited();
            while (!tpr.eof()) {
                auto [tf, tw] = tpr.readTag();
                if (tf == 1 && tw == 2) {
                    // tensor_type
                    auto ttr = tpr.readLengthDelimited();
                    while (!ttr.eof()) {
                        auto [ttf, ttw] = ttr.readTag();
                        if (ttf == 1) v.elemType = (int)ttr.readVarint();
                        else if (ttf == 2 && ttw == 2) {
                            // shape
                            auto sr = ttr.readLengthDelimited();
                            while (!sr.eof()) {
                                auto [sf, sw] = sr.readTag();
                                if (sf == 1 && sw == 2) {
                                    auto dr = sr.readLengthDelimited();
                                    int64_t dimVal = -1;
                                    while (!dr.eof()) {
                                        auto [df, dw] = dr.readTag();
                                        if (df == 1) dimVal = (int64_t)dr.readVarint();
                                        else if (df == 2) { dr.readString(); dimVal = -1; } // dim_param (symbolic)
                                        else dr.skip(dw);
                                    }
                                    v.shape.push_back(dimVal);
                                } else sr.skip(sw);
                            }
                        } else ttr.skip(ttw);
                    }
                } else tpr.skip(tw);
            }
        } else r.skip(w);
    }
    return v;
}

}  // anonymous namespace

// ─── Load ONNX Model ────────────────────────────────────────────────────────

bool GraphExecutor::Load(GPUContext& gpuCtx, const std::string& onnxPath) {
    gpu = &gpuCtx;
    auto t0 = std::chrono::steady_clock::now();

    // Read .onnx file
    auto fileSize = fs::file_size(onnxPath);
    FILE* f = fopen(onnxPath.c_str(), "rb");
    if (!f) { fprintf(stderr, "GraphExecutor: cannot open %s\n", onnxPath.c_str()); return false; }
    onnxData_.resize(fileSize);
    fread(onnxData_.data(), 1, fileSize, f);
    fclose(f);

    // Read external data if present
    // Support multi-file external data: model.onnx_data, model.onnx_data_1, model.onnx_data_2, ...
    {
        auto dir = fs::path(onnxPath).parent_path();
        auto stem = fs::path(onnxPath).filename().string();

        // Try primary external data file (model.onnx_data or model.onnx.data)
        std::string baseName = stem + "_data";
        std::string extPath = (dir / baseName).string();
        if (!fs::exists(extPath)) {
            extPath = onnxPath + ".data";
            baseName = stem + ".data";
        }
        if (fs::exists(extPath)) {
            // Load primary file
            auto extSize = fs::file_size(extPath);
            printf("  Loading external data: %s (%.0f MB)...\n", baseName.c_str(), extSize / 1048576.0);
            fflush(stdout);
            auto& data = externalDataFiles_[baseName];
            data.resize(extSize);
            FILE* ef = fopen(extPath.c_str(), "rb");
            if (ef) {
                size_t read = 0;
                while (read < extSize) {
                    size_t chunk = std::min((size_t)(256*1024*1024), extSize - read);
                    size_t n = fread(data.data() + read, 1, chunk, ef);
                    if (n == 0) break;
                    read += n;
                }
                fclose(ef);
            }
            // For backward compat, also set externalData_ to primary file
            externalData_ = data;

            // Load numbered continuation files: _data_1, _data_2, ...
            for (int idx = 1; idx < 100; idx++) {
                std::string contName = stem + "_data_" + std::to_string(idx);
                std::string contPath = (dir / contName).string();
                if (!fs::exists(contPath)) break;
                auto contSize = fs::file_size(contPath);
                printf("  Loading external data: %s (%.0f MB)...\n", contName.c_str(), contSize / 1048576.0);
                fflush(stdout);
                auto& contData = externalDataFiles_[contName];
                contData.resize(contSize);
                FILE* cf = fopen(contPath.c_str(), "rb");
                if (cf) {
                    size_t read = 0;
                    while (read < contSize) {
                        size_t chunk = std::min((size_t)(256*1024*1024), contSize - read);
                        size_t n = fread(contData.data() + read, 1, chunk, cf);
                        if (n == 0) break;
                        read += n;
                    }
                    fclose(cf);
                }
            }
        }
    }

    // Parse protobuf: ModelProto → GraphProto
    std::vector<PBTensor> initTensors;
    std::vector<PBNode> pbNodes;
    std::vector<PBValueInfo> graphInputs, graphOutputs;
    int64_t extCursor = 0;

    PBReader model(onnxData_.data(), onnxData_.size());
    while (!model.eof()) {
        auto [mf, mw] = model.readTag();
        if (mf == 7 && mw == 2) {
            auto graph = model.readLengthDelimited();
            while (!graph.eof()) {
                auto [gf, gw] = graph.readTag();
                if (gf == 1 && gw == 2) {
                    auto nr = graph.readLengthDelimited();
                    pbNodes.push_back(parseNode(nr));
                } else if (gf == 5 && gw == 2) {
                    auto tr = graph.readLengthDelimited();
                    auto tensor = parseTensor(tr);
                    // Resolve external data
                    if (!tensor.extLocation.empty()) {
                        // Try multi-file lookup first
                        auto fileIt = externalDataFiles_.find(tensor.extLocation);
                        const std::vector<uint8_t>* extData = nullptr;
                        if (fileIt != externalDataFiles_.end()) {
                            extData = &fileIt->second;
                        } else if (!externalData_.empty()) {
                            extData = &externalData_;  // fallback to primary
                        }
                        if (extData) {
                            int64_t off = (tensor.extOffset >= 0) ? tensor.extOffset : 0;
                            int64_t len = tensor.extLength;
                            if (len < 0) {
                                int64_t nel = 1;
                                for (auto d : tensor.dims) nel *= d;
                                int bpe = 1;
                                switch (tensor.dataType) {
                                    case DT_FLOAT: bpe = 4; break;
                                    case DT_FLOAT16: bpe = 2; break;
                                    case DT_INT64: bpe = 8; break;
                                    case DT_INT32: bpe = 4; break;
                                    default: bpe = 1; break;
                                }
                                len = nel * bpe;
                            }
                            if (off + len <= (int64_t)extData->size()) {
                                tensor.rawData = extData->data() + off;
                                tensor.rawSize = (size_t)len;
                            }
                        }
                    }
                    if (!tensor.name.empty())
                        initTensors.push_back(std::move(tensor));
                } else if (gf == 11 && gw == 2) {
                    // graph input
                    auto vr = graph.readLengthDelimited();
                    graphInputs.push_back(parseValueInfo(vr));
                } else if (gf == 12 && gw == 2) {
                    // graph output
                    auto vr = graph.readLengthDelimited();
                    graphOutputs.push_back(parseValueInfo(vr));
                } else {
                    graph.skip(gw);
                }
            }
        } else {
            model.skip(mw);
        }
    }

    printf("  Parsed: %zu initializers, %zu nodes, %zu inputs, %zu outputs\n",
           initTensors.size(), pbNodes.size(), graphInputs.size(), graphOutputs.size());
    fflush(stdout);

    // Build OnnxGraph
    for (auto& vi : graphInputs) {
        // Skip initializers that also appear as inputs (ONNX convention)
        bool isInit = false;
        for (auto& t : initTensors) if (t.name == vi.name) { isInit = true; break; }
        if (!isInit)
            graph_.inputs.push_back({vi.name, fromOnnxDtype(vi.elemType), vi.shape});
    }
    for (auto& vi : graphOutputs)
        graph_.outputs.push_back({vi.name, fromOnnxDtype(vi.elemType), vi.shape});

    for (auto& n : pbNodes) {
        OnnxGraphNode node;
        node.opType = n.opType;
        node.name = n.name;
        node.inputs = n.inputs;
        node.outputs = n.outputs;
        for (auto& a : n.attrs) {
            if (a.type == 2) node.attrInts[a.name] = a.i;       // INT
            else if (a.type == 1) node.attrFloats[a.name] = a.f; // FLOAT
            else if (a.type == 3) node.attrStrings[a.name] = a.s; // STRING
            else if (a.type == 7) node.attrIntLists[a.name] = a.ints; // INTS

            // Debug: log perm attribute for Transpose nodes
            // (disabled for performance)

            // Handle tensor-valued 'value' attribute (Constant nodes)
            if (a.hasTensor && a.name == "value" && node.opType == "Constant" && !node.outputs.empty()) {
                auto& t = a.tensor;
                auto dtype = fromOnnxDtype(t.dataType);
                const uint8_t* rawPtr = t.rawData;
                size_t rawLen = t.rawSize;
                std::vector<uint8_t> inlineBuf;
                if (!rawPtr && !t.int64Data.empty()) {
                    inlineBuf.resize(t.int64Data.size() * 8);
                    memcpy(inlineBuf.data(), t.int64Data.data(), inlineBuf.size());
                    rawPtr = inlineBuf.data(); rawLen = inlineBuf.size();
                } else if (!rawPtr && !t.floatData.empty()) {
                    inlineBuf.resize(t.floatData.size() * 4);
                    memcpy(inlineBuf.data(), t.floatData.data(), inlineBuf.size());
                    rawPtr = inlineBuf.data(); rawLen = inlineBuf.size();
                }
                if (rawPtr && rawLen > 0) {
                    GpuTensor gt;
                    gt.shape = t.dims;
                    gt.dtype = dtype;
                    gt.isCpuOnly = true;
                    gt.cpuData.resize(rawLen);
                    memcpy(gt.cpuData.data(), rawPtr, rawLen);
                    // Pre-store in tensor store so the Constant op finds it
                    tensorStore_[node.outputs[0]] = std::move(gt);
                    persistentTensors_.insert(node.outputs[0]);
                }
            }
        }
        graph_.nodes.push_back(std::move(node));
    }

    std::set<std::string> nativeFp16Initializers;
    for (const auto& node : graph_.nodes) {
        if (node.opType == "Gemm" && node.inputs.size() >= 2 && node.GetInt("transB", 0) == 1) {
            nativeFp16Initializers.insert(node.inputs[1]);
        }
        if (node.opType == "MatMul" && node.inputs.size() >= 2) {
            nativeFp16Initializers.insert(node.inputs[1]);
        }
        if ((node.opType == "Conv" || node.opType == "ConvTranspose") && node.inputs.size() >= 2) {
            nativeFp16Initializers.insert(node.inputs[1]);
            if (node.inputs.size() >= 3) nativeFp16Initializers.insert(node.inputs[2]);
        }
        if ((node.opType == "SimplifiedLayerNormalization" ||
             node.opType == "SkipSimplifiedLayerNormalization" ||
             node.opType == "LayerNormalization" ||
             node.opType == "InstanceNormalization" ||
             node.opType == "GroupNorm") && node.inputs.size() >= 2) {
            nativeFp16Initializers.insert(node.inputs[1]);
            if (node.inputs.size() >= 3) nativeFp16Initializers.insert(node.inputs[2]);
        }
        if ((node.opType == "Add" || node.opType == "Sub" || node.opType == "Mul" || node.opType == "Div") && node.inputs.size() >= 2) {
            nativeFp16Initializers.insert(node.inputs[0]);
            nativeFp16Initializers.insert(node.inputs[1]);
        }
    }

    // Upload initializers to GPU
    int uploaded = 0, nodata = 0;
    for (auto& t : initTensors) {
        // Resolve inline typed data if no raw_data
        std::vector<uint8_t> inlineDataBuf;
        if (!t.rawData || t.rawSize == 0) {
            if (!t.int64Data.empty()) {
                inlineDataBuf.resize(t.int64Data.size() * 8);
                memcpy(inlineDataBuf.data(), t.int64Data.data(), inlineDataBuf.size());
                t.rawData = inlineDataBuf.data();
                t.rawSize = inlineDataBuf.size();
                if (t.name.find("reshape_shape") != std::string::npos)
                    fprintf(stderr, "    [int64data] '%s': %zu int64 values resolved\n",
                            t.name.c_str(), t.int64Data.size());
            } else if (!t.floatData.empty()) {
                inlineDataBuf.resize(t.floatData.size() * 4);
                memcpy(inlineDataBuf.data(), t.floatData.data(), inlineDataBuf.size());
                t.rawData = inlineDataBuf.data();
                t.rawSize = inlineDataBuf.size();
            } else if (!t.int32Data.empty()) {
                inlineDataBuf.resize(t.int32Data.size() * 4);
                memcpy(inlineDataBuf.data(), t.int32Data.data(), inlineDataBuf.size());
                t.rawData = inlineDataBuf.data();
                t.rawSize = inlineDataBuf.size();
            }
        }
        if (!t.rawData || t.rawSize == 0) {
            nodata++;
            if (nodata <= 3)
                fprintf(stderr, "    [nodata] '%s' dims=%zu rawSize=%zu extLoc='%s' extOff=%lld\n",
                        t.name.c_str(), t.dims.size(), t.rawSize,
                        t.extLocation.c_str(), (long long)t.extOffset);
            // Still register as a valid empty CPU tensor so it's in tensorStore_
            // (other nodes may depend on this name for topo sort resolution)
            auto emptyDtype = fromOnnxDtype(t.dataType);
            GpuTensor gt;
            gt.shape = t.dims;
            gt.dtype = emptyDtype;
            gt.isCpuOnly = true;
            // Empty dims = empty tensor (0 elements), completely valid
            tensorStore_[t.name] = std::move(gt);
            persistentTensors_.insert(t.name);
            uploaded++;
            continue;
        }
        auto dtype = fromOnnxDtype(t.dataType);

        // For small tensors, keep as CPU-only (avoid GPU buffer alignment issues)
        int64_t nel = 1; for (auto d : t.dims) nel *= d;
        bool isTiny = (nel <= 64 && (dtype == TensorDtype::Int64 || dtype == TensorDtype::Int32));

        if (isTiny) {
            GpuTensor gt;
            gt.shape = t.dims;
            gt.dtype = dtype;
            gt.isCpuOnly = true;
            gt.cpuData.resize(t.rawSize);
            memcpy(gt.cpuData.data(), t.rawData, t.rawSize);
            tensorStore_[t.name] = std::move(gt);
            // Store initializer data from the OWNED cpuData (not from inlineDataBuf which is going away)
            auto& stored = tensorStore_[t.name];
            graph_.initializers[t.name] = {stored.cpuData.data(), stored.cpuData.size(), dtype, t.dims};
            persistentTensors_.insert(t.name);
            uploaded++;
            continue;
        }

        // For fp16 tensors that will be used as f32 in compute shaders:
        // Convert to fp32 on upload (except large weight/scale buffers which
        // are read as packed u32 in specialized kernels)
        bool isLargeWeight = (t.name.find("weight_Q4") != std::string::npos ||
                              t.name.find("weight_Q8") != std::string::npos ||
                              t.name.find("_Q4") != std::string::npos ||
                              t.name.find("_Q8") != std::string::npos);
        bool isScale = (t.name.find("scale") != std::string::npos ||
                        t.name.find("Scale") != std::string::npos ||
                        t.name.find("_scales") != std::string::npos);
        bool keepPackedFp16 = (dtype == TensorDtype::Float16 &&
                   nativeFp16Initializers.count(t.name) > 0);
        bool convertToF32 = (dtype == TensorDtype::Float16 && !isLargeWeight && !isScale &&
                     !keepPackedFp16);

        if (convertToF32) {
            // Convert fp16 → fp32
            size_t fp16Count = t.rawSize / 2;
            std::vector<float> fp32(fp16Count);
            const uint16_t* src = (const uint16_t*)t.rawData;
            for (size_t i = 0; i < fp16Count; i++) fp32[i] = fp16_to_f32(src[i]);
            size_t f32Size = fp16Count * 4;
            GpuTensor gt;
            gt.shape = t.dims;
            gt.dtype = TensorDtype::Float32;
            gt.buffer = gpu->createBuffer(t.name, f32Size);
            gpu->writeBuffer(gt.buffer, fp32.data(), f32Size);
            tensorStore_[t.name] = std::move(gt);
            graph_.initializers[t.name] = {t.rawData, t.rawSize, TensorDtype::Float16, t.dims};
            persistentTensors_.insert(t.name);
            uploaded++;
            continue;
        }

        // Pad buffer size to 4-byte alignment for WebGPU
        size_t bufSize = (t.rawSize + 3) & ~(size_t)3;
        GpuTensor gt;
        gt.shape = t.dims;
        gt.dtype = dtype;
        gt.buffer = gpu->createBuffer(t.name, bufSize);
        if (t.rawSize < 4) {
            // Tiny buffer: pad to 4 bytes
            uint8_t padded[4] = {0};
            memcpy(padded, t.rawData, t.rawSize);
            gpu->writeBuffer(gt.buffer, padded, 4);
        } else {
            // Write aligned size
            size_t writeSize = t.rawSize & ~(size_t)3;
            if (writeSize > 0)
                gpu->writeBuffer(gt.buffer, t.rawData, writeSize);
            // Write remaining bytes padded to 4
            if (t.rawSize > writeSize) {
                uint8_t padded[4] = {0};
                memcpy(padded, t.rawData + writeSize, t.rawSize - writeSize);
                gpu->writeBuffer(gt.buffer, padded, 4, writeSize);
            }
        }
        tensorStore_[t.name] = std::move(gt);

        // Also record in graph initializers (for metadata)
        graph_.initializers[t.name] = {t.rawData, t.rawSize, dtype, t.dims};
        persistentTensors_.insert(t.name);
        uploaded++;
    }

    auto warmupT0 = std::chrono::steady_clock::now();
    size_t warmedPipelines = warmupGraphPipelines(*this);
    auto warmupT1 = std::chrono::steady_clock::now();

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto warmupMs = std::chrono::duration_cast<std::chrono::milliseconds>(warmupT1 - warmupT0).count();
    printf("  %d initializers uploaded, %d no-data, %zu pipelines warmed, %lldms\n",
           uploaded, nodata, warmedPipelines, (long long)ms);
    if (warmedPipelines > 0) {
        printf("  Pipeline warmup: %lldms\n", (long long)warmupMs);
    }

    return true;
}

// ─── Tensor Allocation ──────────────────────────────────────────────────────

static std::string g_currentOpLabel;
static const char* g_currentOp = nullptr;  // for debug tracking

GpuTensor GraphExecutor::AllocTensor(std::vector<int64_t> shape,
                                      TensorDtype dtype) {
    GpuTensor t;
    t.shape = std::move(shape);
    t.dtype = dtype;
    size_t bytes = t.ByteSize();
    if (bytes == 0) bytes = 4;  // minimum
    // Sanity check: no single tensor should exceed 2GB
    if (bytes > 2ULL * 1024 * 1024 * 1024) {
        fprintf(stderr, "  [alloc] CORRUPT in %s: shape=[", g_currentOp ? g_currentOp : "?");
        for (size_t i = 0; i < t.shape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)t.shape[i]);
        fprintf(stderr, "] dtype=%d -- RETURNING MINIMAL BUFFER\n", (int)dtype);
        fflush(stderr);
        bytes = 4;  // fallback to minimal
        t.shape = {1};  // fix shape to prevent cascading corruption
    }
    t.buffer = gpu->createBuffer("tmp", bytes);
    return t;
}

GpuTensor GraphExecutor::AllocCpuTensor(const std::vector<int64_t>& shape,
                                          TensorDtype dtype,
                                          const void* data, size_t bytes) {
    GpuTensor t;
    t.shape = shape;
    t.dtype = dtype;
    t.isCpuOnly = true;
    t.cpuData.resize(bytes);
    if (data) memcpy(t.cpuData.data(), data, bytes);
    return t;
}

void GraphExecutor::EnsureGpu(GpuTensor& t) {
    if (t.buffer.handle || !t.isCpuOnly) return;
    size_t bytes = t.cpuData.size();
    if (bytes == 0) {
        // CPU tensor with no data — create minimal buffer
        t.buffer = gpu->createBuffer("cpu2gpu_empty", 4);
        t.isCpuOnly = false;
        return;
    }
    if (bytes == 0) bytes = 4;
    // Align to 4 bytes for WebGPU
    size_t bufSize = (bytes + 3) & ~(size_t)3;
    t.buffer = gpu->createBuffer("cpu2gpu", bufSize);
    if (bytes < 4) {
        uint8_t padded[4] = {0};
        memcpy(padded, t.cpuData.data(), bytes);
        gpu->writeBuffer(t.buffer, padded, 4);
    } else {
        size_t writeSize = bytes & ~(size_t)3;
        if (writeSize > 0)
            gpu->writeBuffer(t.buffer, t.cpuData.data(), writeSize);
        if (bytes > writeSize) {
            uint8_t padded[4] = {0};
            memcpy(padded, t.cpuData.data() + writeSize, bytes - writeSize);
            gpu->writeBuffer(t.buffer, padded, 4, writeSize);
        }
    }
    t.isCpuOnly = false;
}

// ─── Param Buffer Pool ───────────────────────────────────────────────────────

GPUBuffer GraphExecutor::getParamBuffer(uint32_t sizeBytes) {
    // Round up to 16-byte aligned bucket
    int bucket;
    if (sizeBytes <= 16) { bucket = 0; sizeBytes = 16; }
    else if (sizeBytes <= 32) { bucket = 1; sizeBytes = 32; }
    else if (sizeBytes <= 48) { bucket = 2; sizeBytes = 48; }
    else { bucket = 3; sizeBytes = 64; }

    auto& pool = paramPool_[bucket];
    if (pool.buffers.empty()) {
        // Lazy init: pre-allocate a batch
        pool.buffers.resize(PARAM_POOL_SIZE);
        for (int i = 0; i < PARAM_POOL_SIZE; i++) {
            pool.buffers[i] = gpu->createBuffer("param_pool", sizeBytes);
        }
        pool.nextIdx = 0;
    }
    GPUBuffer buf = pool.buffers[pool.nextIdx];
    pool.nextIdx = (pool.nextIdx + 1) % (int)pool.buffers.size();
    return buf;
}

// ─── Pipeline / Dispatch Helpers ─────────────────────────────────────────────

const CompiledPipeline& GraphExecutor::GetPipeline(const std::string& name,
                                                     const std::string& wgsl,
                                                     uint32_t numBindings) {
    return gpu->getOrCreatePipeline(name, wgsl, numBindings);
}

WGPUBindGroup GraphExecutor::MakeBindGroup(
        const CompiledPipeline& pl,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings) {
    WGPUBindGroupEntry entries[16];
    for (size_t i = 0; i < bindings.size() && i < 16; i++) {
        memset(&entries[i], 0, sizeof(WGPUBindGroupEntry));
        entries[i].binding = bindings[i].first;
        entries[i].buffer = bindings[i].second.handle;
        entries[i].size = bindings[i].second.size;
        if (!entries[i].buffer) {
            // Create a dummy 4-byte buffer to avoid null handle crash
            auto dummy = gpu->createBuffer("dummy_bind", 4);
            entries[i].buffer = dummy.handle;
            entries[i].size = 4;
        }
    }
    WGPUBindGroupDescriptor d{};
    d.layout = pl.bgLayout;
    d.entryCount = (uint32_t)bindings.size();
    d.entries = entries;
    return wgpuDeviceCreateBindGroup(gpu->device, &d);
}

void GraphExecutor::Submit(const std::vector<Dispatch>& dispatches) {
    gpu->submitDispatches(dispatches);
    gpu->waitForQueue();
}

void GraphExecutor::FlushPendingWork() {
    bool didWork = false;
    if (!pendingDispatches_.empty()) {
        if (gpuProfiler && gpuProfiler->enabled()) {
            gpu->submitOnlyProfiled(pendingDispatches_, *gpuProfiler);
        } else {
            gpu->submitOnly(pendingDispatches_, false);
        }
        pendingDispatches_.clear();
        didWork = true;
    }
    if (!pendingCopies_.empty()) {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        for (auto& c : pendingCopies_)
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                c.src.handle, c.srcOff, c.dst.handle, c.dstOff, c.size);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
        pendingCopies_.clear();
        didWork = true;
    }
    // Always wait (callers expect data to be ready after flush)
    gpu->waitForQueue();
    if (didWork) {
        flushCount_++;
        std::string key = g_currentOp ? g_currentOp : "(no-op-context)";
        flushSources_[key]++;
    }
}

void GraphExecutor::SubmitAsync(const std::vector<Dispatch>& dispatches) {
    pendingDispatches_.insert(pendingDispatches_.end(),
                              dispatches.begin(), dispatches.end());
}

void GraphExecutor::QueueCopy(GPUBuffer src, uint64_t srcOffset,
                               GPUBuffer dst, uint64_t dstOffset, uint64_t size) {
    if (!src.handle || !dst.handle || size == 0) return;
    // Clamp to buffer sizes
    if (srcOffset + size > src.size) size = (src.size > srcOffset) ? src.size - srcOffset : 0;
    if (dstOffset + size > dst.size) size = (dst.size > dstOffset) ? dst.size - dstOffset : 0;
    if (size == 0) return;
    // WebGPU requires 4-byte alignment for copies
    size = size & ~3ULL;
    srcOffset = srcOffset & ~3ULL;
    dstOffset = dstOffset & ~3ULL;
    if (size == 0) return;

    if (src.handle == dst.handle && srcOffset == dstOffset) return;

    if (!pendingCopies_.empty()) {
        auto& last = pendingCopies_.back();
        if (last.src.handle == src.handle && last.dst.handle == dst.handle &&
            last.srcOff + last.size == srcOffset &&
            last.dstOff + last.size == dstOffset) {
            last.size += size;
            return;
        }
    }

    pendingCopies_.push_back({src, srcOffset, dst, dstOffset, size});
}

void GraphExecutor::Sync() {
    gpu->waitForQueue();
}

// ─── Execute Graph ──────────────────────────────────────────────────────────

void GraphExecutor::Execute(
        const std::unordered_map<std::string, GpuTensor*>& inputs,
        std::unordered_map<std::string, GpuTensor*>& outputs) {

    // Bind graph inputs into tensor store
    // Clear stale intermediate tensors from any previous Execute() call.
    // Persistent entries (initializers + pre-store constants) are preserved.
    for (auto it = tensorStore_.begin(); it != tensorStore_.end(); ) {
        if (persistentTensors_.count(it->first))
            ++it;
        else
            it = tensorStore_.erase(it);
    }

    // Save persistent tensor state (dtype + buffer) before execution.
    // Ops may convert persistent fp16 tensors to f32 in-place via
    // ensureTensorFloat32/ensureCpuBackedFloat32. We restore them at the end
    // so the next Execute() call sees the original state.
    struct SavedState { TensorDtype dtype; GPUBuffer buffer; std::vector<uint8_t> cpuData; bool isCpuOnly; };
    std::unordered_map<std::string, SavedState> savedPersistent;
    for (auto& name : persistentTensors_) {
        auto it = tensorStore_.find(name);
        if (it != tensorStore_.end()) {
            savedPersistent[name] = {it->second.dtype, it->second.buffer,
                                     it->second.cpuData, it->second.isCpuOnly};
        }
    }

    for (auto& [name, tensor] : inputs) {
        tensorStore_[name] = *tensor;
        // For int64 inputs, also store as CPU data for metadata ops
        if (tensor->dtype == TensorDtype::Int64 && tensor->buffer.handle && !tensor->isCpuOnly) {
            int64_t nel = tensor->ElementCount();
            if (nel <= 1024) {
                // Readback small int64 tensor for CPU ops (ReduceSum, Sub, etc.)
                if (!tensor->cpuData.empty()) {
                    tensorStore_[name].cpuData = tensor->cpuData;
                } else {
                    auto rb = gpu->readBuffer(tensor->buffer, nel * 8);
                    tensorStore_[name].cpuData.resize(nel * 8);
                    memcpy(tensorStore_[name].cpuData.data(), rb.data(), nel * 8);
                }
            }
        }
    }

    auto& registry = GetOpRegistry();
    int executed = 0, skipped = 0;
    int totalDispatches = 0, totalCopies = 0;

    // Clear dispatch batch
    pendingDispatches_.clear();

    // Topological sort (cached across Execute calls)
    std::vector<size_t>& execOrder = cachedExecOrder_;
    if (execOrder.empty()) {
        auto sortT0 = std::chrono::steady_clock::now();
        size_t N = graph_.nodes.size();
        std::unordered_set<std::string> available;
        for (auto& [name, _] : inputs) available.insert(name);
        for (auto& [name, _] : tensorStore_) available.insert(name);

        // Build dependency counts
        std::vector<int> depCount(N, 0);
        // Map: tensor name → list of node indices that need it
        std::unordered_map<std::string, std::vector<size_t>> consumers;

        for (size_t ni = 0; ni < N; ni++) {
            auto& node = graph_.nodes[ni];
            for (auto& inName : node.inputs) {
                if (inName.empty() || available.count(inName)) continue;
                depCount[ni]++;
                consumers[inName].push_back(ni);
            }
        }

        // Start with nodes that have 0 dependencies
        std::vector<size_t> ready;
        for (size_t ni = 0; ni < N; ni++)
            if (depCount[ni] == 0) ready.push_back(ni);


        while (!ready.empty()) {
            size_t ni = ready.back(); ready.pop_back();
            execOrder.push_back(ni);
            auto& node = graph_.nodes[ni];
            for (auto& outName : node.outputs) {
                if (outName.empty()) continue;
                available.insert(outName);
                auto it = consumers.find(outName);
                if (it != consumers.end()) {
                    for (auto ci : it->second) {
                        depCount[ci]--;
                        if (depCount[ci] == 0) ready.push_back(ci);
                    }
                }
            }
        }

        // Add remaining unresolved nodes
        if (execOrder.size() < N) {
            size_t unresolved = N - execOrder.size();
            fprintf(stderr, "  [topo] WARNING: %zu unresolved nodes (deps not met)!\n", unresolved);
            for (size_t ni = 0; ni < N; ni++)
                if (depCount[ni] > 0) execOrder.push_back(ni);
            fflush(stderr);
        }

        auto sortT1 = std::chrono::steady_clock::now();
        auto sortMs = std::chrono::duration_cast<std::chrono::milliseconds>(sortT1 - sortT0).count();
        fprintf(stderr, "  [exec] topo sort: %zu/%zu nodes in %lldms\n",
                execOrder.size(), N, (long long)sortMs);
        fflush(stderr);
    }

    // Build tensor reference counts for buffer recycling
    std::unordered_map<std::string, int> tensorRefCount;
    for (size_t ni : execOrder) {
        auto& node = graph_.nodes[ni];
        for (auto& inName : node.inputs)
            if (!inName.empty()) tensorRefCount[inName]++;
    }
    // Output tensors should not be released
    for (auto& [name, _] : outputs) tensorRefCount[name] += 1000;
    // Initializers should not be released
    for (auto& [name, _] : graph_.initializers) tensorRefCount[name] += 1000;
    // Model inputs should not be released (they may be read multiple times)
    for (auto& [name, _] : inputs) tensorRefCount[name] += 1000;

    // Track tensor name aliases: if op X aliases output to input (same buffer handle),
    // record the relationship so we know not to recycle the input while output is live.
    std::unordered_map<std::string, std::string> aliasOf;

    // Execute nodes in topological order
    auto execT0 = std::chrono::steady_clock::now();
    for (size_t ei = 0; ei < execOrder.size(); ei++) {
        size_t ni = execOrder[ei];
        auto& node = graph_.nodes[ni];

        // Resolve input tensor NAMES (not pointers — map may rehash during output allocation)
        std::vector<std::string> inNames;
        for (auto& inName : node.inputs) inNames.push_back(inName);

        // Prepare output tensor slots
        std::vector<std::string> outNames;
        for (size_t oi = 0; oi < node.outputs.size(); oi++) {
            outNames.push_back(node.outputs[oi]);
            if (!node.outputs[oi].empty()) {
                auto it = tensorStore_.find(node.outputs[oi]);
                if (it == tensorStore_.end() || !it->second.IsValid())
                    tensorStore_[node.outputs[oi]] = {};
            }
        }

        // NOW resolve pointers (map is stable after all insertions)
        std::vector<GpuTensor*> inTensors;
        for (auto& inName : inNames) {
            if (inName.empty()) {
                inTensors.push_back(nullptr);
                continue;
            }
            auto it = tensorStore_.find(inName);
            if (it != tensorStore_.end()) {
                inTensors.push_back(&it->second);
            } else {
                inTensors.push_back(nullptr);
            }
        }

        std::vector<GpuTensor*> outTensors;
        for (auto& outName : outNames) {
            if (outName.empty()) {
                outTensors.push_back(nullptr);
            } else {
                outTensors.push_back(&tensorStore_[outName]);
            }
        }

        if (kDebugExecTrace && graph_.nodes.size() > 1000 && ei < 128) {
            auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - execT0).count();
            int validCount = 0, invalidCount = 0;
            for (auto* t : inTensors)
                if (t && t->IsValid()) validCount++; else if (t) invalidCount++;
            fprintf(stderr, "  [exec] ei=%zu node %zu/%zu: %s (valid=%d invalid=%d, %lldms) pending=%zu\n",
                    ei, ni, graph_.nodes.size(), node.opType.c_str(),
                    validCount, invalidCount, (long long)nowMs,
                    pendingDispatches_.size());
            fflush(stderr);
        }
        // Dispatch op
        auto opIt = registry.find(node.opType);
        if (opIt != registry.end()) {
            // Safety: verify all input pointers are still valid before dispatch
            bool inputsOk = true;
            for (size_t ti = 0; ti < inTensors.size(); ti++) {
                if (inTensors[ti] && !inTensors[ti]->IsValid() && !node.inputs[ti].empty()) {
                    // Re-resolve from tensorStore_ in case it was updated
                    auto it2 = tensorStore_.find(node.inputs[ti]);
                    if (it2 != tensorStore_.end())
                        inTensors[ti] = &it2->second;
                }
            }
            g_currentOpLabel = node.opType;
            if (!node.name.empty()) {
                g_currentOpLabel += ":";
                g_currentOpLabel += node.name;
            }
            g_currentOp = g_currentOpLabel.c_str();

            auto opT0 = profilingEnabled ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
            opIt->second(*this, node, inTensors, outTensors);
            if (profilingEnabled) {
                // Include GPU sync in the op time if the op flushed
                auto opT1 = std::chrono::steady_clock::now();
                double opMs = std::chrono::duration<double, std::milli>(opT1 - opT0).count();
                profileData_[node.opType] += opMs;
                profileCounts_[node.opType]++;
            }

            // Cache small int outputs to CPU for downstream metadata ops.
            // Skip for ops whose int outputs are consumed purely by GPU kernels.
            {
                bool isCpuIntConsumer = true;
                // TopK indices go to GPU GatherElements/ScatterElements — skip readback
                if (node.opType == "TopK") isCpuIntConsumer = false;

                if (isCpuIntConsumer) {
                    bool flushed = false;
                    for (auto* outTensor : outTensors) {
                        if (!outTensor || !outTensor->buffer.handle || !outTensor->cpuData.empty()) continue;
                        if (outTensor->dtype != TensorDtype::Int64 && outTensor->dtype != TensorDtype::Int32) continue;
                        int64_t nel = outTensor->ElementCount();
                        if (nel <= 0 || nel > 1024) continue;
                        if (!flushed) {
                            FlushPendingWork();
                            flushed = true;
                            intReadbackSyncs_++;
                        }
                        size_t bytes = (size_t)nel * outTensor->DtypeSize();
                        auto rb = gpu->readBuffer(outTensor->buffer, bytes);
                        if (rb.size() >= bytes) {
                            outTensor->cpuData.resize(bytes);
                            memcpy(outTensor->cpuData.data(), rb.data(), bytes);
                        }
                    }
                }
            }
            executed++;
            if (kDebugExecTrace && graph_.nodes.size() > 1000 && ei < 128) {
                fprintf(stderr, "  [exec] ei=%zu done %s pending=%zu\n",
                        ei, node.opType.c_str(), pendingDispatches_.size() + pendingCopies_.size());
                fflush(stderr);
            }

            // Track aliases: if an op aliased its output to an input buffer
            for (size_t oi = 0; oi < outTensors.size() && oi < node.outputs.size(); oi++) {
                if (!outTensors[oi] || !outTensors[oi]->buffer.handle || node.outputs[oi].empty()) continue;
                // Trace output buffer for graph outputs (disabled for perf)
                for (size_t ti = 0; ti < inTensors.size() && ti < node.inputs.size(); ti++) {
                    if (!inTensors[ti] || node.inputs[ti].empty()) continue;
                    if (outTensors[oi]->buffer.handle == inTensors[ti]->buffer.handle) {
                        aliasOf[node.outputs[oi]] = node.inputs[ti];
                    }
                }
            }

            // Debug: readback outputs to trace zero propagation
            // Debug readback for first few ops (production mode: change to false)
            if (false && graph_.nodes.size() > 1000) {
                // Always log the op type and output validity
                fprintf(stderr, "  [dbg] ei=%zu %s valid=%d buf=%p",
                        ei, node.opType.c_str(),
                        (outTensors[0] && outTensors[0]->IsValid()) ? 1 : 0,
                        outTensors[0] ? (void*)outTensors[0]->buffer.handle : nullptr);
                // If valid, readback
                if (outTensors[0] && outTensors[0]->IsValid() && outTensors[0]->buffer.handle) {
                    if (!pendingDispatches_.empty()) {
                        gpu->submitOnly(pendingDispatches_, false);
                        pendingDispatches_.clear();
                    }
                    if (!pendingCopies_.empty()) {
                        WGPUCommandEncoderDescriptor enD{};
                        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
                        for (auto& c : pendingCopies_)
                            wgpuCommandEncoderCopyBufferToBuffer(enc,
                                c.src.handle, c.srcOff, c.dst.handle, c.dstOff, c.size);
                        WGPUCommandBufferDescriptor cbD{};
                        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
                        wgpuQueueSubmit(gpu->queue, 1, &cb);
                        wgpuCommandBufferRelease(cb);
                        wgpuCommandEncoderRelease(enc);
                        pendingCopies_.clear();
                    }
                    gpu->waitForQueue();
                    size_t readN = std::min((size_t)4, outTensors[0]->buffer.size / 4);
                    if (readN > 0) {
                        auto rb = gpu->readBuffer(outTensors[0]->buffer, readN * 4);
                        float vals[4] = {0};
                        memcpy(vals, rb.data(), std::min(readN * 4, rb.size()));
                        fprintf(stderr, " out=[%.4f, %.4f, %.4f, %.4f]", vals[0], vals[1], vals[2], vals[3]);
                    }
                }
                fprintf(stderr, "\n"); fflush(stderr);
            }

            // Buffer lifetime tracking: decrement refcounts for later cleanup.
            // Actual release happens at end of Execute() to avoid GPU read-after-free.
            if (graph_.nodes.size() > 100) {
                for (size_t ti = 0; ti < node.inputs.size(); ti++) {
                    auto& inName = node.inputs[ti];
                    if (inName.empty()) continue;
                    tensorRefCount[inName]--;
                }
            }

            // Submit pending work periodically (with sync to prevent TDR)
            if (pendingDispatches_.size() + pendingCopies_.size() >= 64) {
                totalDispatches += (int)pendingDispatches_.size();
                totalCopies += (int)pendingCopies_.size();
                if (!pendingDispatches_.empty()) {
                    if (gpuProfiler && gpuProfiler->enabled())
                        gpu->submitOnlyProfiled(pendingDispatches_, *gpuProfiler);
                    else
                        gpu->submitOnly(pendingDispatches_, true);
                }
                if (!pendingCopies_.empty()) {
                    WGPUCommandEncoderDescriptor enD{};
                    auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
                    for (auto& c : pendingCopies_)
                        wgpuCommandEncoderCopyBufferToBuffer(enc,
                            c.src.handle, c.srcOff, c.dst.handle, c.dstOff, c.size);
                    WGPUCommandBufferDescriptor cbD{};
                    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
                    wgpuQueueSubmit(gpu->queue, 1, &cb);
                    wgpuCommandBufferRelease(cb);
                    wgpuCommandEncoderRelease(enc);
                }
                gpu->waitForQueue();
                pendingDispatches_.clear();
                pendingCopies_.clear();

                // Buffer release at sync points is disabled for now — the shared
                // handle detection needs more work to handle aliased tensors safely.
                // TODO: implement proper buffer lifetime analysis.
                pendingReleases_.clear();
            }
        } else {
            if (skipped < 5)
                fprintf(stderr, "  [exec] UNIMPL op: %s (%s)\n",
                        node.opType.c_str(), node.name.c_str());
            skipped++;
        }
    }

    // Copy outputs
    for (auto& [name, tensor] : outputs) {
        auto it = tensorStore_.find(name);
        if (it != tensorStore_.end() && it->second.IsValid()) {
            size_t outBytes = it->second.ByteSize();
            if (outBytes == 0) outBytes = 4;
            if (tensor && tensor->buffer.handle && it->second.buffer.handle &&
                tensor->buffer.handle != it->second.buffer.handle &&
                tensor->buffer.size >= outBytes && !it->second.isCpuOnly) {
                tensor->shape = it->second.shape;
                tensor->dtype = it->second.dtype;
                tensor->isCpuOnly = false;
                tensor->cpuData.clear();
                QueueCopy(it->second.buffer, 0, tensor->buffer, 0, outBytes);
            } else {
                *tensor = it->second;
            }
        } else {
            fprintf(stderr, "  [exec] WARNING: output '%s' not found or invalid buf=%p\n",
                    name.c_str(), (it != tensorStore_.end()) ? (void*)it->second.buffer.handle : nullptr);
            fflush(stderr);
        }
    }

    totalDispatches += (int)pendingDispatches_.size();
    totalCopies += (int)pendingCopies_.size();
    fprintf(stderr, "  [exec] %d/%zu ops executed, %d unimplemented, %d dispatches, %d copies, %d syncs (0 from int-readback), bufs=%d (pool=%d)\n",
            executed, graph_.nodes.size(), skipped, totalDispatches, totalCopies, flushCount_, 
            gpu->createBufferCount, gpu->poolHitCount);
    gpu->createBufferCount = 0;
    gpu->poolHitCount = 0;
    // Print sync sources on first call
    static int printSyncCount = 0;
    if (printSyncCount < 2 && !flushSources_.empty()) {
        std::vector<std::pair<std::string, int>> sorted(flushSources_.begin(), flushSources_.end());
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
        fprintf(stderr, "  [sync sources] %d total:\n", flushCount_);
        for (auto& [op, cnt] : sorted)
            fprintf(stderr, "    %-50s %d\n", op.c_str(), cnt);
        printSyncCount++;
    }
    flushCount_ = 0;
    intReadbackSyncs_ = 0;
    flushSources_.clear();

    // Submit all remaining batched GPU work
    if (!pendingDispatches_.empty()) {
        if (gpuProfiler && gpuProfiler->enabled())
            gpu->submitOnlyProfiled(pendingDispatches_, *gpuProfiler);
        else
            gpu->submitOnly(pendingDispatches_, true);
        pendingDispatches_.clear();
    }
    if (!pendingCopies_.empty()) {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        for (auto& c : pendingCopies_)
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                c.src.handle, c.srcOff, c.dst.handle, c.dstOff, c.size);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
        pendingCopies_.clear();
    }
    gpu->waitForQueue();

    // Release non-output, non-persistent intermediate buffers
    // (GPU work is done, safe to free)
    {
        std::set<WGPUBuffer> keepHandles;
        for (auto& [name, tensor] : outputs)
            if (tensor && tensor->buffer.handle) keepHandles.insert(tensor->buffer.handle);
        // Persistent tensors (initializers, pre-store constants) must survive for re-execution
        for (auto& name : persistentTensors_) {
            auto it = tensorStore_.find(name);
            if (it != tensorStore_.end() && it->second.buffer.handle)
                keepHandles.insert(it->second.buffer.handle);
        }

        std::set<WGPUBuffer> released;
        for (auto it = tensorStore_.begin(); it != tensorStore_.end(); ) {
            if (persistentTensors_.count(it->first)) {
                ++it;  // keep persistent entries
                continue;
            }
            if (it->second.buffer.handle &&
                keepHandles.find(it->second.buffer.handle) == keepHandles.end() &&
                released.find(it->second.buffer.handle) == released.end()) {
                released.insert(it->second.buffer.handle);
                gpu->releaseBuffer(it->second.buffer);
            }
            it = tensorStore_.erase(it);
        }
    }

    // Restore persistent tensors that may have been modified during execution
    // (e.g. ensureTensorFloat32 converting fp16 weights to f32 in-place)
    for (auto& [name, saved] : savedPersistent) {
        auto it = tensorStore_.find(name);
        if (it != tensorStore_.end() &&
            (it->second.dtype != saved.dtype || it->second.buffer.handle != saved.buffer.handle)) {
            it->second.dtype = saved.dtype;
            it->second.buffer = saved.buffer;
            it->second.cpuData = std::move(saved.cpuData);
            it->second.isCpuOnly = saved.isCpuOnly;
        }
    }

    // Print profiling report
    if (profilingEnabled && !profileData_.empty()) {
        double totalMs = 0;
        for (auto& [op, ms] : profileData_) totalMs += ms;

        // Sort by time descending
        std::vector<std::pair<std::string, double>> sorted(profileData_.begin(), profileData_.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        fprintf(stderr, "\n  ┌─ Profile (%d ops, %.1fms total) ─────────────────────┐\n",
                executed, totalMs);
        fprintf(stderr, "  │ %-25s %7s %5s %7s │\n", "Op", "ms", "cnt", "%");
        fprintf(stderr, "  ├───────────────────────────────────────────────────────┤\n");
        for (auto& [op, ms] : sorted) {
            int cnt = profileCounts_[op];
            double pct = totalMs > 0 ? 100.0 * ms / totalMs : 0;
            fprintf(stderr, "  │ %-25s %7.1f %5d %6.1f%% │\n", op.c_str(), ms, cnt, pct);
        }
        fprintf(stderr, "  └───────────────────────────────────────────────────────┘\n");
        fflush(stderr);
        profileData_.clear();
        profileCounts_.clear();

        // Print sync sources
        if (!flushSources_.empty()) {
            std::vector<std::pair<std::string, int>> sortedSync(flushSources_.begin(), flushSources_.end());
            std::sort(sortedSync.begin(), sortedSync.end(),
                      [](auto& a, auto& b) { return a.second > b.second; });
            fprintf(stderr, "\n  GPU syncs (%d total):\n", flushCount_);
            for (auto& [op, cnt] : sortedSync) {
                fprintf(stderr, "    %-40s %d\n", op.c_str(), cnt);
            }
            flushSources_.clear();
        }
    }
}

// ─── GPU Hardware Timestamp Profiling ────────────────────────────────────────

void GraphExecutor::enableGpuProfiling() {
    if (!gpu || !gpu->supportsTimestampQuery) {
        fprintf(stderr, "GPU timestamp queries not supported\n");
        return;
    }
    gpuProfiler = new GPUProfiler();
    if (!gpuProfiler->init(gpu->device, gpu->instance, gpu->queue)) {
        fprintf(stderr, "Failed to init GPU profiler\n");
        delete gpuProfiler;
        gpuProfiler = nullptr;
        return;
    }
    auto cal = acquireClockCalibration(gpu->device, gpu->backendType);
    if (cal.valid) {
        clockCalibration = new ClockCalibration(cal);
    }
}

void GraphExecutor::printGpuProfileReport(int nDecodeTokens, double decodeMs,
                                           const std::string& htmlPath) {
    if (!gpuProfiler || !gpuProfiler->enabled() || gpuProfiler->nextIndex == 0) {
        fprintf(stderr, "No GPU profile data\n");
        return;
    }

    // Resolve timestamps
    {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        gpuProfiler->resolveAndReport(enc);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
    }
    gpu->waitForQueue();

    // Map readback buffer
    uint32_t count = gpuProfiler->nextIndex;
    uint64_t readSize = count * 8;
    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(gpuProfiler->readbackBuf, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(gpu->instance, 1, &mw, UINT64_MAX);

    if (ms.status != 1) {
        fprintf(stderr, "Failed to map profiler readback buffer\n");
        return;
    }

    auto ptr = (const uint64_t*)wgpuBufferGetConstMappedRange(
        gpuProfiler->readbackBuf, 0, readSize);

    // Aggregate by kernel name
    struct AggEntry { double totalUs = 0; uint32_t count = 0; };
    std::unordered_map<std::string, AggEntry> agg;
    double totalGpuUs = 0;
    for (auto& e : gpuProfiler->entries) {
        uint64_t begin = ptr[e.beginIdx], end = ptr[e.endIdx];
        if (end <= begin || begin == 0) continue;
        double durUs = (double)(end - begin) / 1000.0;
        agg[e.name].totalUs += durUs;
        agg[e.name].count++;
        totalGpuUs += durUs;
    }

    std::vector<std::pair<std::string, AggEntry>> sorted(agg.begin(), agg.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.second.totalUs > b.second.totalUs; });

    fprintf(stderr, "\n--- GPU Profile (hardware timestamps, %d dispatches) ---\n",
            (int)gpuProfiler->entries.size());
    fprintf(stderr, "%-25s %10s %6s %10s %6s\n",
            "Kernel", "Total(ms)", "Count", "Avg(us)", "%%");
    fprintf(stderr, "%-25s %10s %6s %10s %6s\n",
            "-------------------------", "----------", "------", "----------", "------");
    for (auto& [name, e] : sorted) {
        double totalMs = e.totalUs / 1000.0;
        double avgUs = e.totalUs / e.count;
        double pct = totalGpuUs > 0 ? e.totalUs / totalGpuUs * 100.0 : 0;
        fprintf(stderr, "%-25s %10.2f %6u %10.1f %5.1f%%\n",
                name.c_str(), totalMs, e.count, avgUs, pct);
    }
    double totalGpuMs = totalGpuUs / 1000.0;
    double cpuMs = decodeMs / std::max(1, nDecodeTokens);
    fprintf(stderr, "%-25s %10.2f\n", "GPU TOTAL", totalGpuMs);
    fprintf(stderr, "\nGPU HW time: %.1fms/tok   CPU wall time: %.1fms/tok   Bubble: %.0f%%\n",
            totalGpuMs / std::max(1, nDecodeTokens), cpuMs,
            cpuMs > 0 ? (1.0 - totalGpuMs / std::max(1, nDecodeTokens) / cpuMs) * 100 : 0);

    // Generate HTML timeline
    generateProfileHTML(*gpu, *gpuProfiler, clockCalibration, ptr,
                        nDecodeTokens, 0, 0, decodeMs, htmlPath);
    fprintf(stderr, "Profile HTML: %s\n", htmlPath.c_str());

    wgpuBufferUnmap(gpuProfiler->readbackBuf);
    gpuProfiler->destroy();
    delete gpuProfiler;
    gpuProfiler = nullptr;
    if (clockCalibration) { delete clockCalibration; clockCalibration = nullptr; }
}
