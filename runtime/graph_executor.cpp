/**
 * graph_executor.cpp — ONNX graph executor on WebGPU.
 *
 * Parses ONNX protobuf, uploads initializers to GPU, then executes
 * nodes in topological order by dispatching registered op kernels.
 */

#include "graph_executor.h"
#include "wgsl_shaders.h"
#include "wgsl_template.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <unordered_set>

namespace fs = std::filesystem;

static constexpr bool kDebugExecTrace = false;

GraphExecutor::~GraphExecutor() {
    if (!gpu) return;
    // Release weight store buffers
    for (auto& [name, tensor] : weightStore_) {
        if (tensor.buffer.handle)
            gpu->releaseBuffer(tensor.buffer);
    }
    weightStore_.clear();
}

namespace {

static size_t warmupGraphPipelines(GraphExecutor& ex) {
    // Collect unique op types from the graph
    std::set<std::string> opTypes;
    for (const auto& node : ex.GetGraph().nodes)
        opTypes.insert(node.opType);

    std::set<std::string> seen;
    bool hasF16 = ex.gpu->supportsShaderF16;

    // Collect (name, wgsl, numBindings) specs for batch warmup
    std::vector<std::tuple<std::string, std::string, uint32_t>> specs;

    auto addT = [&](const std::string& name, const char* tmpl, uint32_t nb, TensorDtype dt) {
        if (!seen.insert(name).second) return;
        specs.emplace_back(name, instantiateTemplate(tmpl, dt), nb);
    };
    auto addRaw = [&](const std::string& name, const char* wgsl, uint32_t nb) {
        if (!seen.insert(name).second) return;
        specs.emplace_back(name, std::string(wgsl), nb);
    };

    auto has = [&](const char* op) { return opTypes.count(op) > 0; };

    bool hasBinary = has("Add") || has("Sub") || has("Mul") || has("Div");
    bool hasUnary  = has("Sigmoid") || has("Tanh") || has("Neg") || has("Sqrt") ||
                     has("Sin") || has("Cos") || has("Gelu") || has("Silu") ||
                     has("Erf") || has("Relu") || has("Exp") || has("Log") ||
                     has("Abs") || has("Floor") || has("Ceil") || has("Round");

    // ── Elementwise ops ──
    if (hasBinary) {
        addT("binary_t_f32", WGSL_BINARY_ELEMENTWISE_T, 4, TensorDtype::Float32);
        if (hasF16) addT("binary_t_f16", WGSL_BINARY_ELEMENTWISE_T, 4, TensorDtype::Float16);
    }
    if (hasUnary) {
        addT("unary_t_f32", WGSL_UNARY_ELEMENTWISE_T, 3, TensorDtype::Float32);
        if (hasF16) addT("unary_t_f16", WGSL_UNARY_ELEMENTWISE_T, 3, TensorDtype::Float16);
    }
    if (has("Where"))
        addT("where_select", WGSL_WHERE_SELECT_T, 5, TensorDtype::Float32);
    if (has("Equal"))
        addT("equal_op", WGSL_EQUAL_OP_T, 4, TensorDtype::Float32);
    if (has("Softmax"))
        addT("softmax_t_f32", WGSL_SOFTMAX_T, 3, TensorDtype::Float32);

    // ── Cast ops ──
    if (has("Cast") && hasF16) {
        addRaw("cast_f32_to_f16", WGSL_CAST_F32_TO_F16, 3);
        addRaw("cast_f16_to_f32", WGSL_CAST_F16_TO_F32, 3);
    }

    // ── MatMul / Gemm ──
    if (has("MatMul")) {
        addT("matmul_f32", WGSL_MATMUL_T, 4, TensorDtype::Float32);
        if (hasF16) addRaw("matmul_f16", WGSL_MATMUL_F16, 4);
    }
    if (has("MatMulNBits"))
        addRaw("matmul_q4", WGSL_MATMUL_Q4, 5);
    if (has("Gemm")) {
        addT("gemm", WGSL_GEMM_T, 5, TensorDtype::Float32);
        if (ex.gpu->supportsSubgroups) {
            addRaw("fp16_gemm", WGSL_FP16_GEMM, 5);
            addRaw("fp16_gemm_wide", WGSL_FP16_GEMM_WIDE, 5);
        }
    }

    // ── Conv / Resize ──
    if (has("Conv")) {
        addT("conv2d_f32", WGSL_CONV2D_T, 5, TensorDtype::Float32);
        if (hasF16) addRaw("conv2d_f16", WGSL_CONV2D_F16, 5);
    }
    if (has("ConvTranspose")) {
        addT("conv_transpose2d_f32", WGSL_CONV_TRANSPOSE2D_T, 5, TensorDtype::Float32);
        if (hasF16) addRaw("conv_transpose2d_f16", WGSL_CONV_TRANSPOSE2D_F16, 5);
    }
    if (has("Resize"))
        addT("resize_nearest", WGSL_RESIZE_NEAREST_T, 3, TensorDtype::Float32);

    // ── Normalization ──
    if (has("SimplifiedLayerNormalization") || has("SkipSimplifiedLayerNormalization")) {
        addT("rmsnorm_simple", WGSL_RMSNORM_T, 5, TensorDtype::Float32);
        if (hasF16) addT("rmsnorm_t_f16", WGSL_RMSNORM_T, 5, TensorDtype::Float16);
    }
    if (has("SkipSimplifiedLayerNormalization"))
        addT("skip_rmsnorm", WGSL_SKIP_RMSNORM_T, 6, TensorDtype::Float32);
    if (has("LayerNormalization"))
        addT("layer_norm", WGSL_LAYER_NORM_T, 5, TensorDtype::Float32);
    if (has("InstanceNormalization"))
        addT("instance_norm", WGSL_INSTANCE_NORM_T, 5, TensorDtype::Float32);
    if (has("GroupNorm"))
        addT("group_norm", WGSL_GROUP_NORM_T, 5, TensorDtype::Float32);

    // ── Shape ops ──
    if (has("Gather"))
        addRaw("gather", WGSL_GATHER, 4);
    if (has("Transpose")) {
        addRaw("transpose", WGSL_TRANSPOSE, 3);
        addT("transpose_f32", WGSL_TRANSPOSE_T, 3, TensorDtype::Float32);
    }
    if (has("Slice")) {
        addRaw("slice", WGSL_SLICE, 3);
        addT("slice_f32", WGSL_SLICE_T, 3, TensorDtype::Float32);
    }
    if (has("Expand"))
        addT("expand_t_f32", WGSL_EXPAND_T, 3, TensorDtype::Float32);
    if (has("Concat"))
        addT("concat_2t_f32", WGSL_CONCAT_2INPUT_T, 4, TensorDtype::Float32);

    // ── Attention ops ──
    if (has("MultiHeadAttention") || has("GroupQueryAttention"))
        addT("bidirectional_attn", WGSL_BIDIRECTIONAL_ATTN_T, 5, TensorDtype::Float32);
    if (has("RotaryEmbedding"))
        addT("rotary_embedding", WGSL_ROTARY_EMBEDDING_T, 6, TensorDtype::Float32);
    if (has("GroupQueryAttention")) {
        addT("rope_inplace", WGSL_ROPE_INPLACE_T, 4, TensorDtype::Float32);
        addT("kv_cache_append", WGSL_KV_CACHE_APPEND_T, 4, TensorDtype::Float32);
        addT("kv_cache_write", WGSL_KV_CACHE_WRITE_T, 3, TensorDtype::Float32);
        addT("gqa_decode", WGSL_GQA_DECODE_T, 5, TensorDtype::Float32);
        if (hasF16) {
            addRaw("cast_f32_to_f16", WGSL_CAST_F32_TO_F16, 3);
            addRaw("cast_f16_to_f32", WGSL_CAST_F16_TO_F32, 3);
        }
    }

    // ── MoE ops ──
    if (has("QMoE")) {
        addT("moe_gate", WGSL_MOE_GATE_T, 4, TensorDtype::Float32);
        addT("matmul_q4_indirect", WGSL_MATMUL_Q4_INDIRECT_T, 6, TensorDtype::Float32);
        addT("swiglu", WGSL_SWIGLU_T, 3, TensorDtype::Float32);
        addT("weighted_add_indirect", WGSL_WEIGHTED_ADD_INDIRECT_T, 4, TensorDtype::Float32);
    }
    if (has("TopK"))
        addT("topk_f32", WGSL_TOPK_T, 4, TensorDtype::Float32);
    if (has("GatherElements"))
        addT("gather_elements_f32", WGSL_GATHER_ELEMENTS_T, 4, TensorDtype::Float32);
    if (has("ScatterElements"))
        addT("scatter_elements_f32", WGSL_SCATTER_ELEMENTS_T, 5, TensorDtype::Float32);
    if (has("LinearMxfp4"))
        addRaw("mxfp4_matmul", WGSL_MXFP4_MATMUL, 6);
    if (has("GptOssGate"))
        addT("gptoss_gate_f32", WGSL_GPTOSS_GATE_T, 3, TensorDtype::Float32);
    if (has("GQAAttnSink"))
        addRaw("gqa_decode_attn_sink", WGSL_GQA_DECODE_ATTN_SINK, 6);
    if (has("GeluMul"))
        addT("gelu_mul_f32", WGSL_GELU_MUL_T, 4, TensorDtype::Float32);
    if (has("AddScaled"))
        addT("add_scaled_f32", WGSL_ADD_SCALED_T, 3, TensorDtype::Float32);
    if (has("ModScaleShift"))
        addT("mod_scale_shift_f32", WGSL_MOD_SCALE_SHIFT_T, 5, TensorDtype::Float32);
    if (has("GateResidualAdd"))
        addT("gate_residual_add_f32", WGSL_GATE_RESIDUAL_ADD_T, 4, TensorDtype::Float32);
    if (has("SigmoidGateInterleaved"))
        addT("sigmoid_gate_interleaved_f32", WGSL_SIGMOID_GATE_INTERLEAVED_T, 3, TensorDtype::Float32);
    if (has("Concat2D"))
        addT("concat_2d_f32", WGSL_CONCAT_2D_T, 4, TensorDtype::Float32);
    if (has("SplitCopy"))
        addT("split_copy_f32", WGSL_SPLIT_COPY_T, 3, TensorDtype::Float32);

    // Batch-compile all collected specs (parallel shader + async pipeline)
    if (!specs.empty())
        ex.gpu->warmupPipelines(specs);

    return seen.size();
}

}  // anonymous namespace

size_t GraphExecutor::WarmupAllPipelines() {
    return warmupGraphPipelines(*this);
}

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

    // Memory-map .onnx file
    if (!onnxMapping_.open(onnxPath)) {
        fprintf(stderr, "GraphExecutor: cannot mmap %s\n", onnxPath.c_str());
        return false;
    }

    // Memory-map external data if present
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
            // Map primary file
            auto extSize = fs::file_size(extPath);
            fprintf(stderr, "  Mapping external data: %s (%.0f MB)...\n", baseName.c_str(), extSize / 1048576.0);
            fflush(stdout);
            auto& mapping = extDataMappings_[baseName];
            mapping.open(extPath);
            // Primary file also accessible via extDataMapping_
            extDataMapping_.open(extPath);

            // Map numbered continuation files: _data_1, _data_2, ...
            for (int idx = 1; idx < 100; idx++) {
                std::string contName = stem + "_data_" + std::to_string(idx);
                std::string contPath = (dir / contName).string();
                if (!fs::exists(contPath)) break;
                auto contSize = fs::file_size(contPath);
                fprintf(stderr, "  Mapping external data: %s (%.0f MB)...\n", contName.c_str(), contSize / 1048576.0);
                fflush(stdout);
                extDataMappings_[contName].open(contPath);
            }
        }
    }

    // Parse protobuf: ModelProto → GraphProto
    std::vector<PBTensor> initTensors;
    std::vector<PBNode> pbNodes;
    std::vector<PBValueInfo> graphInputs, graphOutputs;
    int64_t extCursor = 0;

    PBReader model(onnxMapping_.data, onnxMapping_.size);
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
                        const uint8_t* extData = nullptr;
                        size_t extSize = 0;
                        auto fileIt = extDataMappings_.find(tensor.extLocation);
                        if (fileIt != extDataMappings_.end()) {
                            extData = fileIt->second.data;
                            extSize = fileIt->second.size;
                        } else if (extDataMapping_.data) {
                            extData = extDataMapping_.data;  // fallback to primary
                            extSize = extDataMapping_.size;
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
                            if (off + len <= (int64_t)extSize) {
                                tensor.rawData = extData + off;
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

    fprintf(stderr, "  Parsed: %zu initializers, %zu nodes, %zu inputs, %zu outputs\n",
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
                    weightStore_[node.outputs[0]] = std::move(gt);
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
            // Still register as a valid empty CPU tensor so it's in weightStore_
            // (other nodes may depend on this name for topo sort resolution)
            auto emptyDtype = fromOnnxDtype(t.dataType);
            GpuTensor gt;
            gt.shape = t.dims;
            gt.dtype = emptyDtype;
            gt.isCpuOnly = true;
            // Empty dims = empty tensor (0 elements), completely valid
            weightStore_[t.name] = std::move(gt);
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
            weightStore_[t.name] = std::move(gt);
            // Store initializer data from the OWNED cpuData (not from inlineDataBuf which is going away)
            auto& stored = weightStore_[t.name];
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
            weightStore_[t.name] = std::move(gt);
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
        weightStore_[t.name] = std::move(gt);

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
    fprintf(stderr, "  %d initializers uploaded, %d no-data, %zu pipelines warmed, %lldms\n",
           uploaded, nodata, warmedPipelines, (long long)ms);
    if (warmedPipelines > 0) {
        fprintf(stderr, "  Pipeline warmup: %lldms\n", (long long)warmupMs);
    }

    return true;
}

// ─── Tensor Allocation ──────────────────────────────────────────────────────

std::string g_currentOpLabel;
const char* g_currentOp = nullptr;  // for debug tracking

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
    // Align to 4 bytes for WebGPU
    size_t bufBytes = (bytes + 3) & ~(size_t)3;
    t.buffer = gpu->createBuffer("tmp", bufBytes);
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

// ─── Pipeline / Dispatch Helpers ─────────────────────────────────────────────

const CompiledPipeline& GraphExecutor::GetPipeline(const std::string& name,
                                                     const std::string& wgsl,
                                                     uint32_t numBindings) {
    return gpu->getOrCreatePipeline(name, wgsl, numBindings);
}

WGPUBindGroup GraphExecutor::MakeBindGroup(
        const CompiledPipeline& pl,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings,
        ExecutionContext* captureCtx) {
    WGPUBindGroupEntry entries[16];
    for (size_t i = 0; i < bindings.size() && i < 16; i++) {
        memset(&entries[i], 0, sizeof(WGPUBindGroupEntry));
        entries[i].binding = bindings[i].first;
        entries[i].buffer = bindings[i].second.handle;
        entries[i].offset = bindings[i].second.offset;
        entries[i].size = bindings[i].second.size;
        if (!entries[i].buffer) {
            auto dummy = gpu->createBuffer("dummy_bind", 4);
            entries[i].buffer = dummy.handle;
            entries[i].size = 4;
        }
    }
    WGPUBindGroupDescriptor d{};
    d.layout = pl.bgLayout;
    d.entryCount = (uint32_t)bindings.size();
    d.entries = entries;
    auto bg = wgpuDeviceCreateBindGroup(gpu->device, &d);

    // During fast decode capture: save bindings for replay (rebuild bind groups later)
    if (captureCtx && captureCtx->fastDecodeState_ == ExecutionContext::FastDecodeState::Capturing) {
        captureCtx->lastCapturedBindings_.clear();
        for (size_t i = 0; i < bindings.size() && i < 16; i++) {
            captureCtx->lastCapturedBindings_.push_back({
                entries[i].binding, entries[i].buffer,
                entries[i].offset, entries[i].size});
        }
    }

    return bg;
}

void GraphExecutor::Submit(const std::vector<Dispatch>& dispatches) {
    gpu->submitDispatches(dispatches);
    gpu->waitForQueue();
}

void GraphExecutor::SubmitAsync(const std::vector<Dispatch>& dispatches) {
    gpu->submitDispatches(dispatches);
}

// ─── Fast Decode: Replay & Release (moved to ExecutionContext) ─────────────

// ─── Batched submit (moved to ExecutionContext) ────────────────────────────

void GraphExecutor::Sync() {
    gpu->waitForQueue();
}

// ─── Execute Graph ──────────────────────────────────────────────────────────

void GraphExecutor::Execute(
        ExecutionContext& ctx,
        const std::unordered_map<std::string, GpuTensor*>& inputs,
        std::unordered_map<std::string, GpuTensor*>& outputs) {

    // Initialize per-session GPU context
    ctx.gpu = gpu;
    ctx.profilingEnabled = profilingEnabled;

    // Enable writeBuffer recording for fast decode capture (after input setup, inside Execute)
    if (ctx.fastDecodeState_ == ExecutionContext::FastDecodeState::Capturing && !gpu->captureWritesCb_) {
        gpu->captureWritesCb_ = [](WGPUBuffer handle, uint64_t offset,
                                    const void* data, uint64_t size, void* userCtx) {
            auto* execCtx = static_cast<ExecutionContext*>(userCtx);
            ExecutionContext::CapturedWrite w;
            w.handle = handle;
            w.offset = offset;
            w.data.resize((size_t)size);
            memcpy(w.data.data(), data, (size_t)size);
            execCtx->capturedWrites_.push_back(std::move(w));

            // Log non-trivial writes for debugging
            if (size > 64) {
                fprintf(stderr, "    [capture-write] %zu bytes to %p (op=%s)\n",
                        (size_t)size, (void*)handle,
                        g_currentOp ? g_currentOp : "?");
            }
        };
        gpu->captureWritesCtx_ = &ctx;
    }

    // Bind graph inputs into tensor store
    // With tensor plan: reuse intermediate buffers from previous Execute().
    // Without: clear stale intermediates (persistent entries preserved).
    //
    // Tensor lookup layering:
    //   ctx.tensorStore_ = per-session intermediates (cleared/restored each Execute)
    //   weightStore_     = shared persistent weights (read-only during Execute)
    //   OpContext::GetTensor() checks ctx.tensorStore_ first, then weightStore_.

    if (ctx.tensorPlanValid_) {
        // Tensor plan active: restore planned intermediate buffers instead of erasing.
        // First, remove non-planned entries from per-session store.
        for (auto it = ctx.tensorStore_.begin(); it != ctx.tensorStore_.end(); ) {
            if (ctx.tensorPlan_.count(it->first))
                ++it;
            else
                it = ctx.tensorStore_.erase(it);
        }
        // Restore planned buffers (shape/dtype will be updated by ops)
        for (auto& [name, alloc] : ctx.tensorPlan_) {
            auto& t = ctx.tensorStore_[name];
            t.buffer = alloc.buffer;
            t.shape = alloc.shape;
            t.dtype = alloc.dtype;
            t.cpuData.clear();
            t.isCpuOnly = false;
        }
    } else {
        // First Execute or plan invalidated: clear per-session intermediates
        ctx.tensorStore_.clear();
        // Start populating tensor plan and shape cache
        ctx.shapeCachePopulating_ = true;
        ctx.tensorPlan_.clear();
        ctx.nodeShapeCache_.clear();
    }


    // Save persistent tensor state (dtype + buffer) before execution.
    // Ops may convert persistent fp16 tensors to f32 in-place via
    // ensureTensorFloat32/ensureCpuBackedFloat32. We restore them at the end
    // so the next Execute() call sees the original state.
    struct SavedState { TensorDtype dtype; GPUBuffer buffer; std::vector<uint8_t> cpuData; bool isCpuOnly; };
    std::unordered_map<std::string, SavedState> savedPersistent;
    for (auto& name : persistentTensors_) {
        auto it = weightStore_.find(name);
        if (it != weightStore_.end()) {
            savedPersistent[name] = {it->second.dtype, it->second.buffer,
                                     it->second.cpuData, it->second.isCpuOnly};
        }
    }

    // Copy inputs into per-session tensor store
    for (auto& [name, tensor] : inputs) {
        ctx.tensorStore_[name] = *tensor;
        // For int64 inputs, also store as CPU data for metadata ops
        if (tensor->dtype == TensorDtype::Int64 && tensor->buffer.handle && !tensor->isCpuOnly) {
            int64_t nel = tensor->ElementCount();
            if (nel <= 1024) {
                // Readback small int64 tensor for CPU ops (ReduceSum, Sub, etc.)
                if (!tensor->cpuData.empty()) {
                    ctx.tensorStore_[name].cpuData = tensor->cpuData;
                } else {
                    auto rb = gpu->readBuffer(tensor->buffer, nel * 8);
                    ctx.tensorStore_[name].cpuData.resize(nel * 8);
                    memcpy(ctx.tensorStore_[name].cpuData.data(), rb.data(), nel * 8);
                }
            }
        }
    }

    auto& registry = GetOpRegistry();
    int executed = 0, skipped = 0;
    int totalDispatches = 0, totalCopies = 0;

    // Clear dispatch batch
    ctx.pendingDispatches_.clear();
    ctx.pendingCopies_.clear();


    // Topological sort (cached across Execute calls)
    std::vector<size_t>& execOrder = cachedExecOrder_;
    if (execOrder.empty()) {
        auto sortT0 = std::chrono::steady_clock::now();
        size_t N = graph_.nodes.size();
        std::unordered_set<std::string> available;
        for (auto& [name, _] : inputs) available.insert(name);
        for (auto& [name, _] : weightStore_) available.insert(name);

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

        // Detect fuseable patterns (one-time, cached with topo order)
        DetectFusions();
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

    // Helper: find tensor by name in per-session store first, then shared weights
    auto findTensor = [&](const std::string& name) -> GpuTensor* {
        auto it = ctx.tensorStore_.find(name);
        if (it != ctx.tensorStore_.end()) return &it->second;
        auto it2 = weightStore_.find(name);
        return (it2 != weightStore_.end()) ? &it2->second : nullptr;
    };

    // Create OpContext for op dispatch (shared graph + per-session exec)
    OpContext opCtx{*this, ctx};

    // Execute nodes in topological order
    auto execT0 = std::chrono::steady_clock::now();
    for (size_t ei = 0; ei < execOrder.size(); ei++) {
        size_t ni = execOrder[ei];
        auto& node = graph_.nodes[ni];

        // Resolve input tensor NAMES (not pointers — map may rehash during output allocation)
        std::vector<std::string> inNames;
        for (auto& inName : node.inputs) inNames.push_back(inName);

        // Prepare output tensor slots in per-session store
        std::vector<std::string> outNames;
        for (size_t oi = 0; oi < node.outputs.size(); oi++) {
            outNames.push_back(node.outputs[oi]);
            if (!node.outputs[oi].empty()) {
                // Check both stores
                auto* existing = findTensor(node.outputs[oi]);
                if (!existing || !existing->IsValid())
                    ctx.tensorStore_[node.outputs[oi]] = {};
            }
        }

        // NOW resolve pointers (map is stable after all insertions)
        std::vector<GpuTensor*> inTensors;
        for (auto& inName : inNames) {
            if (inName.empty()) {
                inTensors.push_back(nullptr);
                continue;
            }
            inTensors.push_back(findTensor(inName));
        }

        std::vector<GpuTensor*> outTensors;
        for (auto& outName : outNames) {
            if (outName.empty()) {
                outTensors.push_back(nullptr);
            } else {
                // Outputs go to per-session store
                outTensors.push_back(&ctx.tensorStore_[outName]);
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
                    ctx.pendingDispatches_.size());
            fflush(stderr);
        }

        // ─── Fused dispatch ──────────────────────────────────────────────
        // Skip nodes that are interior to a fused group (not the first node)
        if (fusedNodeIndices_.count(ni) && !fusedGroups_.count(ni)) {
            continue;  // handled by the group leader
        }

        // If this is the first node of a fused group, dispatch the fused kernel
        if (auto fIt = fusedGroups_.find(ni); fIt != fusedGroups_.end()) {
            auto& group = fIt->second;
            bool ok = true;

            // Resolve external inputs
            std::vector<GpuTensor*> extInputs;
            for (auto& name : group.externalInputs) {
                extInputs.push_back(findTensor(name));
            }

            // Determine dtype from first input
            TensorDtype dtype = TensorDtype::Float32;
            if (!extInputs.empty() && extInputs[0] && extInputs[0]->IsValid()) {
                dtype = extInputs[0]->dtype;
            }
            if (dtype != TensorDtype::Float16) dtype = TensorDtype::Float32;

            // Get primary input info
            auto* primaryIn = extInputs.empty() ? nullptr : extInputs[0];
            if (!primaryIn || !primaryIn->IsValid()) {
                ok = false;
            }

            if (ok) {
                EnsureGpu(*primaryIn);
                int64_t N = 1;
                for (auto d : primaryIn->shape) N *= d;

                // Ensure output tensor exists (last node's output) in per-session store
                auto& outTensor = ctx.tensorStore_[group.outputName];
                outTensor = AllocTensor(primaryIn->shape, dtype);

                // Build pipeline with dtype suffix
                std::string pname = group.pipelineName + dtypeSuffix(dtype);
                auto& pl = GetPipelineT(pname, group.numBindings,
                    [&group, dtype]() {
                        return group.shaderGenerator(dtype);
                    });

                // Build params: [N, N_E0, N_E1, ...]
                std::vector<uint32_t> params = {(uint32_t)N};
                for (size_t ei2 = 1; ei2 < extInputs.size(); ei2++) {
                    if (extInputs[ei2] && extInputs[ei2]->IsValid()) {
                        int64_t en = 1;
                        for (auto d : extInputs[ei2]->shape) en *= d;
                        params.push_back((uint32_t)en);
                    } else {
                        params.push_back((uint32_t)N);
                    }
                }
                size_t pbSize = std::max((size_t)16, params.size() * 4);
                auto paramBuf = ctx.getParamBuffer((uint32_t)pbSize);
                gpu->writeBuffer(paramBuf, params.data(), params.size() * 4);

                // Build bind group
                std::vector<std::pair<uint32_t, GPUBuffer>> bindings = {
                    {0, primaryIn->buffer},
                    {1, outTensor.buffer},
                    {2, paramBuf}
                };
                for (size_t ei2 = 1; ei2 < extInputs.size(); ei2++) {
                    if (extInputs[ei2] && extInputs[ei2]->IsValid()) {
                        EnsureGpu(*extInputs[ei2]);
                        bindings.push_back({(uint32_t)(2 + ei2), extInputs[ei2]->buffer});
                    }
                }

                auto bg = MakeBindGroup(pl, bindings, &ctx);
                uint32_t nwg = (uint32_t)(((N + 1) / 2 + 255) / 256);
                ctx.QueueDispatch(pl.pipeline, bg,
                    nwg, 1, 1, group.pipelineName.c_str());
                continue;
            }
            // If !ok, fall through to regular dispatch
        }

        // Dispatch op
        auto opIt = registry.find(node.opType);
        if (opIt != registry.end()) {
            // Safety: verify all input pointers are still valid before dispatch
            bool inputsOk = true;
            for (size_t ti = 0; ti < inTensors.size(); ti++) {
                if (inTensors[ti] && !inTensors[ti]->IsValid() && !node.inputs[ti].empty()) {
                    // Re-resolve in case it was updated
                    inTensors[ti] = findTensor(node.inputs[ti]);
                }
            }
            g_currentOpLabel = node.opType;
            if (!node.name.empty()) {
                g_currentOpLabel += ":";
                g_currentOpLabel += node.name;
            }
            g_currentOp = g_currentOpLabel.c_str();

            auto opT0 = ctx.profilingEnabled ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
            opIt->second(opCtx, node, inTensors, outTensors);
            if (ctx.profilingEnabled) {
                // Include GPU sync in the op time if the op flushed
                auto opT1 = std::chrono::steady_clock::now();
                double opMs = std::chrono::duration<double, std::milli>(opT1 - opT0).count();
                ctx.profileData_[node.opType] += opMs;
                ctx.profileCounts_[node.opType]++;
            }

            // Cache small int outputs to CPU for downstream metadata ops.
            // With shape cache: inject cached cpuData instead of GPU readback.
            {
                bool isCpuIntConsumer = true;
                if (node.opType == "TopK") isCpuIntConsumer = false;

                if (isCpuIntConsumer) {
                    for (size_t oi = 0; oi < outTensors.size() && oi < node.outputs.size(); oi++) {
                        auto* outTensor = outTensors[oi];
                        if (!outTensor || node.outputs[oi].empty()) continue;
                        if (outTensor->dtype != TensorDtype::Int64 && outTensor->dtype != TensorDtype::Int32) continue;
                        if (!outTensor->cpuData.empty()) continue;
                        int64_t nel = outTensor->ElementCount();
                        if (nel <= 0 || nel > 1024) continue;

                        if (ctx.shapeCacheValid_ && !ctx.shapeCachePopulating_) {
                            // Use cached CPU data — skip GPU readback entirely
                            auto cIt = ctx.nodeShapeCache_.find(node.outputs[oi]);
                            if (cIt != ctx.nodeShapeCache_.end() && cIt->second.hasCpuData) {
                                outTensor->cpuData = cIt->second.cpuData;
                                continue;
                            }
                        }

                        // Cold path: read from GPU (first Execute or cache miss)
                        if (!outTensor->buffer.handle) continue;
                        ctx.FlushPendingWork();
                        ctx.intReadbackSyncs_++;
                        size_t bytes = (size_t)nel * outTensor->DtypeSize();
                        auto rb = gpu->readBuffer(outTensor->buffer, bytes);
                        if (rb.size() >= bytes) {
                            outTensor->cpuData.resize(bytes);
                            memcpy(outTensor->cpuData.data(), rb.data(), bytes);
                        }

                        // Populate shape cache
                        if (ctx.shapeCachePopulating_) {
                            auto& cached = ctx.nodeShapeCache_[node.outputs[oi]];
                            cached.shape = outTensor->shape;
                            cached.dtype = outTensor->dtype;
                            cached.cpuData = outTensor->cpuData;
                            cached.hasCpuData = true;
                        }
                    }
                }

                // Also cache non-int output shapes/dtypes for tensor plan
                if (ctx.shapeCachePopulating_) {
                    for (size_t oi = 0; oi < outTensors.size() && oi < node.outputs.size(); oi++) {
                        auto* outTensor = outTensors[oi];
                        if (!outTensor || node.outputs[oi].empty()) continue;
                        if (!outTensor->IsValid()) continue;
                        if (ctx.nodeShapeCache_.count(node.outputs[oi])) continue;
                        auto& cached = ctx.nodeShapeCache_[node.outputs[oi]];
                        cached.shape = outTensor->shape;
                        cached.dtype = outTensor->dtype;
                        cached.hasCpuData = false;
                    }
                }
            }
            executed++;
            if (kDebugExecTrace && graph_.nodes.size() > 1000 && ei < 128) {
                fprintf(stderr, "  [exec] ei=%zu done %s pending=%zu\n",
                        ei, node.opType.c_str(), ctx.pendingDispatches_.size() + ctx.pendingCopies_.size());
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

            // Buffer lifetime tracking: decrement refcounts for later cleanup.
            // Actual release happens at end of Execute() to avoid GPU read-after-free.
            if (graph_.nodes.size() > 100) {
                for (size_t ti = 0; ti < node.inputs.size(); ti++) {
                    auto& inName = node.inputs[ti];
                    if (inName.empty()) continue;
                    tensorRefCount[inName]--;
                }
            }

            // Flush to encoder periodically (prevent huge CB on very large graphs)
            if (ctx.pendingDispatches_.size() + ctx.pendingCopies_.size() >= 256) {
                totalDispatches += (int)ctx.pendingDispatches_.size();
                totalCopies += (int)ctx.pendingCopies_.size();
                ctx.flushToEncoder();
                ctx.pendingReleases_.clear();
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
        auto* srcTensor = findTensor(name);
        if (srcTensor && srcTensor->IsValid()) {
            size_t outBytes = srcTensor->ByteSize();
            if (outBytes == 0) outBytes = 4;
            if (tensor && tensor->buffer.handle && srcTensor->buffer.handle &&
                tensor->buffer.handle != srcTensor->buffer.handle &&
                tensor->buffer.size >= outBytes && !srcTensor->isCpuOnly) {
                tensor->shape = srcTensor->shape;
                tensor->dtype = srcTensor->dtype;
                tensor->isCpuOnly = false;
                tensor->cpuData.clear();
                ctx.QueueCopy(srcTensor->buffer, 0, tensor->buffer, 0, outBytes);
            } else {
                *tensor = *srcTensor;
            }
        } else {
            fprintf(stderr, "  [exec] WARNING: output '%s' not found or invalid buf=%p\n",
                    name.c_str(), srcTensor ? (void*)srcTensor->buffer.handle : nullptr);
            fflush(stderr);
        }
    }

    // Append readback copy after output copies — this ensures logits data
    // is in the source buffer before the readback copy executes.
    // Goes into pendingCopies_ so it's included in the final flushToEncoder CB.
    if (ctx.readbackSize_ > 0 && ctx.readbackSrc_.handle && ctx.readbackDst_.handle) {
        ctx.pendingCopies_.push_back({ctx.readbackSrc_, 0, ctx.readbackDst_, 0, ctx.readbackSize_});
        ctx.readbackSize_ = 0;
        ctx.readbackSrc_ = {}; ctx.readbackDst_ = {};
    }

    totalDispatches += (int)ctx.pendingDispatches_.size();
    totalCopies += (int)ctx.pendingCopies_.size();


    fprintf(stderr, "  [exec] %d/%zu ops executed, %d unimplemented, %d dispatches, %d copies, %d syncs (%d from int-readback), bufs=%d (pool=%d)\n",
            executed, graph_.nodes.size(), skipped, totalDispatches, totalCopies, ctx.flushCount_,
            ctx.intReadbackSyncs_, gpu->createBufferCount, gpu->poolHitCount);
    gpu->createBufferCount = 0;
    gpu->poolHitCount = 0;
    // Print sync sources on first call
    static int printSyncCount = 0;
    if (printSyncCount < 2 && !ctx.flushSources_.empty()) {
        std::vector<std::pair<std::string, int>> sorted(ctx.flushSources_.begin(), ctx.flushSources_.end());
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
        fprintf(stderr, "  [sync sources] %d total:\n", ctx.flushCount_);
        for (auto& [op, cnt] : sorted)
            fprintf(stderr, "    %-50s %d\n", op.c_str(), cnt);
        printSyncCount++;
    }
    ctx.flushCount_ = 0;
    ctx.intReadbackSyncs_ = 0;
    ctx.flushSources_.clear();

    // Submit all remaining batched GPU work
    ctx.flushToEncoder();
    gpu->waitForQueue();

    // Release non-output, non-persistent intermediate buffers
    // (GPU work is done, safe to free)
    // With tensor plan: keep buffers alive for reuse. Without: release normally.
    // In fast decode capture, keep ALL buffers alive — they're referenced by captured bind groups.
    if (ctx.fastDecodeState_ != ExecutionContext::FastDecodeState::Capturing) {
        if (ctx.tensorPlanValid_) {
            // Tensor plan active: DON'T release planned buffers — they'll be reused.
            // Just clear non-planned entries from per-session store.
            for (auto it = ctx.tensorStore_.begin(); it != ctx.tensorStore_.end(); ) {
                if (ctx.tensorPlan_.count(it->first))
                    ++it;
                else
                    it = ctx.tensorStore_.erase(it);
            }
        } else if (ctx.shapeCachePopulating_) {
            // First Execute: record the tensor plan, then keep buffers alive.
            std::set<WGPUBuffer> keepHandles;
            for (auto& [name, tensor] : outputs)
                if (tensor && tensor->buffer.handle) keepHandles.insert(tensor->buffer.handle);
            for (auto& name : persistentTensors_) {
                auto it = weightStore_.find(name);
                if (it != weightStore_.end() && it->second.buffer.handle)
                    keepHandles.insert(it->second.buffer.handle);
            }

            // Record non-persistent intermediate buffers into tensor plan
            for (auto& [name, tensor] : ctx.tensorStore_) {
                if (!tensor.buffer.handle) continue;
                if (keepHandles.count(tensor.buffer.handle)) continue;
                ctx.tensorPlan_[name] = {tensor.buffer, tensor.buffer.size,
                                         tensor.shape, tensor.dtype};
            }

            // Mark caches as valid for next Execute
            ctx.shapeCacheValid_ = true;
            ctx.shapeCachePopulating_ = false;
            ctx.tensorPlanValid_ = true;

            fprintf(stderr, "  [warm-exec] shape cache: %zu entries, tensor plan: %zu buffers\n",
                    ctx.nodeShapeCache_.size(), ctx.tensorPlan_.size());

            // Don't release any buffers — keep them for reuse
            for (auto it = ctx.tensorStore_.begin(); it != ctx.tensorStore_.end(); ) {
                if (ctx.tensorPlan_.count(it->first))
                    ++it;
                else
                    it = ctx.tensorStore_.erase(it);
            }
        } else {
            // No tensor plan, not populating: normal cleanup
            std::set<WGPUBuffer> keepHandles;
            for (auto& [name, tensor] : outputs)
                if (tensor && tensor->buffer.handle) keepHandles.insert(tensor->buffer.handle);
            for (auto& name : persistentTensors_) {
                auto it = weightStore_.find(name);
                if (it != weightStore_.end() && it->second.buffer.handle)
                    keepHandles.insert(it->second.buffer.handle);
            }

            std::set<WGPUBuffer> released;
            for (auto it = ctx.tensorStore_.begin(); it != ctx.tensorStore_.end(); ) {
                if (it->second.buffer.handle &&
                    keepHandles.find(it->second.buffer.handle) == keepHandles.end() &&
                    released.find(it->second.buffer.handle) == released.end()) {
                    released.insert(it->second.buffer.handle);
                    gpu->releaseBuffer(it->second.buffer);
                }
                it = ctx.tensorStore_.erase(it);
            }
        }
    } else {
        // Capture mode: just erase tensor store entries but DON'T release buffers
        ctx.tensorStore_.clear();
    }

    // Restore persistent tensors that may have been modified during execution
    for (auto& [name, saved] : savedPersistent) {
        auto it = weightStore_.find(name);
        if (it != weightStore_.end() &&
            (it->second.dtype != saved.dtype || it->second.buffer.handle != saved.buffer.handle)) {
            it->second.dtype = saved.dtype;
            it->second.buffer = saved.buffer;
            it->second.cpuData = std::move(saved.cpuData);
            it->second.isCpuOnly = saved.isCpuOnly;
        }
    }

    // Print profiling report
    if (ctx.profilingEnabled && !ctx.profileData_.empty()) {
        double totalMs = 0;
        for (auto& [op, ms] : ctx.profileData_) totalMs += ms;

        // Sort by time descending
        std::vector<std::pair<std::string, double>> sorted(ctx.profileData_.begin(), ctx.profileData_.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        fprintf(stderr, "\n  ┌─ Profile (%d ops, %.1fms total) ─────────────────────┐\n",
                executed, totalMs);
        fprintf(stderr, "  │ %-25s %7s %5s %7s │\n", "Op", "ms", "cnt", "%");
        fprintf(stderr, "  ├───────────────────────────────────────────────────────┤\n");
        for (auto& [op, ms] : sorted) {
            int cnt = ctx.profileCounts_[op];
            double pct = totalMs > 0 ? 100.0 * ms / totalMs : 0;
            fprintf(stderr, "  │ %-25s %7.1f %5d %6.1f%% │\n", op.c_str(), ms, cnt, pct);
        }
        fprintf(stderr, "  └───────────────────────────────────────────────────────┘\n");
        fflush(stderr);
        ctx.profileData_.clear();
        ctx.profileCounts_.clear();

        // Print sync sources
        if (!ctx.flushSources_.empty()) {
            std::vector<std::pair<std::string, int>> sortedSync(ctx.flushSources_.begin(), ctx.flushSources_.end());
            std::sort(sortedSync.begin(), sortedSync.end(),
                      [](auto& a, auto& b) { return a.second > b.second; });
            fprintf(stderr, "\n  GPU syncs (%d total):\n", ctx.flushCount_);
            for (auto& [op, cnt] : sortedSync) {
                fprintf(stderr, "    %-40s %d\n", op.c_str(), cnt);
            }
            ctx.flushSources_.clear();
        }
    }
}

// enableGpuProfiling() and printGpuProfileReport() are now in execution_context.cpp

// ─── Lifetime Interval Analysis ─────────────────────────────────────────────

const std::vector<LifetimeInterval>& GraphExecutor::computeLifetimeIntervals() {
    if (cachedExecOrder_.empty()) return cachedLifetimeIntervals_;

    if (cachedExecOrder_.size() == lifetimeExecOrderSize_ &&
        !cachedLifetimeIntervals_.empty()) {
        return cachedLifetimeIntervals_;
    }

    std::set<std::string> immortal;
    for (auto& [name, _] : weightStore_) immortal.insert(name);
    for (auto& [name, _] : graph_.initializers) immortal.insert(name);
    for (auto& inp : graph_.inputs) immortal.insert(inp.name);
    for (auto& out : graph_.outputs) immortal.insert(out.name);

    std::unordered_map<std::string, int> firstUse, lastUse;

    for (size_t step = 0; step < cachedExecOrder_.size(); ++step) {
        auto& node = graph_.nodes[cachedExecOrder_[step]];
        int s = static_cast<int>(step);

        for (auto& name : node.inputs) {
            if (name.empty() || immortal.count(name)) continue;
            if (firstUse.find(name) == firstUse.end()) firstUse[name] = s;
            lastUse[name] = s;
        }
        for (auto& name : node.outputs) {
            if (name.empty() || immortal.count(name)) continue;
            if (firstUse.find(name) == firstUse.end()) firstUse[name] = s;
            lastUse[name] = s;
        }
    }

    cachedLifetimeIntervals_.clear();
    cachedLifetimeIntervals_.reserve(firstUse.size());
    for (auto& [name, first] : firstUse) {
        cachedLifetimeIntervals_.push_back({name, first, lastUse[name], 0});
    }

    lifetimeExecOrderSize_ = cachedExecOrder_.size();
    return cachedLifetimeIntervals_;
}
