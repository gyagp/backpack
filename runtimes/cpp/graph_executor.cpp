/**
 * graph_executor.cpp — ONNX graph executor on WebGPU.
 *
 * Parses ONNX protobuf, uploads initializers to GPU, then executes
 * nodes in topological order by dispatching registered op kernels.
 */

#include "graph_executor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <unordered_set>

namespace fs = std::filesystem;

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
    std::string extPath = onnxPath + ".data";  // model.onnx.data
    // Also check without _data suffix pattern
    if (!fs::exists(extPath)) {
        auto dir = fs::path(onnxPath).parent_path();
        auto stem = fs::path(onnxPath).filename().string();
        extPath = (dir / (stem + "_data")).string();
    }
    if (fs::exists(extPath)) {
        auto extSize = fs::file_size(extPath);
        printf("  Loading external data: %.0f MB...\n", extSize / 1048576.0);
        fflush(stdout);
        FILE* ef = fopen(extPath.c_str(), "rb");
        if (ef) {
            externalData_.resize(extSize);
            size_t read = 0;
            while (read < extSize) {
                size_t chunk = std::min((size_t)(256*1024*1024), extSize - read);
                size_t n = fread(externalData_.data() + read, 1, chunk, ef);
                if (n == 0) break;
                read += n;
            }
            fclose(ef);
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
                    if (!tensor.extLocation.empty() && !externalData_.empty()) {
                        int64_t off = (tensor.extOffset >= 0) ? tensor.extOffset : extCursor;
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
                        if (off + len <= (int64_t)externalData_.size()) {
                            tensor.rawData = externalData_.data() + off;
                            tensor.rawSize = (size_t)len;
                            extCursor = off + len;
                            extCursor = (extCursor + 63) & ~63LL;
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
                    if (node.outputs[0].find("INT64") != std::string::npos) {
                        auto& stored = tensorStore_[node.outputs[0]];
                        fprintf(stderr, "  [pre-store] '%s' shape=[", node.outputs[0].c_str());
                        for (size_t i = 0; i < stored.shape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)stored.shape[i]);
                        fprintf(stderr, "] cpuData=%zu isCpu=%d\n", stored.cpuData.size(), stored.isCpuOnly);
                        fflush(stderr);
                    }
                }
            }
        }
        graph_.nodes.push_back(std::move(node));
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
            graph_.initializers[t.name] = {t.rawData, t.rawSize, dtype, t.dims};
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
        bool convertToF32 = (dtype == TensorDtype::Float16 && !isLargeWeight && !isScale);

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
        uploaded++;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("  %d initializers uploaded, %d no-data, %lldms\n", uploaded, nodata, (long long)ms);

    return true;
}

// ─── Tensor Allocation ──────────────────────────────────────────────────────

GpuTensor GraphExecutor::AllocTensor(const std::vector<int64_t>& shape,
                                      TensorDtype dtype) {
    GpuTensor t;
    t.shape = shape;
    t.dtype = dtype;
    size_t bytes = t.ByteSize();
    if (bytes == 0) bytes = 4;  // minimum
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

    // Topological sort using Kahn's algorithm (O(N+E), not O(N²))
    std::vector<size_t> execOrder;
    {
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
        if (execOrder.size() < N)
            for (size_t ni = 0; ni < N; ni++)
                if (depCount[ni] > 0) execOrder.push_back(ni);

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

        if (ei < 100 || ei % 10 == 0 || ei > 498) {
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
            opIt->second(*this, node, inTensors, outTensors);
            executed++;

            // Debug: readback outputs to trace zero propagation
            // Debug readback disabled for production
            if (false && ei < 200 && outTensors[0] && outTensors[0]->IsValid() && outTensors[0]->buffer.handle
                && node.opType != "Constant" && node.opType != "Cast" && node.opType != "Reshape"
                && node.opType != "Shape" && node.opType != "Unsqueeze" && node.opType != "Squeeze"
                && node.opType != "Flatten" && node.opType != "Gather") {
                // Flush ALL pending work (dispatches + copies)
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
                    fprintf(stderr, "  [dbg] ei=%zu %s out=[%.4f, %.4f, %.4f, %.4f]\n",
                            ei, node.opType.c_str(), vals[0], vals[1], vals[2], vals[3]);
                    fflush(stderr);
                }
            }

            // Buffer release: alias-aware, only for large models
            if (graph_.nodes.size() > 800) {
                for (auto& inName : node.inputs) {
                    if (inName.empty()) continue;
                    auto rc = --tensorRefCount[inName];
                    if (rc <= 0) {
                        auto it = tensorStore_.find(inName);
                        if (it != tensorStore_.end() && it->second.buffer.handle && !it->second.isCpuOnly) {
                            bool aliased = false;
                            WGPUBuffer h = it->second.buffer.handle;
                            for (auto& [oname, otensor] : tensorStore_) {
                                if (oname != inName && otensor.buffer.handle == h) {
                                    auto rcIt = tensorRefCount.find(oname);
                                    if (rcIt != tensorRefCount.end() && rcIt->second > 0) {
                                        aliased = true;
                                        break;
                                    }
                                }
                            }
                            if (!aliased) {
                                gpu->releaseBuffer(it->second.buffer);
                                for (auto& [oname, otensor] : tensorStore_) {
                                    if (otensor.buffer.handle == h) {
                                        otensor.buffer = {nullptr, 0};
                                    }
                                }
                            } else {
                                it->second.buffer = {nullptr, 0};
                            }
                        }
                    }
                }
            }

            // Submit pending work periodically (with sync to prevent TDR)
            if (pendingDispatches_.size() + pendingCopies_.size() >= 8) {
                totalDispatches += (int)pendingDispatches_.size();
                totalCopies += (int)pendingCopies_.size();
                if (!pendingDispatches_.empty())
                    gpu->submitOnly(pendingDispatches_, false);
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
                gpu->waitForQueue();  // Sync to prevent TDR
                pendingDispatches_.clear();
                pendingCopies_.clear();
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
            *tensor = it->second;
        }
    }

    totalDispatches += (int)pendingDispatches_.size();
    totalCopies += (int)pendingCopies_.size();
    fprintf(stderr, "  [exec] %d/%zu ops executed, %d unimplemented, %d dispatches, %d copies\n",
            executed, graph_.nodes.size(), skipped, totalDispatches, totalCopies);

    // Submit all remaining batched GPU work
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

    // Release non-output intermediate buffers (GPU work is done, safe to free)
    {
        std::set<WGPUBuffer> outputHandles;
        for (auto& [name, tensor] : outputs)
            if (tensor && tensor->buffer.handle) outputHandles.insert(tensor->buffer.handle);

        std::set<WGPUBuffer> released;
        for (auto& [name, tensor] : tensorStore_) {
            if (tensor.buffer.handle &&
                outputHandles.find(tensor.buffer.handle) == outputHandles.end() &&
                released.find(tensor.buffer.handle) == released.end()) {
                released.insert(tensor.buffer.handle);
                gpu->releaseBuffer(tensor.buffer);
            }
            tensor.buffer = {nullptr, 0};
        }
        tensorStore_.clear();
    }
}
