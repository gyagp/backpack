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
            case 15: dataLoc = (int)r.readVarint(); break;
            case 13: { // raw_data or misplaced external_data
                uint64_t len = r.readVarint();
                const uint8_t* start = r.data;
                // Check if this is a StringStringEntryProto (external_data encoded as field 13)
                if (len > 4 && len < 200 && start[0] == 0x0A) {
                    PBReader sub(start, (size_t)len);
                    std::string key, value;
                    bool isEntry = true;
                    while (!sub.eof()) {
                        auto [f2, w2] = sub.readTag();
                        if (f2 == 1 && w2 == 2) key = sub.readString();
                        else if (f2 == 2 && w2 == 2) value = sub.readString();
                        else { isEntry = false; break; }
                    }
                    if (isEntry && !key.empty()) {
                        if (key == "location") t.extLocation = value;
                        else if (key == "offset") t.extOffset = std::stoll(value);
                        else if (key == "length") t.extLength = std::stoll(value);
                        r.data += len;
                        break;
                    }
                }
                t.rawData = start;
                t.rawSize = (size_t)len;
                r.data += len;
                break;
            }
            case 14: // external_data or data_location
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
};

static PBAttribute parseAttr(PBReader& r) {
    PBAttribute a;
    while (!r.eof()) {
        auto [f, w] = r.readTag();
        switch (f) {
            case 1: a.name = r.readString(); break;
            case 4: a.type = (int)r.readVarint(); break;
            case 5: { uint32_t bits = r.readFixed32(); memcpy(&a.f, &bits, 4); break; }
            case 6: a.i = (int64_t)r.readVarint(); break;
            case 7: a.s = r.readString(); break;
            case 8: // ints (repeated)
                if (w == 2) { auto sub = r.readLengthDelimited(); while (!sub.eof()) a.ints.push_back((int64_t)sub.readVarint()); }
                else a.ints.push_back((int64_t)r.readVarint());
                break;
            case 9: // floats (repeated)
                if (w == 2) { auto sub = r.readLengthDelimited(); while (!sub.eof()) { uint32_t bits = sub.readFixed32(); float fv; memcpy(&fv, &bits, 4); a.floats.push_back(fv); } }
                else { uint32_t bits = r.readFixed32(); float fv; memcpy(&fv, &bits, 4); a.floats.push_back(fv); }
                break;
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
        }
        graph_.nodes.push_back(std::move(node));
    }

    // Upload initializers to GPU
    int uploaded = 0;
    for (auto& t : initTensors) {
        if (!t.rawData || t.rawSize == 0) continue;
        auto dtype = fromOnnxDtype(t.dataType);
        GpuTensor gt;
        gt.shape = t.dims;
        gt.dtype = dtype;
        gt.buffer = gpu->createBuffer(t.name, t.rawSize);
        gpu->writeBuffer(gt.buffer, t.rawData, t.rawSize);
        tensorStore_[t.name] = std::move(gt);

        // Also record in graph initializers (for metadata)
        graph_.initializers[t.name] = {t.rawData, t.rawSize, dtype, t.dims};
        uploaded++;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("  %d initializers uploaded, %lldms\n", uploaded, (long long)ms);

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
    // Batch into pending — submitted all at once at end of Execute()
    pendingDispatches_.insert(pendingDispatches_.end(),
                              dispatches.begin(), dispatches.end());
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
    }

    auto& registry = GetOpRegistry();
    int executed = 0, skipped = 0;

    // Clear dispatch batch
    pendingDispatches_.clear();

    // Execute nodes in order (ONNX guarantees topological order)
    for (size_t ni = 0; ni < graph_.nodes.size(); ni++) {
        auto& node = graph_.nodes[ni];

        if (ni < 10 || ni % 100 == 0) {
            fprintf(stderr, "  [exec] node %zu/%zu: %s\n",
                    ni, graph_.nodes.size(), node.opType.c_str());
            fflush(stderr);
        }

        // Resolve input tensors
        std::vector<GpuTensor*> inTensors;
        for (auto& inName : node.inputs) {
            if (inName.empty()) {
                inTensors.push_back(nullptr);  // optional input
                continue;
            }
            auto it = tensorStore_.find(inName);
            if (it != tensorStore_.end()) {
                inTensors.push_back(&it->second);
            } else {
                inTensors.push_back(nullptr);
            }
        }

        // Prepare output tensor pointers
        std::vector<GpuTensor*> outTensors(node.outputs.size(), nullptr);
        // Pre-allocate slots in the store
        for (size_t oi = 0; oi < node.outputs.size(); oi++) {
            if (!node.outputs[oi].empty()) {
                tensorStore_[node.outputs[oi]] = {};
                outTensors[oi] = &tensorStore_[node.outputs[oi]];
            }
        }

        // Dispatch op
        auto opIt = registry.find(node.opType);
        if (opIt != registry.end()) {
            opIt->second(*this, node, inTensors, outTensors);
            executed++;
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

    fprintf(stderr, "  [exec] %d/%zu ops executed, %d unimplemented, %zu dispatches\n",
            executed, graph_.nodes.size(), skipped, pendingDispatches_.size());

    // Submit ALL batched GPU dispatches in one command buffer
    if (!pendingDispatches_.empty()) {
        gpu->submitOnly(pendingDispatches_, false);
        gpu->waitForQueue();
        pendingDispatches_.clear();
    }
}
