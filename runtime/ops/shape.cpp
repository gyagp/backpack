/**
 * ops/shape.cpp — Shape manipulation ONNX ops.
 * Uses embedded WGSL kernels from runtime/kernels/shared/.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include "../wgsl_template.h"
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static constexpr bool kDebugModOp = false;

static std::vector<int64_t> preferredIntOutputShape(const GpuTensor* A, const GpuTensor* B,
                                                    int64_t N_A, int64_t N_B) {
    const auto& preferredShape = (N_A >= N_B) ? A->shape : B->shape;
    if (!preferredShape.empty()) return preferredShape;
    return (std::max<int64_t>(N_A, N_B) <= 1) ? std::vector<int64_t>{}
                                               : std::vector<int64_t>{std::max<int64_t>(N_A, N_B)};
}

static bool loadTensorBytes(OpContext& ex, GpuTensor* t,
                            const std::string& name,
                            size_t neededBytes,
                            std::vector<uint8_t>& out) {
    out.clear();
    if (!t) return false;

    auto copyFrom = [&](const uint8_t* src, size_t available) -> bool {
        if (!src || available < neededBytes) return false;
        out.resize(neededBytes);
        memcpy(out.data(), src, neededBytes);
        return true;
    };

    if (copyFrom(t->cpuData.data(), t->cpuData.size())) return true;

    if (!name.empty()) {
        if (auto* init = ex.GetInitData(name); init && copyFrom(init->data, init->size))
            return true;
    }

    if (!t->buffer.handle || neededBytes == 0 || neededBytes > t->buffer.size)
        return false;

    ex.FlushPendingWork();
    out = ex.getGpu()->readBuffer(t->buffer, neededBytes);
    if (out.size() < neededBytes) {
        out.clear();
        return false;
    }
    if (t->cpuData.size() < neededBytes) {
        t->cpuData.resize(neededBytes);
        memcpy(t->cpuData.data(), out.data(), neededBytes);
    }
    return true;
}

static bool readTensorInt64Values(OpContext& ex, GpuTensor* t,
                                  const std::string& name,
                                  std::vector<int64_t>& out) {
    out.clear();
    if (!t) return false;

    int64_t nel = t->ElementCount();
    if (nel < 0 || nel > 1024) return false;

    std::vector<uint8_t> raw;
    if (t->dtype == TensorDtype::Int64) {
        if (!loadTensorBytes(ex, t, name, (size_t)nel * 8, raw)) return false;
        out.resize((size_t)nel);
        memcpy(out.data(), raw.data(), raw.size());
        return true;
    }

    if (t->dtype == TensorDtype::Int32) {
        if (!loadTensorBytes(ex, t, name, (size_t)nel * 4, raw)) return false;
        out.resize((size_t)nel);
        auto* src = reinterpret_cast<const int32_t*>(raw.data());
        for (int64_t i = 0; i < nel; i++) out[(size_t)i] = src[i];
        return true;
    }

    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero-copy ops (no GPU work, just change shape metadata)
// ═══════════════════════════════════════════════════════════════════════════

static void opReshape(OpContext& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* shape = in.size() > 1 ? in[1] : nullptr;
    if (!data || !data->IsValid()) return;

    std::vector<int64_t> newShape;
    if (shape && n.inputs.size() > 1) {
        readTensorInt64Values(ex, shape, n.inputs[1], newShape);
    }
    if (newShape.empty() && shape) {
        // Shape input exists but empty — fallback to passthrough
    }
    if (newShape.empty()) {
        *out[0] = *data; return;
    }

    int64_t totalIn = tensorNel(data);
    int64_t known = 1; int inferIdx = -1;
    for (int i = 0; i < (int)newShape.size(); i++) {
        if (newShape[i] == 0 && i < (int)data->shape.size()) newShape[i] = data->shape[i];
        if (newShape[i] == -1) inferIdx = i; else known *= newShape[i];
    }
    if (inferIdx >= 0 && known > 0) newShape[inferIdx] = totalIn / known;

    // Debug: log unpatchification reshapes
    if (n.name == "/Reshape_13" || n.name == "/Reshape_14" || n.name == "/Reshape_5" || n.name == "/Reshape_6") {
        fprintf(stderr, "  [reshape-dbg] %s: in=[", n.name.c_str());
        for (size_t i = 0; i < data->shape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)data->shape[i]);
        fprintf(stderr, "] -> [");
        for (size_t i = 0; i < newShape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)newShape[i]);
        fprintf(stderr, "] total=%lld\n", (long long)totalIn);
        fflush(stderr);
    }

    *out[0] = *data;
    out[0]->shape = newShape;
}

static void opSqueeze(OpContext& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1]) {
        readTensorInt64Values(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", axes);
    } else if (n.attrIntLists.count("axes")) axes = n.attrIntLists.at("axes");
    std::vector<int64_t> ns;
    for (int i = 0; i < (int)data->shape.size(); i++) {
        bool sq = axes.empty() ? (data->shape[i]==1) : false;
        for (auto a : axes) { if ((a<0?a+(int64_t)data->shape.size():a)==i) { sq=true; break; } }
        if (!sq) ns.push_back(data->shape[i]);
    }
    *out[0] = *data; out[0]->shape = ns;
}

static void opUnsqueeze(OpContext& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1]) {
        if (in[1]->isCpuOnly || !in[1]->cpuData.empty()) {
            readTensorInt64Values(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", axes);
        } else if (auto* init = ex.GetInitData(n.inputs.size() > 1 ? n.inputs[1] : ""); init && init->data) {
            readTensorInt64Values(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", axes);
        } else if (in[1]->buffer.handle) {
            readTensorInt64Values(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", axes);
        }
    } else if (n.attrIntLists.count("axes")) axes = n.attrIntLists.at("axes");
    if (axes.empty()) {
        // No axes → passthrough
        *out[0] = *data;
        return;
    }
    int ndim = (int)data->shape.size() + (int)axes.size();
    for (auto& a : axes) if (a < 0) a += ndim;
    std::sort(axes.begin(), axes.end());
    std::vector<int64_t> ns(ndim); int si = 0;
    for (int i = 0; i < ndim; i++) {
        if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
            ns[i] = 1;
        } else {
            if (si < (int)data->shape.size()) {
                ns[i] = data->shape[si++];
            } else {
                ns[i] = 1;
            }
        }
    }
    *out[0] = *data; out[0]->shape = ns;
}

static void opFlatten(OpContext& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    int64_t axis = n.GetInt("axis", 1); if (axis < 0) axis += data->shape.size();
    int64_t d0=1, d1=1;
    for (int i=0; i<axis; i++) d0 *= data->shape[i];
    for (int i=(int)axis; i<(int)data->shape.size(); i++) d1 *= data->shape[i];
    *out[0] = *data; out[0]->shape = {d0, d1};
}

// ═══════════════════════════════════════════════════════════════════════════
// Metadata ops (CPU compute, immediate GPU upload)
// ═══════════════════════════════════════════════════════════════════════════

static void opShape(OpContext& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data) return;
    int ndim = (int)data->shape.size();
    *out[0] = ex.AllocCpuTensor({ndim}, TensorDtype::Int64, data->shape.data(), ndim*8);
}

static void opConstant(OpContext& ex, const OnnxGraphNode& n,
                        const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Check if output was already pre-stored (tensor-valued 'value' attribute)
    if (out[0] && out[0]->IsValid()) return;

    if (n.attrIntLists.count("value_ints")) {
        auto& v = n.attrIntLists.at("value_ints");
        *out[0] = ex.AllocCpuTensor({(int64_t)v.size()}, TensorDtype::Int64, v.data(), v.size()*8);
    } else if (n.attrInts.count("value_int")) {
        int64_t v = n.GetInt("value_int");
        *out[0] = ex.AllocCpuTensor({}, TensorDtype::Int64, &v, 8);
    } else if (n.attrFloats.count("value_float")) {
        float v = n.GetFloat("value_float");
        *out[0] = ex.AllocCpuTensor({}, TensorDtype::Float32, &v, 4);
    } else {
        float z = 0; *out[0] = ex.AllocCpuTensor({}, TensorDtype::Float32, &z, 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Gather — uses embedded gather kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opGather(OpContext& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; auto* indices = in[1];
    if (!data || !indices || !data->IsValid() || !indices->IsValid()) return;
    int64_t axis = n.GetInt("axis", 0);
    int64_t nIdx = tensorNel(indices);

    std::vector<uint8_t> dataRaw;
    std::vector<int64_t> indexVals;
    bool smallMetadataData =
        (data->dtype == TensorDtype::Int64 || data->dtype == TensorDtype::Int32) &&
        data->ElementCount() <= 1024;
    bool haveDataRaw = false;
    bool needCpuData = (data->isCpuOnly || !data->cpuData.empty() ||
                        ex.GetInitData(n.inputs.empty() ? "" : n.inputs[0]) ||
                        smallMetadataData);
    // For non-zero axis gathers, force GPU readback if needed
    if (needCpuData) {
        haveDataRaw = loadTensorBytes(ex, data, n.inputs.empty() ? "" : n.inputs[0],
                                      data->ByteSize(), dataRaw);
    }
    bool haveIndexVals = readTensorInt64Values(ex, indices,
                                               n.inputs.size() > 1 ? n.inputs[1] : "",
                                               indexVals);

    if (haveDataRaw && haveIndexVals && axis == 0 && !data->shape.empty()) {
        int64_t inner = 1;
        for (size_t i=1; i<data->shape.size(); i++) inner *= data->shape[i];
        size_t sliceB = (size_t)(inner * data->DtypeSize());
        std::vector<int64_t> os;
        for (auto d : indices->shape) if (d > 0) os.push_back(d);
        for (size_t i=1; i<data->shape.size(); i++) os.push_back(data->shape[i]);
        if (os.empty()) for (size_t i=1; i<data->shape.size(); i++) os.push_back(data->shape[i]);
        std::vector<uint8_t> od(nIdx * sliceB);
        for (int64_t i=0; i<nIdx; i++) {
            int64_t idx = indexVals[(size_t)i]; if (idx<0) idx += data->shape[0];
            if (idx>=0 && (size_t)((idx+1)*sliceB) <= dataRaw.size())
                memcpy(od.data()+i*sliceB, dataRaw.data()+idx*sliceB, sliceB);
        }
        *out[0] = ex.AllocCpuTensor(os, data->dtype, od.data(), od.size());
        return;
    }

    // GPU Gather only handles axis=0. For axis!=0, force CPU readback.
    if (axis != 0 && !haveDataRaw && haveIndexVals && data->buffer.handle) {
        size_t readBytes = std::min(data->ByteSize(), data->buffer.size);
        if (readBytes >= data->DtypeSize() && readBytes <= 8000000) {
            ex.FlushPendingWork();
            dataRaw = ex.getGpu()->readBuffer(data->buffer, readBytes);
            haveDataRaw = (dataRaw.size() >= readBytes);
        }
    }

    // CPU path for non-zero axis gather
    if (haveDataRaw && haveIndexVals && axis != 0 && !data->shape.empty()) {
        int ndim = (int)data->shape.size();
        int64_t normAxis = axis;
        if (normAxis < 0) normAxis += ndim;
        size_t elemSize = data->DtypeSize();

        std::vector<int64_t> os;
        for (int i = 0; i < ndim; i++) {
            if (i == (int)normAxis) {
                for (auto d : indices->shape) if (d > 0) os.push_back(d);
            } else {
                os.push_back(data->shape[i]);
            }
        }
        if (os.empty()) os.push_back(1);

        int64_t outerSize = 1, innerSize = 1;
        for (int i = 0; i < (int)normAxis; i++) outerSize *= data->shape[i];
        for (int i = (int)normAxis + 1; i < ndim; i++) innerSize *= data->shape[i];

        int64_t totalOut = 1;
        for (auto d : os) totalOut *= d;

        std::vector<uint8_t> od((size_t)totalOut * elemSize, 0);

        for (int64_t outer = 0; outer < outerSize; outer++) {
            for (int64_t idxI = 0; idxI < nIdx; idxI++) {
                int64_t idx = indexVals[(size_t)idxI];
                if (idx < 0) idx += data->shape[normAxis];
                for (int64_t inner = 0; inner < innerSize; inner++) {
                    size_t srcOff = ((size_t)(outer * data->shape[normAxis] + idx) * (size_t)innerSize + (size_t)inner) * elemSize;
                    size_t dstOff = ((size_t)(outer * nIdx + idxI) * (size_t)innerSize + (size_t)inner) * elemSize;
                    if (srcOff + elemSize <= dataRaw.size() && dstOff + elemSize <= od.size()) {
                        memcpy(od.data() + dstOff, dataRaw.data() + srcOff, elemSize);
                    }
                }
            }
        }

        *out[0] = ex.AllocCpuTensor(os, data->dtype, od.data(), od.size());
        ex.EnsureGpu(*out[0]);
        return;
    }

    // GPU path: dispatch gather kernel (axis=0 only)
    ex.EnsureGpu(*data);
    ex.EnsureGpu(*indices);

    int64_t inner = 1;
    for (size_t i=1; i<data->shape.size(); i++) inner *= data->shape[i];
    size_t elemSize = data->DtypeSize();
    uint32_t sliceSizeU32 = (uint32_t)((inner * elemSize + 3) / 4);
    uint32_t dataStrideU32 = sliceSizeU32;

    std::vector<int64_t> os;
    for (auto d : indices->shape) if (d > 0) os.push_back(d);
    for (size_t i=1; i<data->shape.size(); i++) os.push_back(data->shape[i]);
    if (os.empty()) for (size_t i=1; i<data->shape.size(); i++) os.push_back(data->shape[i]);

    *out[0] = ex.AllocTensor(os, data->dtype);

    // Convert int64 indices to int32 on GPU if needed
    // For now, assume indices are small and uploadable
    GPUBuffer idxBuf = indices->buffer;
    if (indices->dtype == TensorDtype::Int64) {
        // Need int32 indices for the kernel — upload converted
        // For small index tensors, do CPU conversion
        if (nIdx <= 1024 && haveIndexVals) {
            std::vector<int32_t> i32(nIdx);
            for (int64_t i = 0; i < nIdx; i++) i32[(size_t)i] = (int32_t)indexVals[(size_t)i];
            idxBuf = ex.getGpu()->createBuffer("gather_idx32", nIdx * 4);
            ex.getGpu()->writeBuffer(idxBuf, i32.data(), nIdx * 4);
        }
    }

    uint32_t total = (uint32_t)(nIdx * sliceSizeU32);
    uint32_t params[4] = {(uint32_t)nIdx, sliceSizeU32, dataStrideU32, 0};
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipelineT("gather", 4, []() { return std::string(WGSL_GATHER); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, data->buffer}, {1, idxBuf}, {2, out[0]->buffer}, {3, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (total + 255) / 256, 1, 1, "gather");
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Concat (CopyBufferToBuffer, no sync)
// ═══════════════════════════════════════════════════════════════════════════

static void opConcat(OpContext& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    int64_t axis = n.GetInt("axis", 0);
    std::vector<GpuTensor*> validIn;
    for (size_t i = 0; i < in.size(); i++) {
        auto* t = in[i];
        if (!t) continue;
        if (t->IsValid()) validIn.push_back(t);
    }
    if (validIn.empty()) return;

    int ndim = (int)validIn[0]->shape.size();
    if (ndim == 0) {
        // Scalar tensors: treat as 1D
        for (auto* t : validIn) if (t->shape.empty()) t->shape = {1};
        ndim = 1;
        axis = 0;
    }
    if (axis < 0) axis += ndim;
    if (axis < 0) axis = 0;
    if (axis >= ndim) axis = ndim - 1;
    auto outShape = validIn[0]->shape;
    int64_t totalOnAxis = 0;
    for (auto* t : validIn)
        if (axis < (int64_t)t->shape.size()) totalOnAxis += t->shape[axis];
    if (axis < (int64_t)outShape.size()) outShape[axis] = totalOnAxis;

    // CPU fast path: small int64 tensors (shape computation metadata)
    // Keep these on CPU to avoid GPU buffer pool handle aliasing issues
    bool allSmallInt64 = (validIn[0]->dtype == TensorDtype::Int64 && totalOnAxis <= 32);
    for (auto* t : validIn) {
        if (t->dtype != TensorDtype::Int64 || tensorNel(t) > 32) {
            allSmallInt64 = false; break;
        }
    }
    if (allSmallInt64 && axis == 0) {
        // Concatenate on CPU — no GPU sync needed
        int64_t totalElements = 0;
        for (auto* t : validIn) totalElements += tensorNel(t);
        std::vector<uint8_t> cpuBuf(totalElements * 8, 0);
        size_t offset = 0;
        for (auto* t : validIn) {
            std::vector<int64_t> vals;
            if (readTensorInt64Values(ex, t, "", vals) && !vals.empty()) {
                size_t bytes = vals.size() * sizeof(int64_t);
                memcpy(cpuBuf.data() + offset, vals.data(), bytes);
                offset += bytes;
            } else {
                offset += (size_t)tensorNel(t) * 8;
            }
        }
        *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, cpuBuf.data(), cpuBuf.size());
        return;
    }

    for (auto* t : validIn) ex.EnsureGpu(*t);

    // Normalize dtypes: if inputs have mixed f32/f16, convert all to f32.
    // For pure fp16 non-axis-0 with 2 inputs, the GPU concat kernel handles it directly.
    bool hasF32 = false, hasF16 = false;
    for (auto* t : validIn) {
        if (t->dtype == TensorDtype::Float32) hasF32 = true;
        if (t->dtype == TensorDtype::Float16) hasF16 = true;
    }
    bool canUseF16GpuConcat = (hasF16 && !hasF32 && validIn.size() == 2 &&
                                axis > 0 && ex.getGpu()->supportsShaderF16);
    bool needF16ToF32 = (hasF32 && hasF16) ||
                         (hasF16 && axis > 0 && !canUseF16GpuConcat);
    if (needF16ToF32 && ex.getGpu()->supportsShaderF16) {
        // Convert f16 inputs to f32 using cast kernel
        for (auto* t : validIn) {
            if (t->dtype != TensorDtype::Float16) continue;
            int64_t nel = tensorNel(t);
            if (nel <= 0) continue;
            GpuTensor f32t = ex.AllocTensor(t->shape, TensorDtype::Float32);
            uint32_t params[4] = {(uint32_t)nel, 0, 0, 0};
            auto paramBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(paramBuf, params, 16);
            auto& pl = ex.GetPipelineT("cast_f16_to_f32", 3, []() { return std::string(WGSL_CAST_F16_TO_F32); });
            auto bg = ex.MakeBindGroup(pl, {
                {0, t->buffer}, {1, f32t.buffer}, {2, paramBuf}});
            ex.SubmitAsync({{pl.pipeline, bg,
                (uint32_t)((nel + 255) / 256), 1, 1, "concat_cast_f16_f32"}});
            *t = f32t;
        }
        // Don't flush here — GPU kernels execute in order.
        // The f32 concat GPU kernel will see the cast results.
        // Only flush if we need the CPU copy path (axis=0).
    }

    // Filter out tensors with no GPU buffer after EnsureGpu
    std::vector<GpuTensor*> gpuIn;
    for (auto* t : validIn)
        if (t->buffer.handle) gpuIn.push_back(t);

    if (gpuIn.empty()) return;

    *out[0] = ex.AllocTensor(outShape, gpuIn[0]->dtype);

    // For axis=0 or 1D tensors, simple byte concatenation
    if (axis == 0 || ndim <= 1) {
        // Flush any pending cast dispatches before buffer copies
        if (needF16ToF32 && !ex.exec.pendingDispatches_.empty())
            ex.FlushPendingWork();
        size_t offset = 0;
        for (auto* t : gpuIn) {
            size_t bytes = t->ByteSize();
            if (bytes == 0) bytes = t->buffer.size;
            if (bytes > t->buffer.size) bytes = t->buffer.size;
            if (bytes > 0)
                ex.QueueCopy(t->buffer, 0, out[0]->buffer, offset, bytes);
            offset += bytes;
        }
    } else {
        // General axis concatenation: copy slabs
        size_t elemSize = gpuIn[0]->DtypeSize();
        int64_t innerSize = 1;
        for (int i = (int)axis + 1; i < ndim; i++) innerSize *= outShape[i];
        int64_t outerSize = 1;
        for (int i = 0; i < (int)axis; i++) outerSize *= outShape[i];

        // WebGPU requires copy offsets and sizes to be multiples of 4.
        // For fp16 (2 bytes per element), slab copies may not be 4-byte aligned.
        // Fall back to CPU copy in that case.
        bool needsCpuCopy = (elemSize < 4 && (innerSize * elemSize) % 4 != 0);
        // Also check if any slab's offset would be misaligned
        if (!needsCpuCopy && elemSize < 4) {
            int64_t tmpOff = 0;
            for (auto* t : gpuIn) {
                if ((tmpOff * innerSize * elemSize) % 4 != 0 ||
                    (t->shape[axis] * innerSize * elemSize) % 4 != 0) {
                    needsCpuCopy = true;
                    break;
                }
                tmpOff += t->shape[axis];
            }
        }

        // For non-axis-0 concat, use appropriate copy method
        // GPU kernel for 2-input concat (avoids CPU readback)
        if (gpuIn.size() == 2 && gpuIn[0]->dtype == gpuIn[1]->dtype &&
            (gpuIn[0]->dtype == TensorDtype::Float32 ||
             gpuIn[0]->dtype == TensorDtype::Float16)) {
            TensorDtype dtype = gpuIn[0]->dtype;
            int64_t outNel = 1;
            for (auto d : outShape) outNel *= d;

            uint32_t params[4] = {(uint32_t)outNel, (uint32_t)gpuIn[0]->shape[axis],
                                   (uint32_t)outShape[axis], (uint32_t)innerSize};
            auto paramBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(paramBuf, params, 16);

            std::string pname = "concat_2t" + std::string(dtypeSuffix(dtype));
            auto& pl = ex.GetPipelineT(pname, 4, [dtype]() {
                return instantiateTemplate(WGSL_CONCAT_2INPUT_T, dtype);
            });
            auto bg = ex.MakeBindGroup(pl, {
                {0, gpuIn[0]->buffer}, {1, gpuIn[1]->buffer},
                {2, out[0]->buffer}, {3, paramBuf}});
            uint32_t nwg = (uint32_t)(((outNel + 1) / 2 + 255) / 256);
            ex.QueueDispatch(pl.pipeline, bg,
                nwg, 1, 1, "concat");
            return;
        }
        // CPU fallback for non-axis-0 concat
        // Read all inputs, interleave on CPU, upload result.
        {
            int64_t outNel = 1;
            for (auto d : outShape) outNel *= d;
            size_t outBytes = (size_t)(outNel * elemSize);
            std::vector<uint8_t> outBuf(outBytes, 0);

            // Ensure all input data is on GPU and ready
            ex.FlushPendingWork();

            int64_t dstAxisOff = 0;
            for (auto* t : gpuIn) {
                int64_t srcAxisSize = t->shape[axis];
                size_t srcBytes = t->ByteSize();
                auto rb = ex.getGpu()->readBuffer(t->buffer, srcBytes);

                for (int64_t o = 0; o < outerSize; o++) {
                    size_t srcOff = (size_t)(o * srcAxisSize * innerSize * elemSize);
                    size_t dstOff = (size_t)((o * totalOnAxis + dstAxisOff) * innerSize * elemSize);
                    size_t copyLen = (size_t)(srcAxisSize * innerSize * elemSize);
                    if (srcOff + copyLen <= rb.size() && dstOff + copyLen <= outBuf.size())
                        memcpy(outBuf.data() + dstOff, rb.data() + srcOff, copyLen);
                }
                dstAxisOff += srcAxisSize;
            }
            ex.getGpu()->writeBuffer(out[0]->buffer, outBuf.data(), outBytes);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Transpose — uses embedded transpose kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opTranspose(OpContext& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    ex.EnsureGpu(*data);

    std::vector<int64_t> perm;
    if (n.attrIntLists.count("perm")) perm = n.attrIntLists.at("perm");
    else for (int i=(int)data->shape.size()-1; i>=0; i--) perm.push_back(i);

    // ONNX requires perm rank to match data rank. If they don't match,
    // pad perm with identity mappings for leading/trailing dims.
    auto dataShape = data->shape;
    if ((int)perm.size() < (int)dataShape.size()) {
        // Fewer perm dims — pad perm with identity dims for trailing dimensions
        while ((int)perm.size() < (int)dataShape.size())
            perm.push_back((int64_t)perm.size());
    } else if ((int)perm.size() > (int)dataShape.size()) {
        // More perm dims: pad data shape with leading 1s
        while ((int)dataShape.size() < (int)perm.size())
            dataShape.insert(dataShape.begin(), 1);
    }

    // Validate perm values are within range
    for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] < 0 || perm[i] >= (int64_t)dataShape.size()) {
            *out[0] = *data;
            return;
        }
    }

    std::vector<int64_t> outShape(perm.size());
    for (size_t i=0; i<perm.size(); i++) {
        if (perm[i] >= (int64_t)dataShape.size()) {
            *out[0] = *data;
            return;
        }
        outShape[i] = dataShape[perm[i]];
    }

    int64_t nel = 1;
    for (auto d : dataShape) nel *= d;

    // Debug: log unpatchification transpose
    if (n.name == "/Transpose_2" || n.name == "/Transpose_1") {
        fprintf(stderr, "  [transpose-dbg] %s: in=[", n.name.c_str());
        for (size_t i = 0; i < dataShape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)dataShape[i]);
        fprintf(stderr, "] perm=[");
        for (size_t i = 0; i < perm.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)perm[i]);
        fprintf(stderr, "] -> [");
        for (size_t i = 0; i < outShape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)outShape[i]);
        fprintf(stderr, "] nel=%lld dtype=%d\n", (long long)nel, (int)data->dtype);
        fflush(stderr);
    }

    size_t elemSize = data->DtypeSize();
    int ndim = (int)dataShape.size();

    std::vector<int64_t> inStrides(ndim);
    inStrides[ndim-1] = 1;
    for (int i=ndim-2; i>=0; i--) inStrides[i] = inStrides[i+1] * dataShape[i+1];

    std::vector<int64_t> outStrides64(ndim);
    outStrides64[ndim-1] = 1;
    for (int i=ndim-2; i>=0; i--) outStrides64[i] = outStrides64[i+1] * outShape[i+1];

    // Check if transpose is effectively a no-op (dimensions being swapped are size 1)
    bool isNoop = true;
    for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] != (int64_t)i && dataShape[i] != 1 && dataShape[perm[i]] != 1) {
            isNoop = false;
            break;
        }
    }
    // Also check if data has <= 1 element total
    if (nel <= 1) isNoop = true;

    if (isNoop) {
        *out[0] = *data;
        out[0]->shape = outShape;
        return;
    }

    const bool preferCpuMetadata =
        (data->dtype == TensorDtype::Int64 || data->dtype == TensorDtype::Int32) &&
        data->ElementCount() <= 1024;

    // The embedded transpose kernel operates on u32 lanes. Use it for element
    // sizes that map cleanly to u32 units, but keep small integer metadata on
    // CPU to avoid shape corruption in downstream ops like Pad.
    if ((elemSize == 4 || elemSize == 8 || elemSize == 2) && !preferCpuMetadata) {
        if (!data->buffer.handle) { *out[0] = *data; out[0]->shape = outShape; return; }
        if (!data->buffer.handle || data->buffer.size < (size_t)(nel * elemSize)) {
            *out[0] = *data;
            out[0]->shape = outShape;
            return;
        }

        *out[0] = ex.AllocTensor(outShape, data->dtype);

        uint32_t ostride = 1;
        std::vector<uint32_t> outStrides(ndim), permInStrides(ndim);
        for (int i = ndim - 1; i >= 0; i--) {
            outStrides[i] = ostride;
            ostride *= (uint32_t)outShape[i];
            permInStrides[i] = (uint32_t)inStrides[perm[i]];
        }

        if (elemSize == 2 || elemSize == 4) {
            // fp16 or f32: use templated kernel with element-level read/write
            uint32_t nelU = (uint32_t)nel;
            std::vector<uint32_t> params(4 + 2 * ndim, 0);
            params[0] = nelU;
            params[1] = (uint32_t)ndim;
            for (int i = 0; i < ndim; i++) {
                params[4 + i] = outStrides[i];
                params[4 + ndim + i] = permInStrides[i];
            }
            auto paramBuf = ex.getGpu()->createBuffer("tr_p", params.size() * 4);
            ex.getGpu()->writeBuffer(paramBuf, params.data(), params.size() * 4);

            auto& pl = ex.GetPipelineT("transpose" + std::string(dtypeSuffix(data->dtype)), 3,
                [&]() { return instantiateTemplate(WGSL_TRANSPOSE_T, data->dtype); });
            auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
            ex.QueueDispatch(pl.pipeline, bg, (nelU + 255) / 256, 1, 1, "transpose");
        } else {
            // i64: existing u32-level kernel (2 u32 per i64 element)
            uint32_t elemsU32 = (uint32_t)(nel * 2);
            std::vector<uint32_t> params(4 + 2 * ndim, 0);
            params[0] = elemsU32;
            params[1] = (uint32_t)ndim;
            for (int i = 0; i < ndim; i++) {
                params[4 + i] = outStrides[i] * 2;
                params[4 + ndim + i] = permInStrides[i] * 2;
            }
            auto paramBuf = ex.getGpu()->createBuffer("tr_p", params.size() * 4);
            ex.getGpu()->writeBuffer(paramBuf, params.data(), params.size() * 4);

            auto& pl = ex.GetPipelineT("transpose", 3, []() { return std::string(WGSL_TRANSPOSE); });
            auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
            ex.QueueDispatch(pl.pipeline, bg, (elemsU32 + 255) / 256, 1, 1, "transpose");
        }
        return;
    }

    {
        size_t totalBytes = (size_t)nel * elemSize;
        std::vector<uint8_t> inputBytes;
        if (data->cpuData.size() >= totalBytes) {
            inputBytes.assign(data->cpuData.begin(), data->cpuData.begin() + totalBytes);
        } else {
            ex.FlushPendingWork();
            auto raw = ex.getGpu()->readBuffer(data->buffer, totalBytes);
            if (raw.size() < totalBytes) return;
            inputBytes.assign(raw.begin(), raw.begin() + totalBytes);
        }

        std::vector<uint8_t> outputBytes(totalBytes, 0);
        std::vector<int64_t> coords(ndim, 0);
        for (int64_t outIndex = 0; outIndex < nel; outIndex++) {
            int64_t rem = outIndex;
            for (int i = 0; i < ndim; i++) {
                coords[i] = rem / outStrides64[i];
                rem %= outStrides64[i];
            }
            int64_t inIndex = 0;
            for (int i = 0; i < ndim; i++) {
                inIndex += coords[i] * inStrides[perm[i]];
            }
            memcpy(outputBytes.data() + (size_t)outIndex * elemSize,
                   inputBytes.data() + (size_t)inIndex * elemSize,
                   elemSize);
        }

        *out[0] = ex.AllocCpuTensor(outShape, data->dtype, outputBytes.data(), outputBytes.size());
        return;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simple / pass-through ops
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// GPU Slice — uses embedded slice kernel
// ═══════════════════════════════════════════════════════════════════════════

static int64_t readInt64FromTensor(OpContext& ex, const GpuTensor* t, const std::string& name) {
    const uint8_t* p = nullptr;
    if (t && t->isCpuOnly && !t->cpuData.empty()) p = t->cpuData.data();
    else if (t && !t->cpuData.empty()) p = t->cpuData.data();
    else if (auto* init = ex.GetInitData(name); init && init->data) p = init->data;
    if (p) { int64_t v; memcpy(&v, p, 8); return v; }
    return 0;
}

static void opSlice(OpContext& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid() || data->shape.empty()) return;

    int ndim = (int)data->shape.size();

    // Read starts, ends, axes, steps from inputs (all int64)
    auto readI64Vec = [&](int inputIdx) -> std::vector<int64_t> {
        if (inputIdx >= (int)in.size() || !in[inputIdx]) return {};
        std::vector<int64_t> v;
        readTensorInt64Values(ex, in[inputIdx],
                              inputIdx < (int)n.inputs.size() ? n.inputs[inputIdx] : "",
                              v);
        return v;
    };

    auto starts = readI64Vec(1);
    auto ends = readI64Vec(2);
    auto axes = readI64Vec(3);
    auto steps = readI64Vec(4);

    if (starts.empty() || ends.empty()) {
        *out[0] = *data;
        return;
    }

    // If no axes specified, default to 0..len(starts)-1
    if (axes.empty()) {
        for (int i = 0; i < (int)starts.size(); i++) axes.push_back(i);
    }
    // If no steps specified, default to 1
    if (steps.empty()) {
        steps.resize(starts.size(), 1);
    }

    auto normalizeSliceBounds = [](int64_t start, int64_t end, int64_t step, int64_t dimSize,
                                   int64_t& normStart, int64_t& normEnd, int64_t& outLen) {
        if (step == 0) step = 1;
        if (step > 0) {
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;
            normStart = std::max((int64_t)0, std::min(start, dimSize));
            normEnd = std::max((int64_t)0, std::min(end, dimSize));
            outLen = (normEnd <= normStart) ? 0 : ((normEnd - normStart + step - 1) / step);
        } else {
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;
            normStart = std::max((int64_t)-1, std::min(start, dimSize - 1));
            normEnd = std::max((int64_t)-1, std::min(end, dimSize - 1));
            outLen = (normStart <= normEnd) ? 0 : ((normStart - normEnd - 1) / (-step) + 1);
        }
    };

    if (ndim == 1 && data->ElementCount() <= 1024 &&
        (data->dtype == TensorDtype::Int64 || data->dtype == TensorDtype::Int32)) {
        std::vector<int64_t> values;
        if (readTensorInt64Values(ex, data, n.inputs.empty() ? "" : n.inputs[0], values) && !values.empty()) {
            int64_t dimSize = data->shape[0];
            int64_t axis = axes.empty() ? 0 : axes[0];
            if (axis < 0) axis += ndim;
            if (axis == 0) {
                int64_t start = starts.empty() ? 0 : starts[0];
                int64_t end = ends.empty() ? dimSize : ends[0];
                int64_t step = steps.empty() ? 1 : steps[0];
                int64_t sliceLen = 0;
                normalizeSliceBounds(start, end, step, dimSize, start, end, sliceLen);

                std::vector<int64_t> sliced;
                sliced.reserve((size_t)std::max<int64_t>(sliceLen, 0));
                if (step > 0) {
                    for (int64_t i = start; i < end; i += step) sliced.push_back(values[(size_t)i]);
                } else {
                    for (int64_t i = start; i > end; i += step) sliced.push_back(values[(size_t)i]);
                }

                if (data->dtype == TensorDtype::Int32) {
                    std::vector<int32_t> sliced32(sliced.size());
                    for (size_t i = 0; i < sliced.size(); i++) sliced32[i] = (int32_t)sliced[i];
                    *out[0] = ex.AllocCpuTensor({(int64_t)sliced32.size()}, TensorDtype::Int32,
                                                sliced32.data(), sliced32.size() * sizeof(int32_t));
                } else {
                    *out[0] = ex.AllocCpuTensor({(int64_t)sliced.size()}, TensorDtype::Int64,
                                                sliced.data(), sliced.size() * sizeof(int64_t));
                }
                return;
            }
        }
    }

    // Compute output shape
    auto outShape = data->shape;
    std::vector<int64_t> startVals(ndim, 0), stepVals(ndim, 1);
    for (int i = 0; i < (int)axes.size(); i++) {
        int64_t axis = axes[i];
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim) continue;

        int64_t dimSize = data->shape[axis];
        int64_t start = starts[i], end = ends[i], step = steps[i];
        int64_t sliceLen = 0;
        normalizeSliceBounds(start, end, step, dimSize, start, end, sliceLen);

        outShape[axis] = std::max<int64_t>(sliceLen, 0);
        startVals[axis] = start;
        stepVals[axis] = step;
    }

    int64_t totalOut = 1;
    for (auto d : outShape) totalOut *= d;

    // Identity shortcut: if output shape equals input shape, just alias
    if (outShape == data->shape) {
        *out[0] = *data;
        return;
    }

    if (totalOut <= 0) {
        *out[0] = ex.AllocTensor({0}, data->dtype);
        return;
    }

    if ((data->dtype == TensorDtype::Int64 || data->dtype == TensorDtype::Int32) &&
        data->ElementCount() <= 1024) {
        std::vector<int64_t> values;
        if (readTensorInt64Values(ex, data, n.inputs.empty() ? "" : n.inputs[0], values)) {
            std::vector<int64_t> inStrides64(ndim, 1), outStrides64(ndim, 1);
            for (int i = ndim - 2; i >= 0; i--) {
                inStrides64[i] = inStrides64[i + 1] * data->shape[i + 1];
                outStrides64[i] = outStrides64[i + 1] * outShape[i + 1];
            }

            std::vector<int64_t> sliced((size_t)totalOut);
            std::vector<int64_t> coords(ndim, 0);
            for (int64_t outIndex = 0; outIndex < totalOut; outIndex++) {
                int64_t rem = outIndex;
                for (int i = 0; i < ndim; i++) {
                    coords[i] = rem / outStrides64[i];
                    rem %= outStrides64[i];
                }
                int64_t inIndex = 0;
                for (int i = 0; i < ndim; i++) {
                    inIndex += (startVals[i] + coords[i] * stepVals[i]) * inStrides64[i];
                }
                sliced[(size_t)outIndex] = values[(size_t)inIndex];
            }

            if (data->dtype == TensorDtype::Int32) {
                std::vector<int32_t> sliced32((size_t)totalOut);
                for (int64_t i = 0; i < totalOut; i++) sliced32[(size_t)i] = (int32_t)sliced[(size_t)i];
                *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int32,
                                            sliced32.data(), sliced32.size() * sizeof(int32_t));
            } else {
                *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64,
                                            sliced.data(), sliced.size() * sizeof(int64_t));
            }
            return;
        }
    }

    ex.EnsureGpu(*data);
    size_t elemSize = data->DtypeSize();

    // For fp16 data with non-trivial steps, the GPU kernel (array<u32>) can't
    // handle element-level strided access correctly because u32 = 2 fp16.
    // Fall back to CPU byte-level slice.
    bool hasNonTrivialStep = false;
    for (auto s : stepVals) if (s != 1) { hasNonTrivialStep = true; break; }

    if (hasNonTrivialStep && elemSize == 2 && totalOut <= 1000000) {
        ex.FlushPendingWork();
        size_t inBytes = (size_t)data->ElementCount() * elemSize;
        auto rb = ex.getGpu()->readBuffer(data->buffer, inBytes);
        if (rb.size() >= inBytes) {
            std::vector<int64_t> inStrides64(ndim, 1), outStrides64(ndim, 1);
            for (int i = ndim-2; i >= 0; i--) {
                inStrides64[i] = inStrides64[i+1] * data->shape[i+1];
                outStrides64[i] = outStrides64[i+1] * outShape[i+1];
            }
            std::vector<uint8_t> outBytes((size_t)totalOut * elemSize, 0);
            for (int64_t outIdx = 0; outIdx < totalOut; outIdx++) {
                int64_t rem = outIdx;
                int64_t inIdx = 0;
                for (int i = 0; i < ndim; i++) {
                    int64_t coord = rem / outStrides64[i];
                    rem %= outStrides64[i];
                    inIdx += (startVals[i] + coord * stepVals[i]) * inStrides64[i];
                }
                memcpy(outBytes.data() + (size_t)outIdx * elemSize,
                       rb.data() + (size_t)inIdx * elemSize, elemSize);
            }
            *out[0] = ex.AllocTensor(outShape, data->dtype);
            ex.getGpu()->writeBuffer(out[0]->buffer, outBytes.data(), outBytes.size());
            return;
        }
    }

    // For simple cases (single contiguous slice), use buffer copy
    // For general case, use GPU kernel
    *out[0] = ex.AllocTensor(outShape, data->dtype);

    // GPU copy path: for step=1 slices, use buffer copies (no CPU readback)
    bool allStepsOne = true;
    for (auto s : stepVals) if (s != 1) { allStepsOne = false; break; }

    if (allStepsOne && data->buffer.handle) {
        // Compute slab-based copy offsets
        int64_t sliceAxis = -1;
        for (int i = 0; i < ndim; i++) {
            if (startVals[i] != 0 || outShape[i] != data->shape[i]) {
                sliceAxis = i; break;
            }
        }

        if (sliceAxis >= 0) {
            int64_t innerSize = elemSize;
            for (int i = (int)sliceAxis + 1; i < ndim; i++) innerSize *= outShape[i];
            int64_t outerCount = 1;
            for (int i = 0; i < (int)sliceAxis; i++) outerCount *= outShape[i];

            int64_t srcInner = elemSize;
            for (int i = (int)sliceAxis + 1; i < ndim; i++) srcInner *= data->shape[i];
            int64_t srcStride = data->shape[sliceAxis] * srcInner;
            int64_t dstStride = outShape[sliceAxis] * innerSize;

            // Check 4-byte alignment for GPU copy
            bool aligned = (innerSize % 4 == 0) &&
                            ((startVals[sliceAxis] * srcInner) % 4 == 0);

            if (aligned && outerCount <= 16) {
                // Few outer iterations — use GPU buffer copies directly
                for (int64_t o = 0; o < outerCount; o++) {
                    uint64_t srcOff = (uint64_t)(o * srcStride + startVals[sliceAxis] * srcInner);
                    uint64_t dstOff = (uint64_t)(o * dstStride);
                    uint64_t copySize = (uint64_t)(outShape[sliceAxis] * innerSize);
                    ex.QueueCopy(data->buffer, srcOff, out[0]->buffer, dstOff, copySize);
                }
                return;
            }

            // Many outer iterations — use GPU slice kernel
            if (data->dtype == TensorDtype::Float32 || data->dtype == TensorDtype::Float16) {
                if (ndim == 3) {
                    // Templated GPU slice kernel: each thread copies a pair of elements
                    static const char* SLICE_3D_T = R"WGSL(
${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> src: array<${T}>;
@group(0) @binding(1) var<storage, read_write> dst: array<${T}>;
@group(0) @binding(2) var<storage, read> _p: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p[0];
    let base = gid.x * 2u;
    if (base >= N) { return; }
    let d2 = _p[1]; let d1 = _p[2]; let d0 = _p[3];
    let s2 = _p[4]; let s1 = _p[5]; let s0 = _p[6];
    let id1 = _p[7]; let id2 = _p[8];

    let o0_0 = base / (d1 * d2);
    let rem0 = base % (d1 * d2);
    let o1_0 = rem0 / d2;
    let o2_0 = rem0 % d2;
    let src_idx0 = (o0_0 + s0) * id1 * id2 + (o1_0 + s1) * id2 + (o2_0 + s2);
    let v0 = t_read(&src, src_idx0);

    var v1: f32 = 0.0;
    if (base + 1u < N) {
        let idx1 = base + 1u;
        let o0_1 = idx1 / (d1 * d2);
        let rem1 = idx1 % (d1 * d2);
        let o1_1 = rem1 / d2;
        let o2_1 = rem1 % d2;
        let src_idx1 = (o0_1 + s0) * id1 * id2 + (o1_1 + s1) * id2 + (o2_1 + s2);
        v1 = t_read(&src, src_idx1);
    }
    t_write2(&dst, base, v0, v1);
}
)WGSL";
                    TensorDtype dtype = data->dtype;
                    uint32_t outN = (uint32_t)totalOut;
                    uint32_t params[12] = {
                        outN,
                        (uint32_t)outShape[2], (uint32_t)outShape[1], (uint32_t)outShape[0],
                        (uint32_t)startVals[2], (uint32_t)startVals[1], (uint32_t)startVals[0],
                        (uint32_t)data->shape[1], (uint32_t)data->shape[2], 0, 0, 0
                    };
                    auto pBuf = ex.getParamBuffer(48);
                    ex.getGpu()->writeBuffer(pBuf, params, 48);
                    std::string pname = "slice_3d_t" + std::string(dtypeSuffix(dtype));
                    auto& pl = ex.GetPipelineT(pname, 3, [dtype]() {
                        return instantiateTemplate(SLICE_3D_T, dtype);
                    });
                    auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, pBuf}});
                    uint32_t nwg = (uint32_t)(((outN + 1) / 2 + 255) / 256);
                    ex.QueueDispatch(pl.pipeline, bg,
                        nwg, 1, 1, "slice_3d");
                    return;
                }
            }
        }
    }

    // CPU path for small tensors
    if (totalOut <= 4000000 && elemSize <= 4) {
        ex.FlushPendingWork();
        size_t inBytes = (size_t)data->ElementCount() * elemSize;
        auto rb = ex.getGpu()->readBuffer(data->buffer, inBytes);
        if (rb.size() >= inBytes) {
            std::vector<int64_t> inStrides64(ndim, 1), outStrides64(ndim, 1);
            for (int i = ndim-2; i >= 0; i--) {
                inStrides64[i] = inStrides64[i+1] * data->shape[i+1];
                outStrides64[i] = outStrides64[i+1] * outShape[i+1];
            }
            std::vector<uint8_t> outBuf((size_t)totalOut * elemSize, 0);
            for (int64_t outIdx = 0; outIdx < totalOut; outIdx++) {
                int64_t rem = outIdx;
                int64_t inIdx = 0;
                for (int i = 0; i < ndim; i++) {
                    int64_t coord = rem / outStrides64[i];
                    rem %= outStrides64[i];
                    inIdx += (startVals[i] + coord * stepVals[i]) * inStrides64[i];
                }
                memcpy(outBuf.data() + (size_t)outIdx * elemSize,
                       rb.data() + (size_t)inIdx * elemSize, elemSize);
            }
            ex.getGpu()->writeBuffer(out[0]->buffer, outBuf.data(), outBuf.size());
            return;
        }
    }

    bool simpleContiguousSlice = true;
    int sliceAxis = -1;
    for (int i = 0; i < ndim; i++) {
        bool changed = (outShape[i] != data->shape[i]) || (startVals[i] != 0) || (stepVals[i] != 1);
        if (changed) {
            if (sliceAxis >= 0) {
                simpleContiguousSlice = false;
                break;
            }
            sliceAxis = i;
        }
    }
    if (simpleContiguousSlice && sliceAxis == 0 && stepVals[0] == 1) {
        size_t copyBytes = (size_t)totalOut * elemSize;
        size_t srcOffset = (size_t)startVals[0] * (size_t)(tensorNel(data) / data->shape[0]) * elemSize;
        // WebGPU requires 4-byte aligned copies
        if (copyBytes % 4 == 0 && srcOffset % 4 == 0 && copyBytes > 0) {
            ex.QueueCopy(data->buffer, srcOffset, out[0]->buffer, 0, copyBytes);
            return;
        }
    }

    // For fp16 data, the GPU kernel operates on u32 which doesn't handle
    // element-level slicing correctly. Fall back to CPU.
    if (elemSize == 2 && totalOut <= 4000000) {
        ex.FlushPendingWork();
        size_t inBytes = (size_t)data->ElementCount() * elemSize;
        auto rb = ex.getGpu()->readBuffer(data->buffer, inBytes);
        if (rb.size() >= inBytes) {
            std::vector<int64_t> inStrides64(ndim, 1), outStrides64(ndim, 1);
            for (int i = ndim-2; i >= 0; i--) {
                inStrides64[i] = inStrides64[i+1] * data->shape[i+1];
                outStrides64[i] = outStrides64[i+1] * outShape[i+1];
            }
            std::vector<uint8_t> outBytes((size_t)totalOut * elemSize, 0);
            for (int64_t outIdx = 0; outIdx < totalOut; outIdx++) {
                int64_t rem = outIdx;
                int64_t inIdx = 0;
                for (int i = 0; i < ndim; i++) {
                    int64_t coord = rem / outStrides64[i];
                    rem %= outStrides64[i];
                    inIdx += (startVals[i] + coord * stepVals[i]) * inStrides64[i];
                }
                memcpy(outBytes.data() + (size_t)outIdx * elemSize,
                       rb.data() + (size_t)inIdx * elemSize, elemSize);
            }
            *out[0] = ex.AllocTensor(outShape, data->dtype);
            ex.getGpu()->writeBuffer(out[0]->buffer, outBytes.data(), outBytes.size());
            return;
        }
    }

    // For fp16 and f32: use templated kernel with element-level access
    if (elemSize == 2 || elemSize == 4) {
        // Strides in element units (not u32)
        std::vector<uint32_t> inStridesE(ndim), outStridesE(ndim);
        uint32_t se = 1;
        for (int i = ndim-1; i >= 0; i--) { outStridesE[i] = se; se *= (uint32_t)outShape[i]; }
        se = 1;
        for (int i = ndim-1; i >= 0; i--) { inStridesE[i] = se; se *= (uint32_t)data->shape[i]; }

        std::vector<uint32_t> params(4 + 4 * ndim, 0);
        params[0] = (uint32_t)totalOut;
        params[1] = (uint32_t)ndim;
        for (int i = 0; i < ndim; i++) {
            params[4 + i] = outStridesE[i];
            params[4 + ndim + i] = inStridesE[i];
            params[4 + 2*ndim + i] = (uint32_t)startVals[i];
            params[4 + 3*ndim + i] = (uint32_t)(stepVals[i] < 0 ? (uint32_t)(int32_t)stepVals[i] : (uint32_t)stepVals[i]);
        }
        auto paramBuf = ex.getGpu()->createBuffer("slice_p", params.size() * 4);
        ex.getGpu()->writeBuffer(paramBuf, params.data(), params.size() * 4);

        auto& pl = ex.GetPipelineT("slice" + std::string(dtypeSuffix(data->dtype)), 3,
            [&]() { return instantiateTemplate(WGSL_SLICE_T, data->dtype); });
        auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            ((uint32_t)totalOut + 255) / 256, 1, 1, "slice");
        return;
    }

    // Compute strides
    std::vector<uint32_t> inStrides(ndim), outStrides(ndim);
    uint32_t s = 1;
    for (int i = ndim-1; i >= 0; i--) { inStrides[i] = s; s *= (uint32_t)data->shape[i]; }
    s = 1;
    for (int i = ndim-1; i >= 0; i--) { outStrides[i] = s; s *= (uint32_t)outShape[i]; }

    // Scale strides for element size (kernel works in u32 units)
    uint32_t u32PerElem = (uint32_t)((elemSize + 3) / 4);
    for (int i = ndim-1; i >= 0; i--) {
        if (i == ndim-1) {
            inStrides[i] *= u32PerElem;
            outStrides[i] *= u32PerElem;
        } else {
            inStrides[i] = inStrides[i+1] * (uint32_t)data->shape[i+1];
            outStrides[i] = outStrides[i+1] * (uint32_t)outShape[i+1];
        }
    }
    // Recalculate properly
    inStrides[ndim-1] = u32PerElem;
    for (int i = ndim-2; i >= 0; i--) inStrides[i] = inStrides[i+1] * (uint32_t)data->shape[i+1];
    outStrides[ndim-1] = u32PerElem;
    for (int i = ndim-2; i >= 0; i--) outStrides[i] = outStrides[i+1] * (uint32_t)outShape[i+1];

    uint32_t totalU32 = (uint32_t)(totalOut * u32PerElem);

    // Build params: total, ndim, pad, pad, out_strides[ndim], in_strides[ndim], starts[ndim], steps[ndim]
    std::vector<uint32_t> params(4 + 4 * ndim, 0);
    params[0] = totalU32;
    params[1] = (uint32_t)ndim;
    for (int i = 0; i < ndim; i++) {
        params[4 + i] = outStrides[i];
        params[4 + ndim + i] = inStrides[i];
        params[4 + 2*ndim + i] = (uint32_t)startVals[i];
        params[4 + 3*ndim + i] = (uint32_t)(stepVals[i] < 0 ? (uint32_t)(int32_t)stepVals[i] : (uint32_t)stepVals[i]);
    }
    auto paramBuf = ex.getGpu()->createBuffer("slice_p", params.size() * 4);
    ex.getGpu()->writeBuffer(paramBuf, params.data(), params.size() * 4);

    auto& pl = ex.GetPipelineT("slice", 3, []() { return std::string(WGSL_SLICE); });
    auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (totalU32 + 255) / 256, 1, 1, "slice");
}

static void opConstantOfShape(OpContext& ex, const OnnxGraphNode& n,
                                const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (!in[0] || !in[0]->IsValid()) return;
    std::vector<int64_t> shape;
    if (!readTensorInt64Values(ex, in[0], n.inputs.empty() ? "" : n.inputs[0], shape))
        shape = {1};
    int64_t total = 1; for (auto d : shape) total *= d;
    *out[0] = ex.AllocTensor(shape, TensorDtype::Float32);
    std::vector<float> zeros(total, 0.0f);
    ex.getGpu()->writeBuffer(out[0]->buffer, zeros.data(), total*4);
}

static void opRange(OpContext& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in.size() < 3) return;

    // Detect input dtype to handle both int and float Range ops
    TensorDtype inDtype = in[0] ? in[0]->dtype : TensorDtype::Float32;
    bool isIntRange = (inDtype == TensorDtype::Int32 || inDtype == TensorDtype::Int64);

    auto rdInt = [&](GpuTensor* t, const std::string& nm) -> int64_t {
        if (t && t->isCpuOnly && !t->cpuData.empty()) {
            if (t->dtype == TensorDtype::Int32) { int32_t v; memcpy(&v, t->cpuData.data(), 4); return v; }
            if (t->dtype == TensorDtype::Int64) { int64_t v; memcpy(&v, t->cpuData.data(), 8); return v; }
            float v; memcpy(&v, t->cpuData.data(), 4); return (int64_t)v;
        }
        if (auto* i = ex.GetInitData(nm); i && i->data) {
            if (i->dtype == TensorDtype::Int32) { int32_t v; memcpy(&v, i->data, 4); return v; }
            if (i->dtype == TensorDtype::Int64) { int64_t v; memcpy(&v, i->data, 8); return v; }
            float v; memcpy(&v, i->data, 4); return (int64_t)v;
        }
        return 0;
    };
    auto rdFloat = [&](GpuTensor* t, const std::string& nm) -> float {
        if (t && t->isCpuOnly && !t->cpuData.empty()) { float v; memcpy(&v, t->cpuData.data(), 4); return v; }
        if (auto* i = ex.GetInitData(nm); i && i->data) { float v; memcpy(&v, i->data, 4); return v; }
        return 0;
    };

    if (isIntRange) {
        int64_t start = rdInt(in[0], n.inputs[0]);
        int64_t limit = rdInt(in[1], n.inputs[1]);
        int64_t delta = rdInt(in[2], n.inputs[2]);
        int64_t count = (delta != 0) ? std::max((int64_t)0, (limit - start + delta - (delta > 0 ? 1 : -1)) / delta) : 0;
        std::vector<int32_t> vals((size_t)count);
        for (int64_t i = 0; i < count; i++) vals[(size_t)i] = (int32_t)(start + i * delta);
        *out[0] = ex.AllocCpuTensor({count}, TensorDtype::Int32, vals.data(), count * 4);
        ex.EnsureGpu(*out[0]);
    } else {
        float start = rdFloat(in[0], n.inputs[0]);
        float limit = rdFloat(in[1], n.inputs[1]);
        float delta = rdFloat(in[2], n.inputs[2]);
        int64_t count = (delta != 0) ? (int64_t)std::ceil((limit - start) / delta) : 0;
        if (count <= 0) count = 0;
        std::vector<float> vals((size_t)count);
        for (int64_t i = 0; i < count; i++) vals[(size_t)i] = start + i * delta;
        *out[0] = ex.AllocTensor({count}, TensorDtype::Float32);
        if (count > 0) ex.getGpu()->writeBuffer(out[0]->buffer, vals.data(), count * 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expand — uses embedded expand kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opExpand(OpContext& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* shape = in.size() > 1 ? in[1] : nullptr;
    if (!data || !data->IsValid()) return;

    // Read target shape
    std::vector<int64_t> targetShape;
    if (shape)
        readTensorInt64Values(ex, shape, n.inputs.size() > 1 ? n.inputs[1] : "", targetShape);

    if (targetShape.empty()) {
        ex.EnsureGpu(*data);
        *out[0] = *data;
        return;
    }

    // Broadcast rules: align from right, take max
    int ndim = std::max((int)data->shape.size(), (int)targetShape.size());
    std::vector<int64_t> inPadded(ndim, 1), outPadded(ndim, 1);
    for (int i = 0; i < (int)data->shape.size(); i++)
        inPadded[ndim - (int)data->shape.size() + i] = data->shape[i];
    for (int i = 0; i < (int)targetShape.size(); i++)
        outPadded[ndim - (int)targetShape.size() + i] = targetShape[i];
    for (int i = 0; i < ndim; i++) {
        if (outPadded[i] == -1 || outPadded[i] == 0) outPadded[i] = inPadded[i];
        else if (inPadded[i] == 1) {} // broadcast
        else outPadded[i] = std::max(inPadded[i], outPadded[i]);
    }

    // Check if no actual expansion needed
    bool same = (inPadded == outPadded);
    if (same) {
        ex.EnsureGpu(*data);
        *out[0] = *data;
        out[0]->shape = outPadded;
        return;
    }

    // For int32/int64 tensors, the GPU kernel reads array<f32> which would
    // misinterpret the bit patterns. Use u32 kernel (Transpose) for 4-byte types
    // or do CPU expand for small tensors.
    bool isIntType = (data->dtype == TensorDtype::Int32 || data->dtype == TensorDtype::Int64);
    int64_t totalIn = 1;
    for (auto d : inPadded) totalIn *= d;

    if (isIntType && totalIn <= 65536) {
        // CPU expand for int tensors
        size_t elemSize = data->DtypeSize();
        size_t inBytes = (size_t)totalIn * elemSize;
        std::vector<uint8_t> inputBytes;
        if (!data->cpuData.empty() && data->cpuData.size() >= inBytes) {
            inputBytes.assign(data->cpuData.begin(), data->cpuData.begin() + inBytes);
        } else {
            ex.EnsureGpu(*data);
            ex.FlushPendingWork();
            auto rb = ex.getGpu()->readBuffer(data->buffer, inBytes);
            inputBytes.assign(rb.begin(), rb.begin() + std::min(rb.size(), inBytes));
        }
        int64_t totalOut = 1;
        for (auto d : outPadded) totalOut *= d;

        std::vector<uint8_t> outputBytes((size_t)totalOut * elemSize, 0);

        // Compute strides
        std::vector<int64_t> outStrides(ndim), inStrides(ndim);
        int64_t s = 1;
        for (int i = ndim-1; i >= 0; i--) { outStrides[i] = s; s *= outPadded[i]; }
        s = 1;
        for (int i = ndim-1; i >= 0; i--) { inStrides[i] = s; s *= inPadded[i]; }

        for (int64_t outIdx = 0; outIdx < totalOut; outIdx++) {
            int64_t rem = outIdx;
            int64_t inIdx = 0;
            for (int d = 0; d < ndim; d++) {
                int64_t coord = rem / outStrides[d];
                rem %= outStrides[d];
                inIdx += (coord % inPadded[d]) * inStrides[d];
            }
            memcpy(outputBytes.data() + (size_t)outIdx * elemSize,
                   inputBytes.data() + (size_t)inIdx * elemSize, elemSize);
        }

        *out[0] = ex.AllocCpuTensor(outPadded, data->dtype, outputBytes.data(), outputBytes.size());
        ex.EnsureGpu(*out[0]);
        return;
    }

    ex.EnsureGpu(*data);
    int64_t totalOut = 1;
    for (auto d : outPadded) totalOut *= d;
    *out[0] = ex.AllocTensor(outPadded, data->dtype);

    // Build strides
    std::vector<uint32_t> outStrides(ndim), inDims(ndim), inStrides(ndim);
    uint32_t s = 1;
    for (int i = ndim-1; i >= 0; i--) { outStrides[i] = s; s *= (uint32_t)outPadded[i]; }
    s = 1;
    for (int i = ndim-1; i >= 0; i--) { inStrides[i] = s; s *= (uint32_t)inPadded[i]; inDims[i] = (uint32_t)inPadded[i]; }

    std::vector<uint32_t> params(4 + 3*ndim, 0);
    params[0] = (uint32_t)totalOut;
    params[1] = (uint32_t)ndim;
    for (int i = 0; i < ndim; i++) {
        params[4+i] = outStrides[i];
        params[4+ndim+i] = inDims[i];
        params[4+2*ndim+i] = inStrides[i];
    }
    auto paramBuf = ex.getGpu()->createBuffer("expand_p", params.size()*4);
    ex.getGpu()->writeBuffer(paramBuf, params.data(), params.size()*4);

    // Use templated kernel for dtype-transparent expand
    TensorDtype dtype = data->dtype;
    if (dtype != TensorDtype::Float16) dtype = TensorDtype::Float32;
    std::string pname = "expand_t" + std::string(dtypeSuffix(dtype));
    auto& pl = ex.GetPipelineT(pname, 3, [dtype]() {
        return instantiateTemplate(WGSL_EXPAND_T, dtype);
    });
    auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    // Each thread handles 2 elements
    uint32_t numWorkgroups = (uint32_t)(((totalOut + 1) / 2 + 255) / 256);
    ex.QueueDispatch(pl.pipeline, bg,
        numWorkgroups, 1, 1, "expand");
}

// ═══════════════════════════════════════════════════════════════════════════
// Pad: zero-padding using GPU kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opPad(OpContext& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    // Read pads from input[1]
    std::vector<int64_t> pads;
    if (in.size() > 1 && in[1]) {
        readTensorInt64Values(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", pads);
    }

    // Check if all pads are zero
    bool allZero = true;
    for (auto p : pads) if (p != 0) { allZero = false; break; }
    if (allZero || pads.empty()) {
        ex.EnsureGpu(*data);
        *out[0] = *data;
        return;
    }

    // For now, simple zero-padded allocation with GPU copy
    int ndim = (int)data->shape.size();
    auto outShape = data->shape;
    for (int i = 0; i < ndim && i < (int)pads.size()/2; i++) {
        outShape[i] += pads[i] + pads[i + ndim];
    }

    ex.EnsureGpu(*data);
    *out[0] = ex.AllocTensor(outShape, data->dtype);

    // Zero-fill output via GPU write, then copy inner data via GPU buffer copy
    int64_t totalOut = 1;
    for (auto d : outShape) totalOut *= d;
    size_t outBytes = (size_t)totalOut * data->DtypeSize();
    {
        std::vector<uint8_t> zeros(outBytes, 0);
        ex.getGpu()->writeBuffer(out[0]->buffer, zeros.data(), outBytes);
    }

    // For simple cases (only leading/trailing pads, contiguous inner region),
    // use a single GPU buffer copy
    size_t elemSize = data->DtypeSize();
    int64_t totalIn = 1;
    for (auto d : data->shape) totalIn *= d;

    if (ndim <= 4 && totalIn > 0) {
        // Compute the flat byte offset where the inner data starts in the output
        std::vector<int64_t> outStrides(ndim);
        outStrides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) outStrides[d] = outStrides[d+1] * outShape[d+1];

        // For axis-0-only padding, the inner block is contiguous
        bool innerContiguous = true;
        for (int d = 1; d < ndim; d++) {
            if (pads[d] != 0 || pads[d + ndim] != 0) {
                innerContiguous = false; break;
            }
        }
        if (innerContiguous) {
            int64_t dstOffset = pads[0] * outStrides[0] * (int64_t)elemSize;
            int64_t copySize = totalIn * (int64_t)elemSize;
            if (dstOffset >= 0 && (size_t)(dstOffset + copySize) <= outBytes) {
                ex.QueueCopy(data->buffer, 0, out[0]->buffer, (uint64_t)dstOffset, (uint64_t)copySize);
                return;
            }
        }

        // General case: copy row-by-row along the last axis (no CPU readback)
        std::vector<int64_t> inStrides(ndim);
        inStrides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) inStrides[d] = inStrides[d+1] * data->shape[d+1];

        // Iterate over all "rows" (all dims except the innermost)
        int64_t nRows = totalIn / data->shape[ndim-1];
        int64_t rowLen = data->shape[ndim-1] * (int64_t)elemSize;
        for (int64_t r = 0; r < nRows; r++) {
            // Convert row index to per-dim coordinates
            int64_t rem = r;
            int64_t srcOff = 0, dstOff = 0;
            for (int d = 0; d < ndim - 1; d++) {
                int64_t coord = rem / (inStrides[d] / data->shape[ndim-1]);
                rem %= (inStrides[d] / data->shape[ndim-1]);
                srcOff += coord * inStrides[d];
                dstOff += (coord + pads[d]) * outStrides[d];
            }
            dstOff += pads[ndim-1];  // pad on innermost dim
            srcOff *= (int64_t)elemSize;
            dstOff *= (int64_t)elemSize;
            if (rowLen > 0)
                ex.QueueCopy(data->buffer, (uint64_t)srcOff, out[0]->buffer, (uint64_t)dstOff, (uint64_t)rowLen);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Split: split tensor along axis into multiple outputs
// ═══════════════════════════════════════════════════════════════════════════

static void opSplit(OpContext& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid() || data->shape.empty()) return;
    ex.EnsureGpu(*data);

    int64_t axis = n.GetInt("axis", 0);
    int ndim = (int)data->shape.size();
    if (axis < 0) axis += ndim;

    // Read split sizes from input[1] or attribute
    std::vector<int64_t> splits;
    if (in.size() > 1 && in[1]) {
        readTensorInt64Values(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", splits);
    }

    if (splits.empty()) {
        // Equal split
        int64_t nOut = (int64_t)out.size();
        if (nOut == 0) nOut = 1;
        int64_t dimSize = data->shape[axis];
        int64_t chunkSize = dimSize / nOut;
        for (int64_t i = 0; i < nOut; i++) splits.push_back(chunkSize);
    }

    if (out.size() == 1 && !splits.empty() && splits[0] == data->shape[axis]) {
        *out[0] = *data;
        return;
    }

    // Compute inner and outer sizes for buffer offset calculation
    int64_t innerSize = 1;
    for (int i = (int)axis + 1; i < ndim; i++) innerSize *= data->shape[i];
    int64_t outerSize = 1;
    for (int i = 0; i < (int)axis; i++) outerSize *= data->shape[i];

    size_t elemSize = data->DtypeSize();
    int64_t offset = 0;
    bool usedAlias = false;
    for (size_t i = 0; i < out.size() && i < splits.size(); i++) {
        auto outShape = data->shape;
        outShape[axis] = splits[i];
        int64_t chunkElements = splits[i] * innerSize;
        int64_t chunkBytes = chunkElements * elemSize;

        // For contiguous splits along axis 0, use buffer alias (zero-copy)
        if (axis == 0 || outerSize == 1) {
            size_t srcOffset = (size_t)(offset * innerSize * elemSize);
            size_t viewSize = (size_t)(chunkBytes * outerSize);

            // WebGPU requires 256-byte offset alignment for storage buffers
            if ((data->buffer.offset + srcOffset) % 256 == 0) {
                // Zero-copy alias: reference same buffer at offset
                out[i]->shape = outShape;
                out[i]->dtype = data->dtype;
                out[i]->buffer.handle = data->buffer.handle;
                out[i]->buffer.offset = data->buffer.offset + srcOffset;
                out[i]->buffer.size = viewSize;
                out[i]->isCpuOnly = false;
                usedAlias = true;
            } else {
                // Offset not aligned — fall back to copy
                *out[i] = ex.AllocTensor(outShape, data->dtype);
                ex.QueueCopy(data->buffer, srcOffset, out[i]->buffer, 0, viewSize);
            }
        } else {
            // General case: copy slabs
            *out[i] = ex.AllocTensor(outShape, data->dtype);
            for (int64_t o = 0; o < outerSize; o++) {
                size_t srcOff = (size_t)((o * data->shape[axis] + offset) * innerSize * elemSize);
                size_t dstOff = (size_t)(o * chunkElements * elemSize);
                ex.QueueCopy(data->buffer, srcOff, out[i]->buffer, dstOff, (size_t)chunkBytes);
            }
        }
        offset += splits[i];
    }
    // Flush pending dispatches+copies so downstream ops see the copy results.
    // Even when all splits used aliases (no copies), flush pending dispatches
    // for CPU-GPU pipelining — submitting work early gives the GPU a head start.
    if (!ex.exec.pendingDispatches_.empty() || !ex.exec.pendingCopies_.empty()) {
        ex.exec.flushToEncoder();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ScatterND: write updates into data at given indices
// ═══════════════════════════════════════════════════════════════════════════

static void opScatterND(OpContext& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;
    ex.EnsureGpu(*data);

    // For now, copy data and apply updates if available
    *out[0] = ex.AllocTensor(data->shape, data->dtype);
    size_t bytes = data->ByteSize();
    if (bytes > 0)
        ex.QueueCopy(data->buffer, 0, out[0]->buffer, 0, bytes);

    // TODO: apply updates from in[2] at indices in[1]
}

// ═══════════════════════════════════════════════════════════════════════════
// Mod: modulo operation (integer)
// ═══════════════════════════════════════════════════════════════════════════

static void opMod(OpContext& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in.size() > 1 ? in[1] : nullptr;
    if (kDebugModOp) {
        fprintf(stderr, "    [mod] enter A=%p B=%p\n", (void*)A, (void*)B);
        fflush(stderr);
    }
    if (!A || !B || !A->IsValid() || !B->IsValid()) {
        if (A && A->IsValid()) { ex.EnsureGpu(*A); *out[0] = *A; }
        if (kDebugModOp) {
            fprintf(stderr, "    [mod] early passthrough\n");
            fflush(stderr);
        }
        return;
    }

    // CPU path for small int64 tensors
    if ((A->isCpuOnly || A->ElementCount() <= 64) &&
        (A->dtype == TensorDtype::Int64 || A->dtype == TensorDtype::Int32) &&
        (B->dtype == TensorDtype::Int64 || B->dtype == TensorDtype::Int32)) {
        int64_t N_A = A->ElementCount();
        int64_t N_B = B->ElementCount();
        if (kDebugModOp) {
            fprintf(stderr, "    [mod] cpu path N_A=%lld N_B=%lld dtypeA=%d dtypeB=%d\n",
                (long long)N_A, (long long)N_B, (int)A->dtype, (int)B->dtype);
            fflush(stderr);
        }

        auto readI64 = [&](GpuTensor* t, const std::string& nm) -> std::vector<int64_t> {
            std::vector<int64_t> v;
            readTensorInt64Values(ex, t, nm, v);
            return v;
        };

        auto a = readI64(A, n.inputs[0]);
        auto b = readI64(B, n.inputs[1]);
        if (kDebugModOp) {
            fprintf(stderr, "    [mod] read a=%zu b=%zu\n", a.size(), b.size());
            fflush(stderr);
        }
        if (a.empty() || b.empty() || N_A <= 0 || N_B <= 0) {
            int64_t outCount = std::max<int64_t>(1, std::max(N_A, N_B));
            std::vector<int64_t> zeros((size_t)outCount, 0);
            std::vector<int64_t> outShape = preferredIntOutputShape(A, B, N_A, N_B);
            *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64,
                                        zeros.data(), zeros.size() * sizeof(int64_t));
            if (kDebugModOp) {
                fprintf(stderr, "    [mod] fallback zero tensor\n");
                fflush(stderr);
            }
            return;
        }

        int64_t N = std::max(N_A, N_B);
        std::vector<int64_t> outShape = preferredIntOutputShape(A, B, N_A, N_B);
        std::vector<int64_t> c(N);
        for (int64_t i = 0; i < N; i++) {
            int64_t bv = b[i % N_B];
            c[i] = (bv != 0) ? (a[i % N_A] % bv) : 0;
        }
        *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, c.data(), N * 8);
        if (kDebugModOp) {
            fprintf(stderr, "    [mod] cpu path done N=%lld\n", (long long)N);
            fflush(stderr);
        }
        return;
    }

    ex.EnsureGpu(*A);
    *out[0] = *A;
    if (kDebugModOp) {
        fprintf(stderr, "    [mod] gpu passthrough\n");
        fflush(stderr);
    }
}

REGISTER_OP(Reshape, opReshape)
REGISTER_OP(Squeeze, opSqueeze)
REGISTER_OP(Unsqueeze, opUnsqueeze)
REGISTER_OP(Flatten, opFlatten)
REGISTER_OP(Constant, opConstant)
REGISTER_OP(Shape, opShape)
REGISTER_OP(Gather, opGather)
REGISTER_OP(Concat, opConcat)
REGISTER_OP(Transpose, opTranspose)
REGISTER_OP(Slice, opSlice)
REGISTER_OP(ConstantOfShape, opConstantOfShape)
REGISTER_OP(Range, opRange)
REGISTER_OP(Expand, opExpand)
REGISTER_OP(Pad, opPad)
REGISTER_OP(Split, opSplit)
REGISTER_OP(ScatterND, opScatterND)
REGISTER_OP(Mod, opMod)

// ─── Concat2D: Flat GPU concatenation of two buffers ─────────────────────────
// Out = [A; B]. Simple 1D concat via compute shader.

static void opConcat2D(OpContext& ex, const OnnxGraphNode& n,
                        const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0];
    auto* B = in[1];
    if (!A || !A->IsValid() || !B || !B->IsValid()) return;

    TensorDtype dtype = A->dtype;
    if (dtype != TensorDtype::Float16) dtype = TensorDtype::Float32;
    ex.EnsureGpu(*A);
    ex.EnsureGpu(*B);

    int64_t N_a = tensorNel(A);
    int64_t N_b = tensorNel(B);
    int64_t N_total = N_a + N_b;

    *out[0] = ex.AllocTensor({N_total}, dtype);

    uint32_t params[4] = {(uint32_t)N_a, (uint32_t)N_total, 0, 0};
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    std::string pname = "concat_2d" + std::string(dtypeSuffix(dtype));
    auto& pl = ex.GetPipelineT(pname, 4, [dtype]() {
        return instantiateTemplate(WGSL_CONCAT_2D_T, dtype);
    });
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, paramBuf}});
    ex.SubmitAsync({{pl.pipeline, bg,
        (uint32_t)((N_total + 255) / 256), 1, 1, "concat_2d"}});
}

REGISTER_OP(Concat2D, opConcat2D)

// ─── SplitCopy: GPU slice-copy without host readback ─────────────────────────
// Dst = Src[offset:offset+N].

static void opSplitCopy(OpContext& ex, const OnnxGraphNode& n,
                         const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Src = in[0];
    if (!Src || !Src->IsValid()) return;

    TensorDtype dtype = Src->dtype;
    if (dtype != TensorDtype::Float16) dtype = TensorDtype::Float32;
    ex.EnsureGpu(*Src);

    int64_t offset = n.GetInt("offset", 0);
    int64_t count = n.GetInt("count", 0);
    if (count <= 0) {
        // Infer from output shape if available
        count = tensorNel(out[0]);
        if (count <= 0) count = tensorNel(Src) - offset;
    }

    *out[0] = ex.AllocTensor({count}, dtype);

    uint32_t params[4] = {(uint32_t)offset, (uint32_t)count, 0, 0};
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    std::string pname = "split_copy" + std::string(dtypeSuffix(dtype));
    auto& pl = ex.GetPipelineT(pname, 3, [dtype]() {
        return instantiateTemplate(WGSL_SPLIT_COPY_T, dtype);
    });
    auto bg = ex.MakeBindGroup(pl, {
        {0, Src->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    ex.SubmitAsync({{pl.pipeline, bg,
        (uint32_t)((count + 255) / 256), 1, 1, "split_copy"}});
}

REGISTER_OP(SplitCopy, opSplitCopy)
