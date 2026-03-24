/**
 * ops/shape.cpp — Shape manipulation ONNX ops.
 * Uses embedded WGSL kernels from compiler/kernels/shared/.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero-copy ops (no GPU work, just change shape metadata)
// ═══════════════════════════════════════════════════════════════════════════

static void opReshape(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* shape = in.size() > 1 ? in[1] : nullptr;
    if (!data || !data->IsValid()) return;

    std::vector<int64_t> newShape;
    // Read shape from CPU data (initializer or prior CPU-computed tensor)
    if (shape && n.inputs.size() > 1) {
        auto* init = ex.GetInitData(n.inputs[1]);
        if (init && init->data) {
            int64_t nel = 1; for (auto d : init->shape) nel *= d;
            newShape.resize(nel);
            memcpy(newShape.data(), init->data, nel * 8);
        } else if (shape->isCpuOnly && !shape->cpuData.empty()) {
            int64_t nel = tensorNel(shape);
            newShape.resize(nel);
            memcpy(newShape.data(), shape->cpuData.data(), nel * 8);
        } else if (!shape->cpuData.empty()) {
            int64_t nel = tensorNel(shape);
            newShape.resize(nel);
            memcpy(newShape.data(), shape->cpuData.data(), nel * 8);
        }
    }
    if (newShape.empty() && shape) {
        fprintf(stderr, "    [reshape] EMPTY: shape->isCpu=%d cpuData=%zu shape=[",
                shape->isCpuOnly, shape->cpuData.size());
        for (size_t i = 0; i < shape->shape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)shape->shape[i]);
        fprintf(stderr, "] buf=%p\n", (void*)shape->buffer.handle);
        fflush(stderr);
    }
    if (newShape.empty()) {
        fprintf(stderr, "    [reshape] empty shape! data=[");
        for (size_t i = 0; i < data->shape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)data->shape[i]);
        fprintf(stderr, "] shape_input=%s isCpu=%d cpuData=%zu\n",
                (shape ? "valid" : "null"), (shape ? shape->isCpuOnly : 0),
                (shape ? shape->cpuData.size() : 0));
        if (n.inputs.size() > 1) {
            auto* init = ex.GetInitData(n.inputs[1]);
            fprintf(stderr, "    [reshape] init=%p name='%s'\n",
                    (void*)init, n.inputs[1].c_str());
        }
        fflush(stderr);
        *out[0] = *data; return;
    }

    int64_t totalIn = tensorNel(data);
    int64_t known = 1; int inferIdx = -1;
    for (int i = 0; i < (int)newShape.size(); i++) {
        if (newShape[i] == 0 && i < (int)data->shape.size()) newShape[i] = data->shape[i];
        if (newShape[i] == -1) inferIdx = i; else known *= newShape[i];
    }
    if (inferIdx >= 0 && known > 0) newShape[inferIdx] = totalIn / known;
    fprintf(stderr, "    [reshape] in=[");
    for (size_t i = 0; i < data->shape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)data->shape[i]);
    fprintf(stderr, "] -> out=[");
    for (size_t i = 0; i < newShape.size(); i++) fprintf(stderr, "%s%lld", i?",":"", (long long)newShape[i]);
    fprintf(stderr, "] total=%lld\n", (long long)totalIn);
    fflush(stderr);
    *out[0] = *data;
    out[0]->shape = newShape;
}

static void opSqueeze(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1]) {
        int64_t nel = tensorNel(in[1]); axes.resize(nel);
        if (in[1]->isCpuOnly) memcpy(axes.data(), in[1]->cpuData.data(), nel*8);
        else if (auto* i = ex.GetInitData(n.inputs[1]); i && i->data) memcpy(axes.data(), i->data, nel*8);
    } else if (n.attrIntLists.count("axes")) axes = n.attrIntLists.at("axes");
    std::vector<int64_t> ns;
    for (int i = 0; i < (int)data->shape.size(); i++) {
        bool sq = axes.empty() ? (data->shape[i]==1) : false;
        for (auto a : axes) { if ((a<0?a+(int64_t)data->shape.size():a)==i) { sq=true; break; } }
        if (!sq) ns.push_back(data->shape[i]);
    }
    *out[0] = *data; out[0]->shape = ns;
}

static void opUnsqueeze(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1]) {
        int64_t nel = tensorNel(in[1]); axes.resize(nel);
        if (in[1]->isCpuOnly) memcpy(axes.data(), in[1]->cpuData.data(), nel*8);
        else if (auto* i = ex.GetInitData(n.inputs[1]); i && i->data) memcpy(axes.data(), i->data, nel*8);
    } else if (n.attrIntLists.count("axes")) axes = n.attrIntLists.at("axes");
    int ndim = (int)data->shape.size() + (int)axes.size();
    for (auto& a : axes) if (a < 0) a += ndim;
    std::sort(axes.begin(), axes.end());
    std::vector<int64_t> ns(ndim); int si = 0;
    for (int i = 0; i < ndim; i++)
        ns[i] = (std::find(axes.begin(), axes.end(), i) != axes.end()) ? 1 : data->shape[si++];
    *out[0] = *data; out[0]->shape = ns;
}

static void opFlatten(GraphExecutor& ex, const OnnxGraphNode& n,
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

static void opShape(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data) return;
    int ndim = (int)data->shape.size();
    *out[0] = ex.AllocCpuTensor({ndim}, TensorDtype::Int64, data->shape.data(), ndim*8);
}

static void opConstant(GraphExecutor& ex, const OnnxGraphNode& n,
                        const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Check if output was already pre-stored (tensor-valued 'value' attribute)
    if (out[0] && out[0]->IsValid()) return;

    if (n.attrIntLists.count("value_ints")) {
        auto& v = n.attrIntLists.at("value_ints");
        *out[0] = ex.AllocCpuTensor({(int64_t)v.size()}, TensorDtype::Int64, v.data(), v.size()*8);
    } else if (n.attrInts.count("value_int")) {
        int64_t v = n.GetInt("value_int");
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Int64, &v, 8);
    } else if (n.attrFloats.count("value_float")) {
        float v = n.GetFloat("value_float");
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &v, 4);
    } else {
        float z = 0; *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &z, 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Gather — uses embedded gather kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opGather(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; auto* indices = in[1];
    if (!data || !indices || !data->IsValid() || !indices->IsValid()) return;
    int64_t axis = n.GetInt("axis", 0);
    int64_t nIdx = tensorNel(indices);

    // CPU path: both inputs available on CPU
    const uint8_t* dataPtr = nullptr; size_t dataBytes = 0;
    const uint8_t* idxPtr = nullptr;
    if (data->isCpuOnly) { dataPtr = data->cpuData.data(); dataBytes = data->cpuData.size(); }
    else if (auto* i = ex.GetInitData(n.inputs[0]); i && i->data) { dataPtr = i->data; dataBytes = i->size; }
    if (indices->isCpuOnly) idxPtr = indices->cpuData.data();
    else if (n.inputs.size()>1) if (auto* i = ex.GetInitData(n.inputs[1]); i && i->data) idxPtr = i->data;

    if (dataPtr && idxPtr && axis == 0 && !data->shape.empty()) {
        int64_t inner = 1;
        for (size_t i=1; i<data->shape.size(); i++) inner *= data->shape[i];
        size_t sliceB = (size_t)(inner * data->DtypeSize());
        std::vector<int64_t> os;
        for (auto d : indices->shape) if (d > 0) os.push_back(d);
        for (size_t i=1; i<data->shape.size(); i++) os.push_back(data->shape[i]);
        if (os.empty()) for (size_t i=1; i<data->shape.size(); i++) os.push_back(data->shape[i]);
        std::vector<int64_t> iv(nIdx);
        if (indices->dtype == TensorDtype::Int64) memcpy(iv.data(), idxPtr, nIdx*8);
        else { auto* s = (const int32_t*)idxPtr; for (int64_t i=0; i<nIdx; i++) iv[i]=s[i]; }
        std::vector<uint8_t> od(nIdx * sliceB);
        for (int64_t i=0; i<nIdx; i++) {
            int64_t idx = iv[i]; if (idx<0) idx += data->shape[0];
            if (idx>=0 && (size_t)((idx+1)*sliceB) <= dataBytes)
                memcpy(od.data()+i*sliceB, dataPtr+idx*sliceB, sliceB);
        }
        *out[0] = ex.AllocCpuTensor(os, data->dtype, od.data(), od.size());
        return;
    }

    // GPU path: dispatch gather kernel (NO sync)
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
        if (nIdx <= 1024 && (indices->isCpuOnly || idxPtr)) {
            std::vector<int32_t> i32(nIdx);
            const uint8_t* src = indices->isCpuOnly ? indices->cpuData.data() : idxPtr;
            if (src) { for (int64_t i=0;i<nIdx;i++) { int64_t v; memcpy(&v, src+i*8, 8); i32[i]=(int32_t)v; } }
            idxBuf = ex.gpu->createBuffer("gather_idx32", nIdx * 4);
            ex.gpu->writeBuffer(idxBuf, i32.data(), nIdx * 4);
        }
    }

    uint32_t total = (uint32_t)(nIdx * sliceSizeU32);
    uint32_t params[4] = {(uint32_t)nIdx, sliceSizeU32, dataStrideU32, 0};
    auto paramBuf = ex.gpu->createBuffer("gather_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("gather", WGSL_GATHER, 4);
    auto bg = ex.MakeBindGroup(pl, {
        {0, data->buffer}, {1, idxBuf}, {2, out[0]->buffer}, {3, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (total + 255) / 256, 1, 1, "gather"});
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Concat (CopyBufferToBuffer, no sync)
// ═══════════════════════════════════════════════════════════════════════════

static void opConcat(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    fprintf(stderr, "    [concat-enter] in.size=%zu out.size=%zu\n", in.size(), out.size());
    fflush(stderr);

    int64_t axis = n.GetInt("axis", 0);
    std::vector<GpuTensor*> validIn;
    for (size_t i = 0; i < in.size(); i++) {
        auto* t = in[i];
        if (!t) continue;
        // Safety: check the pointer is dereferenceable
        fprintf(stderr, "      [concat] in[%zu] ptr=%p\n", i, (void*)t);
        fflush(stderr);
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

    for (auto* t : validIn) ex.EnsureGpu(*t);

    // Filter out tensors with no GPU buffer after EnsureGpu
    std::vector<GpuTensor*> gpuIn;
    for (auto* t : validIn)
        if (t->buffer.handle) gpuIn.push_back(t);

    fprintf(stderr, "    [concat] axis=%lld ndim=%d validIn=%zu gpuIn=%zu\n",
            (long long)axis, ndim, validIn.size(), gpuIn.size());
    for (size_t i = 0; i < validIn.size(); i++) {
        auto* t = validIn[i];
        fprintf(stderr, "      [%zu] shape=[", i);
        for (size_t j = 0; j < t->shape.size(); j++) fprintf(stderr, "%s%lld", j?",":"", (long long)t->shape[j]);
        fprintf(stderr, "] buf=%p size=%zu isCpu=%d cpuData=%zu\n",
                (void*)t->buffer.handle, t->buffer.size, t->isCpuOnly, t->cpuData.size());
    }
    fflush(stderr);

    if (gpuIn.empty()) return;

    *out[0] = ex.AllocTensor(outShape, gpuIn[0]->dtype);

    // For axis=0 or 1D tensors, simple byte concatenation
    if (axis == 0 || ndim <= 1) {
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

        int64_t dstAxisOffset = 0;
        for (auto* t : gpuIn) {
            int64_t srcAxisSize = t->shape[axis];
            for (int64_t o = 0; o < outerSize; o++) {
                size_t srcOff = (size_t)(o * srcAxisSize * innerSize * elemSize);
                size_t dstOff = (size_t)((o * totalOnAxis + dstAxisOffset) * innerSize * elemSize);
                size_t copySize = (size_t)(srcAxisSize * innerSize * elemSize);
                if (copySize > 0 && t->buffer.handle)
                    ex.QueueCopy(t->buffer, srcOff, out[0]->buffer, dstOff, copySize);
            }
            dstAxisOffset += srcAxisSize;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Transpose — uses embedded transpose kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opTranspose(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    if (data->isCpuOnly) { *out[0] = *data; return; }
    ex.EnsureGpu(*data);
    if (!data->buffer.handle) { *out[0] = *data; return; } // buffer was freed

    std::vector<int64_t> perm;
    if (n.attrIntLists.count("perm")) perm = n.attrIntLists.at("perm");
    else for (int i=(int)data->shape.size()-1; i>=0; i--) perm.push_back(i);

    // Ensure data shape matches perm dimensions
    // If perm has more dims than data shape, pad shape with 1s (local copy)
    auto dataShape = data->shape;
    while ((int)dataShape.size() < (int)perm.size()) {
        dataShape.insert(dataShape.begin(), 1);
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
    size_t elemSize = data->DtypeSize();
    int ndim = (int)dataShape.size();

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

    if (isNoop || elemSize != 4) {
        *out[0] = *data;
        out[0]->shape = outShape;
        return;
    }

    // Safety: verify buffer is large enough for the kernel
    if (!data->buffer.handle || data->buffer.size < (size_t)(nel * elemSize)) {
        *out[0] = *data;
        out[0]->shape = outShape;
        return;
    }

    *out[0] = ex.AllocTensor(outShape, data->dtype);

    std::vector<int64_t> inStrides(ndim);
    inStrides[ndim-1] = 1;
    for (int i=ndim-2; i>=0; i--) inStrides[i] = inStrides[i+1] * dataShape[i+1];

    uint32_t ostride = 1;
    std::vector<uint32_t> outStrides(ndim), permInStrides(ndim);
    for (int i=ndim-1; i>=0; i--) {
        outStrides[i] = ostride;
        ostride *= (uint32_t)outShape[i];
        permInStrides[i] = (uint32_t)inStrides[perm[i]];
    }

    uint32_t elemsU32 = (elemSize == 8) ? (uint32_t)(nel * 2) : (uint32_t)nel;
    std::vector<uint32_t> params(4 + 2*ndim, 0);
    params[0] = elemsU32; params[1] = (uint32_t)ndim;
    for (int i=0; i<ndim; i++) {
        params[4+i] = (elemSize==8) ? outStrides[i]*2 : outStrides[i];
        params[4+ndim+i] = (elemSize==8) ? permInStrides[i]*2 : permInStrides[i];
    }
    auto paramBuf = ex.gpu->createBuffer("tr_p", params.size()*4);
    ex.gpu->writeBuffer(paramBuf, params.data(), params.size()*4);

    auto& pl = ex.GetPipeline("transpose", WGSL_TRANSPOSE, 3);
    auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg, (elemsU32+255)/256, 1, 1, "transpose"});
}

// ═══════════════════════════════════════════════════════════════════════════
// Simple / pass-through ops
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// GPU Slice — uses embedded slice kernel
// ═══════════════════════════════════════════════════════════════════════════

static int64_t readInt64FromTensor(GraphExecutor& ex, const GpuTensor* t, const std::string& name) {
    const uint8_t* p = nullptr;
    if (t && t->isCpuOnly && !t->cpuData.empty()) p = t->cpuData.data();
    else if (t && !t->cpuData.empty()) p = t->cpuData.data();
    else if (auto* init = ex.GetInitData(name); init && init->data) p = init->data;
    if (p) { int64_t v; memcpy(&v, p, 8); return v; }
    return 0;
}

static void opSlice(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid() || data->shape.empty()) return;

    int ndim = (int)data->shape.size();

    // Read starts, ends, axes, steps from inputs (all int64)
    auto readI64Vec = [&](int inputIdx) -> std::vector<int64_t> {
        if (inputIdx >= (int)in.size() || !in[inputIdx]) return {};
        auto* t = in[inputIdx];
        const uint8_t* p = nullptr;
        if (t->isCpuOnly && !t->cpuData.empty()) p = t->cpuData.data();
        else if (!t->cpuData.empty()) p = t->cpuData.data();
        else if (inputIdx < (int)n.inputs.size()) {
            if (auto* init = ex.GetInitData(n.inputs[inputIdx]); init && init->data) p = init->data;
        }
        if (!p) return {};
        int64_t nel = t->ElementCount();
        std::vector<int64_t> v(nel);
        memcpy(v.data(), p, nel * 8);
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

    // Compute output shape
    auto outShape = data->shape;
    std::vector<int64_t> startVals(ndim, 0), stepVals(ndim, 1);
    for (int i = 0; i < (int)axes.size(); i++) {
        int64_t axis = axes[i];
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim) continue;

        int64_t dimSize = data->shape[axis];
        int64_t start = starts[i], end = ends[i], step = steps[i];

        // Clamp negative indices
        if (start < 0) start += dimSize;
        if (end < 0) end += dimSize;
        // Clamp to [0, dimSize]
        start = std::max((int64_t)0, std::min(start, dimSize));
        end = std::max((int64_t)0, std::min(end, dimSize));

        if (step < 0) {
            // Reverse slice
            if (start > dimSize - 1) start = dimSize - 1;
            if (end < -1) end = -1;
            outShape[axis] = (start - end + (-step - 1)) / (-step);
        } else {
            outShape[axis] = (end - start + step - 1) / step;
        }
        if (outShape[axis] < 0) outShape[axis] = 0;
        startVals[axis] = start;
        stepVals[axis] = step;
    }

    int64_t totalOut = 1;
    for (auto d : outShape) totalOut *= d;
    if (totalOut <= 0) {
        *out[0] = ex.AllocTensor({0}, data->dtype);
        return;
    }

    ex.EnsureGpu(*data);
    size_t elemSize = data->DtypeSize();

    // For simple cases (single contiguous slice), use buffer copy
    // For general case, use GPU kernel
    *out[0] = ex.AllocTensor(outShape, data->dtype);

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
    auto paramBuf = ex.gpu->createBuffer("slice_p", params.size() * 4);
    ex.gpu->writeBuffer(paramBuf, params.data(), params.size() * 4);

    auto& pl = ex.GetPipeline("slice", WGSL_SLICE, 3);
    auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (totalU32 + 255) / 256, 1, 1, "slice"});
}

static void opConstantOfShape(GraphExecutor& ex, const OnnxGraphNode& n,
                                const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (!in[0] || !in[0]->IsValid()) return;
    std::vector<int64_t> shape;
    if (in[0]->isCpuOnly) {
        int64_t nel = tensorNel(in[0]); shape.resize(nel);
        memcpy(shape.data(), in[0]->cpuData.data(), nel*8);
    } else if (auto* i = ex.GetInitData(n.inputs[0]); i && i->data) {
        int64_t nel = 1; for (auto d : i->shape) nel *= d;
        shape.resize(nel); memcpy(shape.data(), i->data, nel*8);
    } else shape = {1};
    int64_t total = 1; for (auto d : shape) total *= d;
    *out[0] = ex.AllocTensor(shape, TensorDtype::Float32);
    std::vector<float> zeros(total, 0.0f);
    ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), total*4);
}

static void opRange(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in.size() < 3) return;
    auto rd = [&](GpuTensor* t, const std::string& nm) -> float {
        if (t && t->isCpuOnly && !t->cpuData.empty()) { float v; memcpy(&v, t->cpuData.data(), 4); return v; }
        if (auto* i = ex.GetInitData(nm); i && i->data) { float v; memcpy(&v, i->data, 4); return v; }
        return 0;
    };
    float start = rd(in[0], n.inputs[0]), limit = rd(in[1], n.inputs[1]), delta = rd(in[2], n.inputs[2]);
    int64_t count = (delta != 0) ? (int64_t)std::ceil((limit-start)/delta) : 0;
    if (count <= 0) count = 0;
    std::vector<float> vals(count);
    for (int64_t i=0; i<count; i++) vals[i] = start + i*delta;
    *out[0] = ex.AllocTensor({count}, TensorDtype::Float32);
    if (count > 0) ex.gpu->writeBuffer(out[0]->buffer, vals.data(), count*4);
}

// ═══════════════════════════════════════════════════════════════════════════
// Expand — uses embedded expand kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opExpand(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* shape = in.size() > 1 ? in[1] : nullptr;
    if (!data || !data->IsValid()) return;

    // Read target shape
    std::vector<int64_t> targetShape;
    if (shape) {
        const uint8_t* p = nullptr;
        if (shape->isCpuOnly && !shape->cpuData.empty()) p = shape->cpuData.data();
        else if (!shape->cpuData.empty()) p = shape->cpuData.data();
        else if (n.inputs.size() > 1) {
            if (auto* init = ex.GetInitData(n.inputs[1]); init && init->data) p = init->data;
        }
        if (p) {
            int64_t nel = shape->ElementCount();
            targetShape.resize(nel);
            memcpy(targetShape.data(), p, nel * 8);
        }
    }

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
    auto paramBuf = ex.gpu->createBuffer("expand_p", params.size()*4);
    ex.gpu->writeBuffer(paramBuf, params.data(), params.size()*4);

    auto& pl = ex.GetPipeline("expand", WGSL_EXPAND, 3);
    auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((totalOut+255)/256), 1, 1, "expand"});
}

// ═══════════════════════════════════════════════════════════════════════════
// Pad: zero-padding using GPU kernel
// ═══════════════════════════════════════════════════════════════════════════

static void opPad(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    // Read pads from input[1]
    std::vector<int64_t> pads;
    if (in.size() > 1 && in[1]) {
        auto* t = in[1];
        const uint8_t* p = nullptr;
        if (t->isCpuOnly && !t->cpuData.empty()) p = t->cpuData.data();
        else if (!t->cpuData.empty()) p = t->cpuData.data();
        else if (auto* init = ex.GetInitData(n.inputs[1]); init && init->data) p = init->data;
        if (p) {
            int64_t nel = t->ElementCount();
            pads.resize(nel);
            memcpy(pads.data(), p, nel * 8);
        }
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
    // Zero-fill output, then copy data into padded position
    int64_t totalOut = 1;
    for (auto d : outShape) totalOut *= d;
    std::vector<float> zeros((size_t)totalOut, 0.0f);
    ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), totalOut * 4);
    // TODO: GPU copy of inner region for non-trivial padding
}

// ═══════════════════════════════════════════════════════════════════════════
// Split: split tensor along axis into multiple outputs
// ═══════════════════════════════════════════════════════════════════════════

static void opSplit(GraphExecutor& ex, const OnnxGraphNode& n,
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
        const uint8_t* p = nullptr;
        if (in[1]->isCpuOnly && !in[1]->cpuData.empty()) p = in[1]->cpuData.data();
        else if (!in[1]->cpuData.empty()) p = in[1]->cpuData.data();
        else if (auto* init = ex.GetInitData(n.inputs[1]); init && init->data) p = init->data;
        if (p) {
            int64_t nel = in[1]->ElementCount();
            splits.resize(nel);
            memcpy(splits.data(), p, nel * 8);
        }
    }

    if (splits.empty()) {
        // Equal split
        int64_t nOut = (int64_t)out.size();
        if (nOut == 0) nOut = 1;
        int64_t dimSize = data->shape[axis];
        int64_t chunkSize = dimSize / nOut;
        for (int64_t i = 0; i < nOut; i++) splits.push_back(chunkSize);
    }

    // Compute inner and outer sizes for buffer offset calculation
    int64_t innerSize = 1;
    for (int i = (int)axis + 1; i < ndim; i++) innerSize *= data->shape[i];
    int64_t outerSize = 1;
    for (int i = 0; i < (int)axis; i++) outerSize *= data->shape[i];

    size_t elemSize = data->DtypeSize();
    int64_t offset = 0;
    for (size_t i = 0; i < out.size() && i < splits.size(); i++) {
        auto outShape = data->shape;
        outShape[axis] = splits[i];
        int64_t chunkElements = splits[i] * innerSize;
        int64_t chunkBytes = chunkElements * elemSize;

        *out[i] = ex.AllocTensor(outShape, data->dtype);

        // For contiguous splits along axis 0 or last axis, use buffer copy
        if (axis == 0 || outerSize == 1) {
            size_t srcOffset = (size_t)(offset * innerSize * elemSize);
            ex.QueueCopy(data->buffer, srcOffset, out[i]->buffer, 0, (size_t)(chunkBytes * outerSize));
        } else {
            // General case: copy slabs
            for (int64_t o = 0; o < outerSize; o++) {
                size_t srcOff = (size_t)((o * data->shape[axis] + offset) * innerSize * elemSize);
                size_t dstOff = (size_t)(o * chunkElements * elemSize);
                ex.QueueCopy(data->buffer, srcOff, out[i]->buffer, dstOff, (size_t)chunkBytes);
            }
        }
        offset += splits[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ScatterND: write updates into data at given indices
// ═══════════════════════════════════════════════════════════════════════════

static void opScatterND(GraphExecutor& ex, const OnnxGraphNode& n,
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

static void opMod(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in.size() > 1 ? in[1] : nullptr;
    if (!A || !B || !A->IsValid() || !B->IsValid()) {
        if (A && A->IsValid()) { ex.EnsureGpu(*A); *out[0] = *A; }
        return;
    }

    // CPU path for small int64 tensors
    if ((A->isCpuOnly || A->ElementCount() <= 64) &&
        (A->dtype == TensorDtype::Int64 || A->dtype == TensorDtype::Int32)) {
        int64_t N_A = A->ElementCount();
        int64_t N_B = B->ElementCount();
        int64_t N = std::max(N_A, N_B);
        auto& outShape = (N_A >= N_B) ? A->shape : B->shape;

        auto readI64 = [&](GpuTensor* t, const std::string& nm) -> std::vector<int64_t> {
            std::vector<int64_t> v(t->ElementCount());
            const uint8_t* p = nullptr;
            if (t->isCpuOnly && !t->cpuData.empty()) p = t->cpuData.data();
            else if (!t->cpuData.empty()) p = t->cpuData.data();
            else if (auto* init = ex.GetInitData(nm); init && init->data) p = init->data;
            if (p) memcpy(v.data(), p, v.size() * 8);
            return v;
        };

        auto a = readI64(A, n.inputs[0]);
        auto b = readI64(B, n.inputs[1]);
        std::vector<int64_t> c(N);
        for (int64_t i = 0; i < N; i++) {
            int64_t bv = b[i % N_B];
            c[i] = (bv != 0) ? (a[i % N_A] % bv) : 0;
        }
        *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, c.data(), N * 8);
        return;
    }

    ex.EnsureGpu(*A);
    *out[0] = *A;
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
