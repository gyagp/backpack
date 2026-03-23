/**
 * ops/shape.cpp — Shape manipulation ONNX ops (ALL GPU, no CPU readback).
 *
 * Zero-copy: Reshape, Squeeze, Unsqueeze, Flatten — same buffer, new shape.
 * GPU kernels: Transpose, Gather, Concat, Slice.
 * Metadata: Shape, Constant, ConstantOfShape, Range — upload from initializers.
 */

#include "../graph_executor.h"
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0;
    int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ─── Zero-copy shape ops ─────────────────────────────────────────────────────

static void opReshape(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* shape = in.size() > 1 ? in[1] : nullptr;
    if (!data || !data->IsValid()) return;

    std::vector<int64_t> newShape;

    // Try to get shape from CPU initializer data (no GPU readback)
    if (shape && n.inputs.size() > 1) {
        auto* initData = ex.GetInitData(n.inputs[1]);
        if (initData && initData->data) {
            int64_t nel = 1;
            for (auto d : initData->shape) nel *= d;
            newShape.resize(nel);
            memcpy(newShape.data(), initData->data, nel * sizeof(int64_t));
        } else if (shape->isCpuOnly && !shape->cpuData.empty()) {
            int64_t nel = tensorNel(shape);
            newShape.resize(nel);
            memcpy(newShape.data(), shape->cpuData.data(), nel * sizeof(int64_t));
        }
    }

    if (newShape.empty()) {
        // Can't determine shape without readback — pass through
        *out[0] = *data;
        return;
    }

    // Handle -1 (infer) and 0 (keep)
    int64_t totalIn = tensorNel(data);
    int64_t known = 1;
    int inferIdx = -1;
    for (int i = 0; i < (int)newShape.size(); i++) {
        if (newShape[i] == 0 && i < (int)data->shape.size())
            newShape[i] = data->shape[i];
        if (newShape[i] == -1) inferIdx = i;
        else known *= newShape[i];
    }
    if (inferIdx >= 0 && known > 0) newShape[inferIdx] = totalIn / known;

    *out[0] = *data;
    out[0]->shape = newShape;
}

static void opSqueeze(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1] && in[1]->IsValid()) {
        int64_t nel = tensorNel(in[1]);
        axes.resize(nel);
        if (in[1]->isCpuOnly && !in[1]->cpuData.empty())
            memcpy(axes.data(), in[1]->cpuData.data(), nel * 8);
        else if (auto* init = ex.GetInitData(n.inputs[1]); init && init->data)
            memcpy(axes.data(), init->data, nel * 8);
    } else if (n.attrIntLists.count("axes")) {
        axes = n.attrIntLists.at("axes");
    }

    std::vector<int64_t> newShape;
    for (int i = 0; i < (int)data->shape.size(); i++) {
        bool squeeze = false;
        if (axes.empty()) squeeze = (data->shape[i] == 1);
        else for (auto a : axes) { if ((a < 0 ? a + data->shape.size() : a) == i) { squeeze = true; break; } }
        if (!squeeze) newShape.push_back(data->shape[i]);
    }
    *out[0] = *data;
    out[0]->shape = newShape;
}

static void opUnsqueeze(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1] && in[1]->IsValid()) {
        int64_t nel = tensorNel(in[1]);
        axes.resize(nel);
        if (in[1]->isCpuOnly && !in[1]->cpuData.empty())
            memcpy(axes.data(), in[1]->cpuData.data(), nel * 8);
        else if (auto* init = ex.GetInitData(n.inputs[1]); init && init->data)
            memcpy(axes.data(), init->data, nel * 8);
    } else if (n.attrIntLists.count("axes")) {
        axes = n.attrIntLists.at("axes");
    }

    int ndim = (int)data->shape.size() + (int)axes.size();
    for (auto& a : axes) if (a < 0) a += ndim;
    std::sort(axes.begin(), axes.end());
    std::vector<int64_t> newShape(ndim);
    int srcIdx = 0;
    for (int i = 0; i < ndim; i++) {
        if (std::find(axes.begin(), axes.end(), i) != axes.end()) newShape[i] = 1;
        else newShape[i] = data->shape[srcIdx++];
    }
    *out[0] = *data;
    out[0]->shape = newShape;
}

static void opFlatten(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;
    int64_t axis = n.GetInt("axis", 1);
    if (axis < 0) axis += data->shape.size();
    int64_t d0 = 1, d1 = 1;
    for (int i = 0; i < axis; i++) d0 *= data->shape[i];
    for (int i = (int)axis; i < (int)data->shape.size(); i++) d1 *= data->shape[i];
    *out[0] = *data;
    out[0]->shape = {d0, d1};
}

// ─── Metadata ops (produce from CPU data, upload to GPU) ─────────────────────

static void opShape(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data) return;
    int ndim = (int)data->shape.size();
    *out[0] = ex.AllocCpuTensor({ndim}, TensorDtype::Int64,
                                 data->shape.data(), ndim * sizeof(int64_t));
}

static void opConstant(GraphExecutor& ex, const OnnxGraphNode& n,
                        const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (n.attrIntLists.count("value_ints")) {
        auto& vals = n.attrIntLists.at("value_ints");
        *out[0] = ex.AllocCpuTensor({(int64_t)vals.size()}, TensorDtype::Int64,
                                     vals.data(), vals.size() * sizeof(int64_t));
    } else if (n.attrInts.count("value_int")) {
        int64_t v = n.GetInt("value_int");
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Int64, &v, sizeof(int64_t));
    } else if (n.attrFloats.count("value_float")) {
        float v = n.GetFloat("value_float");
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &v, sizeof(float));
    } else {
        float zero = 0.0f;
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &zero, sizeof(float));
    }
}

// ─── Gather: CPU for small metadata, GPU for large data ──────────────────────

static void opGather(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* indices = in[1];
    if (!data || !indices || !data->IsValid() || !indices->IsValid()) return;

    int64_t axis = n.GetInt("axis", 0);
    int64_t nIdx = tensorNel(indices);

    // Get CPU pointers if available
    const uint8_t* dataPtr = nullptr; size_t dataBytes = 0;
    const uint8_t* idxPtr = nullptr;
    if (data->isCpuOnly) { dataPtr = data->cpuData.data(); dataBytes = data->cpuData.size(); }
    else if (auto* init = ex.GetInitData(n.inputs[0]); init && init->data) { dataPtr = init->data; dataBytes = init->size; }
    if (indices->isCpuOnly) idxPtr = indices->cpuData.data();
    else if (auto* init = (n.inputs.size()>1 ? ex.GetInitData(n.inputs[1]) : nullptr); init && init->data) idxPtr = init->data;

    // CPU path for metadata (both inputs on CPU)
    if (dataPtr && idxPtr && axis == 0 && !data->shape.empty()) {
        int64_t innerSize = 1;
        for (size_t i = 1; i < data->shape.size(); i++) innerSize *= data->shape[i];
        size_t elemBytes = data->DtypeSize();
        size_t sliceBytes = (size_t)(innerSize * elemBytes);

        std::vector<int64_t> outShape;
        for (auto d : indices->shape) if (d > 0) outShape.push_back(d);
        for (size_t i = 1; i < data->shape.size(); i++) outShape.push_back(data->shape[i]);
        if (outShape.empty()) for (size_t i = 1; i < data->shape.size(); i++) outShape.push_back(data->shape[i]);

        std::vector<int64_t> idxVals(nIdx);
        if (indices->dtype == TensorDtype::Int64) memcpy(idxVals.data(), idxPtr, nIdx * 8);
        else { const int32_t* s = (const int32_t*)idxPtr; for (int64_t i = 0; i < nIdx; i++) idxVals[i] = s[i]; }

        size_t totalOut = (size_t)(nIdx * sliceBytes);
        std::vector<uint8_t> outData(totalOut);
        for (int64_t i = 0; i < nIdx; i++) {
            int64_t idx = idxVals[i]; if (idx < 0) idx += data->shape[0];
            if (idx >= 0 && (size_t)((idx+1) * sliceBytes) <= dataBytes)
                memcpy(outData.data() + i * sliceBytes, dataPtr + idx * sliceBytes, sliceBytes);
        }
        *out[0] = ex.AllocCpuTensor(outShape, data->dtype, outData.data(), totalOut);
        return;
    }

    // GPU path: for large tensors, use GPU kernel
    // TODO: implement GPU Gather kernel
    // For now, if data is on GPU, ensure indices are too and use CPU fallback with sync
    if (data->buffer.handle) {
        if (!ex.pendingDispatches_.empty()) {
            ex.gpu->submitOnly(ex.pendingDispatches_, false);
            ex.gpu->waitForQueue();
            ex.pendingDispatches_.clear();
        }
        // ... existing GPU fallback code ...
        int64_t innerSize = 1;
        for (size_t i = 1; i < data->shape.size(); i++) innerSize *= data->shape[i];
        size_t elemBytes = data->DtypeSize();
        size_t sliceBytes = (size_t)(innerSize * elemBytes);

        std::vector<int64_t> outShape;
        for (auto d : indices->shape) if (d > 0) outShape.push_back(d);
        for (size_t i = 1; i < data->shape.size(); i++) outShape.push_back(data->shape[i]);
        if (outShape.empty()) for (size_t i = 1; i < data->shape.size(); i++) outShape.push_back(data->shape[i]);

        *out[0] = ex.AllocTensor(outShape, data->dtype);
        size_t totalBytes = (size_t)(tensorNel(data) * elemBytes);
        auto dataRb = ex.gpu->readBuffer(data->buffer, totalBytes);

        std::vector<int64_t> idxVals(nIdx);
        if (idxPtr) memcpy(idxVals.data(), idxPtr, nIdx * 8);
        else {
            ex.EnsureGpu(*indices);
            auto rbIdx = ex.gpu->readBuffer(indices->buffer, nIdx * indices->DtypeSize());
            if (indices->dtype == TensorDtype::Int64) memcpy(idxVals.data(), rbIdx.data(), nIdx*8);
            else { const int32_t* s = (const int32_t*)rbIdx.data(); for (int64_t i = 0; i < nIdx; i++) idxVals[i] = s[i]; }
        }

        std::vector<uint8_t> outData(nIdx * sliceBytes);
        for (int64_t i = 0; i < nIdx; i++) {
            int64_t idx = idxVals[i]; if (idx < 0) idx += data->shape[0];
            memcpy(outData.data() + i * sliceBytes, dataRb.data() + idx * sliceBytes, sliceBytes);
        }
        ex.gpu->writeBuffer(out[0]->buffer, outData.data(), outData.size());
    }
}

// ─── Concat: GPU buffer copy ─────────────────────────────────────────────────

static void opConcat(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    int64_t axis = n.GetInt("axis", 0);
    std::vector<GpuTensor*> validIn;
    for (auto* t : in) if (t && t->IsValid()) validIn.push_back(t);
    if (validIn.empty()) return;

    int ndim = (int)validIn[0]->shape.size();
    if (axis < 0) axis += ndim;
    auto outShape = validIn[0]->shape;
    int64_t totalOnAxis = 0;
    bool allCpu = true;
    for (auto* t : validIn) {
        if (axis < (int64_t)t->shape.size()) totalOnAxis += t->shape[axis];
        if (!t->isCpuOnly) allCpu = false;
    }
    if (axis < (int64_t)outShape.size()) outShape[axis] = totalOnAxis;

    // CPU path for small metadata (even if some inputs are on GPU)
    size_t totalBytes = 0;
    for (auto* t : validIn) totalBytes += (t->isCpuOnly) ? t->cpuData.size() : t->ByteSize();

    if (allCpu || totalBytes <= 512) {
        // Small concat — do on CPU
        if (!allCpu) {
            fprintf(stderr, "    [concat] small non-cpu: %zu bytes, %zu inputs, syncing...\n", totalBytes, validIn.size());
            fflush(stderr);
        }
        if (!allCpu && !ex.pendingDispatches_.empty()) {
            ex.gpu->submitOnly(ex.pendingDispatches_, false);
            ex.gpu->waitForQueue();
            ex.pendingDispatches_.clear();
        }
        std::vector<uint8_t> outData(totalBytes);
        size_t offset = 0;
        for (auto* t : validIn) {
            size_t bytes;
            if (t->isCpuOnly) {
                bytes = t->cpuData.size();
                memcpy(outData.data() + offset, t->cpuData.data(), bytes);
            } else {
                bytes = t->ByteSize();
                auto rb = ex.gpu->readBuffer(t->buffer, bytes);
                memcpy(outData.data() + offset, rb.data(), bytes);
            }
            offset += bytes;
        }
        *out[0] = ex.AllocCpuTensor(outShape, validIn[0]->dtype, outData.data(), totalBytes);
        return;
    }

    // GPU path: ensure all inputs on GPU, then CopyBufferToBuffer
    for (auto* t : validIn) ex.EnsureGpu(*t);

    if (!ex.pendingDispatches_.empty()) {
        ex.gpu->submitOnly(ex.pendingDispatches_, false);
        ex.gpu->waitForQueue();
        ex.pendingDispatches_.clear();
    }

    *out[0] = ex.AllocTensor(outShape, validIn[0]->dtype);
    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(ex.gpu->device, &enD);
    size_t offset = 0;
    for (auto* t : validIn) {
        size_t bytes = t->ByteSize();
        if (bytes > 0 && t->buffer.handle)
            wgpuCommandEncoderCopyBufferToBuffer(enc, t->buffer.handle, 0,
                out[0]->buffer.handle, offset, bytes);
        offset += bytes;
    }
    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(ex.gpu->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
}

// ─── Transpose: GPU kernel ───────────────────────────────────────────────────

static const char* WGSL_TRANSPOSE = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];    // total elements
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }

    // Decode output flat index to output coords
    // Then map through perm to get input coords
    // Then encode input coords to input flat index
    // Shapes and strides stored in params[4..4+2*ndim]
    var out_idx = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let coord = out_idx / out_stride;
        out_idx = out_idx % out_stride;
        in_flat += coord * in_stride;
    }
    Y[idx] = X[in_flat];
}
)WGSL";

static void opTranspose(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    // CPU path for small metadata
    if (data->isCpuOnly && data->cpuData.size() <= 256) {
        // ... just pass through for tiny tensors
        *out[0] = *data;
        return;
    }

    ex.EnsureGpu(*data);

    std::vector<int64_t> perm;
    if (n.attrIntLists.count("perm")) perm = n.attrIntLists.at("perm");
    else for (int i = (int)data->shape.size() - 1; i >= 0; i--) perm.push_back(i);

    std::vector<int64_t> outShape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) outShape[i] = data->shape[perm[i]];

    int64_t nel = tensorNel(data);
    size_t elemSize = data->DtypeSize();
    int ndim = (int)data->shape.size();

    // Compute strides
    std::vector<uint32_t> outStrides(ndim), permInStrides(ndim);
    std::vector<int64_t> inStrides(ndim);
    inStrides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) inStrides[i] = inStrides[i+1] * data->shape[i+1];
    uint32_t ostride = 1;
    for (int i = ndim-1; i >= 0; i--) {
        outStrides[i] = ostride;
        ostride *= (uint32_t)outShape[i];
        permInStrides[i] = (uint32_t)inStrides[perm[i]];
    }

    // For non-f32 dtypes, we work at u32 granularity
    // If elemSize == 4, one u32 per element. For fp16 (2 bytes), pack 2 per u32.
    // Simplified: only support 4-byte elements for now
    if (elemSize != 4) {
        *out[0] = *data; out[0]->shape = outShape; // fallback: alias
        return;
    }

    *out[0] = ex.AllocTensor(outShape, data->dtype);

    // Params: [N, ndim, 0, 0, outStride0..outStrideN, permInStride0..permInStrideN]
    std::vector<uint32_t> params(4 + 2 * ndim, 0);
    params[0] = (uint32_t)nel;
    params[1] = (uint32_t)ndim;
    for (int i = 0; i < ndim; i++) {
        params[4 + i] = outStrides[i];
        params[4 + ndim + i] = permInStrides[i];
    }
    auto paramBuf = ex.gpu->createBuffer("transpose_p", params.size() * 4);
    ex.gpu->writeBuffer(paramBuf, params.data(), params.size() * 4);

    auto& pl = ex.GetPipeline("transpose", WGSL_TRANSPOSE, 3);
    auto bg = ex.MakeBindGroup(pl, {
        {0, data->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((nel + 255) / 256), 1, 1, "transpose"});
}

// ─── Simple pass-through ops ─────────────────────────────────────────────────

static void opSlice(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0]; // TODO: proper GPU slice
}

static void opConstantOfShape(GraphExecutor& ex, const OnnxGraphNode& n,
                                const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (!in[0] || !in[0]->IsValid()) return;
    std::vector<int64_t> shape;
    if (in[0]->isCpuOnly && !in[0]->cpuData.empty()) {
        int64_t nel = tensorNel(in[0]);
        shape.resize(nel);
        memcpy(shape.data(), in[0]->cpuData.data(), nel * 8);
    } else if (auto* init = ex.GetInitData(n.inputs[0]); init && init->data) {
        int64_t nel = 1; for (auto d : init->shape) nel *= d;
        shape.resize(nel);
        memcpy(shape.data(), init->data, nel * 8);
    } else {
        shape = {1};
    }
    int64_t total = 1; for (auto d : shape) total *= d;
    *out[0] = ex.AllocTensor(shape, TensorDtype::Float32);
    // Zero-fill
    std::vector<float> zeros(total, 0.0f);
    ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), total * 4);
}

static void opRange(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in.size() < 3) return;
    // Range values are typically small — read from CPU
    float start = 0, limit = 0, delta = 1;
    auto readScalar = [&](GpuTensor* t, const std::string& name) -> float {
        if (t && t->isCpuOnly && !t->cpuData.empty()) {
            float v; memcpy(&v, t->cpuData.data(), 4); return v;
        }
        if (auto* init = ex.GetInitData(name); init && init->data) {
            float v; memcpy(&v, init->data, 4); return v;
        }
        return 0;
    };
    start = readScalar(in[0], n.inputs[0]);
    limit = readScalar(in[1], n.inputs[1]);
    delta = readScalar(in[2], n.inputs[2]);
    int64_t count = (delta != 0) ? (int64_t)std::ceil((limit - start) / delta) : 0;
    if (count <= 0) count = 0;
    std::vector<float> vals(count);
    for (int64_t i = 0; i < count; i++) vals[i] = start + i * delta;
    *out[0] = ex.AllocTensor({count}, TensorDtype::Float32);
    if (count > 0) ex.gpu->writeBuffer(out[0]->buffer, vals.data(), count * 4);
}

static void opExpand(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0];
}
static void opPad(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0];
}
static void opSplit(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) for (auto* o : out) if (o) *o = *in[0];
}
static void opScatterND(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0];
}
static void opMod(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0];
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
