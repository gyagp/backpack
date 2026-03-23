/**
 * ops/shape.cpp — Shape manipulation ONNX ops.
 *
 * Zero-copy: Reshape, Squeeze, Unsqueeze, Flatten — same buffer, new shape.
 * Copy: Transpose, Slice, Concat, Gather, Pad, Expand.
 * CPU: Shape, Constant, ConstantOfShape, Range, Split, Mod.
 */

#include "../graph_executor.h"
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

    // Flush before reading shape from GPU
    if (shape && shape->IsValid() && !ex.pendingDispatches_.empty()) {
        ex.gpu->submitOnly(ex.pendingDispatches_, false);
        ex.gpu->waitForQueue();
        ex.pendingDispatches_.clear();
    }

    // Read shape from initializer (CPU) if possible, else from GPU
    std::vector<int64_t> newShape;
    if (shape && shape->IsValid()) {
        int64_t nel = tensorNel(shape);
        newShape.resize(nel);

        // Try CPU initializer first (no GPU sync needed)
        bool fromCPU = false;
        if (n.inputs.size() > 1) {
            auto* initData = ex.GetInitData(n.inputs[1]);
            if (initData && initData->data) {
                memcpy(newShape.data(), initData->data, nel * sizeof(int64_t));
                fromCPU = true;
            }
        }
        if (!fromCPU) {
            // Flush before reading shape from GPU
            if (!ex.pendingDispatches_.empty()) {
                ex.gpu->submitOnly(ex.pendingDispatches_, false);
                ex.gpu->waitForQueue();
                ex.pendingDispatches_.clear();
            }
            auto readback = ex.gpu->readBuffer(shape->buffer, nel * sizeof(int64_t));
            memcpy(newShape.data(), readback.data(), nel * sizeof(int64_t));
        }
    }

    // Handle -1 (infer) and 0 (keep) dimensions
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

    // Zero-copy: same buffer, new shape
    *out[0] = *data;
    out[0]->shape = newShape;
}

static void opSqueeze(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    // Get axes to squeeze
    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1] && in[1]->IsValid()) {
        int64_t nel = tensorNel(in[1]);
        axes.resize(nel);
        auto rb = ex.gpu->readBuffer(in[1]->buffer, nel * sizeof(int64_t));
        memcpy(axes.data(), rb.data(), nel * sizeof(int64_t));
    } else if (n.attrIntLists.count("axes")) {
        axes = n.attrIntLists.at("axes");
    }

    std::vector<int64_t> newShape;
    for (int i = 0; i < (int)data->shape.size(); i++) {
        bool squeeze = false;
        if (axes.empty()) {
            squeeze = (data->shape[i] == 1);
        } else {
            for (auto a : axes) {
                if (a < 0) a += data->shape.size();
                if (a == i) { squeeze = true; break; }
            }
        }
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
        auto rb = ex.gpu->readBuffer(in[1]->buffer, nel * sizeof(int64_t));
        memcpy(axes.data(), rb.data(), nel * sizeof(int64_t));
    } else if (n.attrIntLists.count("axes")) {
        axes = n.attrIntLists.at("axes");
    }

    int ndim = (int)data->shape.size() + (int)axes.size();
    std::vector<int64_t> newShape(ndim);
    // Normalize negative axes
    for (auto& a : axes) if (a < 0) a += ndim;
    std::sort(axes.begin(), axes.end());

    int srcIdx = 0;
    for (int i = 0; i < ndim; i++) {
        bool isNew = std::find(axes.begin(), axes.end(), i) != axes.end();
        if (isNew) newShape[i] = 1;
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

    int64_t dim0 = 1, dim1 = 1;
    for (int i = 0; i < axis; i++) dim0 *= data->shape[i];
    for (int i = (int)axis; i < (int)data->shape.size(); i++) dim1 *= data->shape[i];

    *out[0] = *data;
    out[0]->shape = {dim0, dim1};
}

// ─── CPU-side ops (produce small tensors) ────────────────────────────────────

static void opConstant(GraphExecutor& ex, const OnnxGraphNode& n,
                        const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Constant nodes have their data in attributes
    // For now, create a small tensor. The actual data should come from the node's
    // "value" attribute (a TensorProto). This is complex to parse from attributes.
    // Simplified: create a scalar 0 tensor.
    if (n.attrIntLists.count("value_ints")) {
        auto& vals = n.attrIntLists.at("value_ints");
        *out[0] = ex.AllocTensor({(int64_t)vals.size()}, TensorDtype::Int64);
        ex.gpu->writeBuffer(out[0]->buffer, vals.data(), vals.size() * sizeof(int64_t));
    } else if (n.attrInts.count("value_int")) {
        int64_t v = n.GetInt("value_int");
        *out[0] = ex.AllocTensor({1}, TensorDtype::Int64);
        ex.gpu->writeBuffer(out[0]->buffer, &v, sizeof(int64_t));
    } else if (n.attrFloats.count("value_float")) {
        float v = n.GetFloat("value_float");
        *out[0] = ex.AllocTensor({1}, TensorDtype::Float32);
        ex.gpu->writeBuffer(out[0]->buffer, &v, sizeof(float));
    } else {
        // Generic: allocate a scalar zero
        *out[0] = ex.AllocTensor({1}, TensorDtype::Float32);
        float zero = 0.0f;
        ex.gpu->writeBuffer(out[0]->buffer, &zero, 4);
    }
}

static void opShape(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data) return;
    int ndim = (int)data->shape.size();
    *out[0] = ex.AllocTensor({ndim}, TensorDtype::Int64);
    ex.gpu->writeBuffer(out[0]->buffer, data->shape.data(), ndim * sizeof(int64_t));
}

static void opGather(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* indices = in[1];
    if (!data || !indices || !data->IsValid() || !indices->IsValid()) return;

    // Flush pending GPU work before CPU readback (only if needed)
    bool needsFlush = true;

    // Try reading indices from CPU initializer (no GPU sync)
    auto* idxInit = (n.inputs.size() > 1) ? ex.GetInitData(n.inputs[1]) : nullptr;
    if (idxInit && idxInit->data) needsFlush = false;
    auto* dataInit = ex.GetInitData(n.inputs[0]);
    if (dataInit && dataInit->data) needsFlush = false;

    if (needsFlush && !ex.pendingDispatches_.empty()) {
        ex.gpu->submitOnly(ex.pendingDispatches_, false);
        ex.gpu->waitForQueue();
        ex.pendingDispatches_.clear();
    }

    int64_t axis = n.GetInt("axis", 0);
    int64_t nIdx = tensorNel(indices);

    // Read indices
    std::vector<int64_t> idxData(nIdx);
    auto rb = ex.gpu->readBuffer(indices->buffer, nIdx * indices->DtypeSize());
    if (indices->dtype == TensorDtype::Int64) {
        memcpy(idxData.data(), rb.data(), nIdx * 8);
    } else if (indices->dtype == TensorDtype::Int32) {
        const int32_t* src = (const int32_t*)rb.data();
        for (int64_t i = 0; i < nIdx; i++) idxData[i] = src[i];
    }

    // For scalar index (nIdx=1) on axis 0: output = data[idx, ...]
    if (axis == 0 && data->shape.size() >= 1) {
        int64_t innerSize = 1;
        for (size_t i = 1; i < data->shape.size(); i++) innerSize *= data->shape[i];
        size_t elemBytes = data->DtypeSize();
        size_t sliceBytes = (size_t)(innerSize * elemBytes);

        std::vector<int64_t> outShape;
        // Output shape: indices_shape + data_shape[1:]
        for (auto d : indices->shape) if (d > 0) outShape.push_back(d);
        for (size_t i = 1; i < data->shape.size(); i++) outShape.push_back(data->shape[i]);
        if (outShape.empty()) {
            for (size_t i = 1; i < data->shape.size(); i++) outShape.push_back(data->shape[i]);
        }

        *out[0] = ex.AllocTensor(outShape, data->dtype);

        // Read all data and write selected slices
        size_t totalBytes = (size_t)(tensorNel(data) * elemBytes);
        auto dataRb = ex.gpu->readBuffer(data->buffer, totalBytes);
        std::vector<uint8_t> outData(nIdx * sliceBytes);
        for (int64_t i = 0; i < nIdx; i++) {
            int64_t idx = idxData[i];
            if (idx < 0) idx += data->shape[0];
            memcpy(outData.data() + i * sliceBytes,
                   dataRb.data() + idx * sliceBytes, sliceBytes);
        }
        ex.gpu->writeBuffer(out[0]->buffer, outData.data(), outData.size());
    } else {
        // Fallback: copy input as-is
        *out[0] = *data;
    }
}

static void opConcat(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    int64_t axis = n.GetInt("axis", 0);

    // Collect valid inputs
    std::vector<GpuTensor*> validIn;
    for (auto* t : in) if (t && t->IsValid()) validIn.push_back(t);
    if (validIn.empty()) return;

    // Normalize axis
    int ndim = (int)validIn[0]->shape.size();
    if (axis < 0) axis += ndim;

    // Compute output shape
    auto outShape = validIn[0]->shape;
    int64_t totalOnAxis = 0;
    for (auto* t : validIn) {
        if (axis < (int64_t)t->shape.size())
            totalOnAxis += t->shape[axis];
    }
    if (axis < (int64_t)outShape.size())
        outShape[axis] = totalOnAxis;

    // Flush pending before GPU copy
    if (!ex.pendingDispatches_.empty()) {
        ex.gpu->submitOnly(ex.pendingDispatches_, false);
        ex.gpu->waitForQueue();
        ex.pendingDispatches_.clear();
    }

    *out[0] = ex.AllocTensor(outShape, validIn[0]->dtype);

    // GPU-to-GPU copy: use CopyBufferToBuffer for each input
    // For simple concat on outer axis, data is contiguous
    size_t offset = 0;
    for (auto* t : validIn) {
        size_t bytes = t->ByteSize();
        if (bytes > 0 && t->buffer.handle && out[0]->buffer.handle) {
            // Use WebGPU buffer copy (GPU-side, no CPU readback)
            WGPUCommandEncoderDescriptor enD{};
            auto enc = wgpuDeviceCreateCommandEncoder(ex.gpu->device, &enD);
            wgpuCommandEncoderCopyBufferToBuffer(enc, t->buffer.handle, 0,
                out[0]->buffer.handle, offset, bytes);
            WGPUCommandBufferDescriptor cbD{};
            auto cb = wgpuCommandEncoderFinish(enc, &cbD);
            wgpuQueueSubmit(ex.gpu->queue, 1, &cb);
            wgpuCommandBufferRelease(cb);
            wgpuCommandEncoderRelease(enc);
        }
        offset += bytes;
    }
    ex.gpu->waitForQueue();
}

static void opTranspose(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    // Flush before CPU readback
    if (!ex.pendingDispatches_.empty()) {
        ex.gpu->submitOnly(ex.pendingDispatches_, false);
        ex.gpu->waitForQueue();
        ex.pendingDispatches_.clear();
    }

    std::vector<int64_t> perm;
    if (n.attrIntLists.count("perm")) perm = n.attrIntLists.at("perm");
    else {
        // Default: reverse dimensions
        for (int i = (int)data->shape.size() - 1; i >= 0; i--) perm.push_back(i);
    }

    // Compute output shape
    std::vector<int64_t> outShape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) outShape[i] = data->shape[perm[i]];

    int64_t nel = tensorNel(data);
    *out[0] = ex.AllocTensor(outShape, data->dtype);

    // CPU transpose (TODO: GPU kernel for large tensors)
    size_t elemSize = data->DtypeSize();
    auto rb = ex.gpu->readBuffer(data->buffer, nel * elemSize);

    int ndim = (int)data->shape.size();
    std::vector<int64_t> inStrides(ndim), outStrides(ndim);
    inStrides[ndim-1] = 1; outStrides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) {
        inStrides[i] = inStrides[i+1] * data->shape[i+1];
        outStrides[i] = outStrides[i+1] * outShape[i+1];
    }

    std::vector<uint8_t> outBuf(nel * elemSize);
    for (int64_t i = 0; i < nel; i++) {
        // Convert flat index to coords
        int64_t tmp = i;
        std::vector<int64_t> outCoords(ndim);
        for (int d = 0; d < ndim; d++) {
            outCoords[d] = tmp / outStrides[d];
            tmp %= outStrides[d];
        }
        // Map to input coords
        int64_t inIdx = 0;
        for (int d = 0; d < ndim; d++) {
            inIdx += outCoords[d] * inStrides[perm[d]];
        }
        memcpy(outBuf.data() + i * elemSize, rb.data() + inIdx * elemSize, elemSize);
    }
    ex.gpu->writeBuffer(out[0]->buffer, outBuf.data(), outBuf.size());
}

static void opSlice(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;
    // Simplified: pass-through (TODO: implement proper slicing)
    *out[0] = *data;
}

static void opConstantOfShape(GraphExecutor& ex, const OnnxGraphNode& n,
                                const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Read shape from input
    if (!in[0] || !in[0]->IsValid()) return;
    int64_t nel = tensorNel(in[0]);
    std::vector<int64_t> shape(nel);
    auto rb = ex.gpu->readBuffer(in[0]->buffer, nel * sizeof(int64_t));
    memcpy(shape.data(), rb.data(), nel * sizeof(int64_t));

    *out[0] = ex.AllocTensor(shape, TensorDtype::Float32);
    // Fill with zeros (or value from attr)
    int64_t total = 1; for (auto d : shape) total *= d;
    std::vector<float> zeros(total, 0.0f);
    ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), total * sizeof(float));
}

static void opRange(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Range(start, limit, delta)
    if (in.size() < 3) return;
    float start = 0, limit = 0, delta = 1;
    auto rb0 = ex.gpu->readBuffer(in[0]->buffer, 4); memcpy(&start, rb0.data(), 4);
    auto rb1 = ex.gpu->readBuffer(in[1]->buffer, 4); memcpy(&limit, rb1.data(), 4);
    auto rb2 = ex.gpu->readBuffer(in[2]->buffer, 4); memcpy(&delta, rb2.data(), 4);

    int64_t count = (int64_t)std::ceil((limit - start) / delta);
    if (count <= 0) count = 0;
    std::vector<float> vals(count);
    for (int64_t i = 0; i < count; i++) vals[i] = start + i * delta;

    *out[0] = ex.AllocTensor({count}, TensorDtype::Float32);
    if (count > 0) ex.gpu->writeBuffer(out[0]->buffer, vals.data(), count * sizeof(float));
}

static void opExpand(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Simplified: just pass through (broadcasting handled by consumers)
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0];
}

static void opPad(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0]; // TODO
}

static void opSplit(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        for (auto* o : out) if (o) *o = *in[0]; // TODO
    }
}

static void opScatterND(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0]; // TODO
}

static void opMod(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // TODO: implement modulo
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0];
}

// ─── Register all shape ops ──────────────────────────────────────────────────

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
