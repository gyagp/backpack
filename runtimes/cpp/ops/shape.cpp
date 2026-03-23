/**
 * ops/shape.cpp — Shape manipulation ONNX ops.
 * ALL GPU, ZERO CPU readback during graph execution.
 *
 * Zero-copy: Reshape, Squeeze, Unsqueeze, Flatten.
 * GPU kernels: Transpose, Gather, Concat.
 * CPU init only: Shape, Constant (produce from known metadata, upload to GPU).
 */

#include "../graph_executor.h"
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
        }
    }
    if (newShape.empty()) { *out[0] = *data; return; }

    int64_t totalIn = tensorNel(data);
    int64_t known = 1; int inferIdx = -1;
    for (int i = 0; i < (int)newShape.size(); i++) {
        if (newShape[i] == 0 && i < (int)data->shape.size()) newShape[i] = data->shape[i];
        if (newShape[i] == -1) inferIdx = i; else known *= newShape[i];
    }
    if (inferIdx >= 0 && known > 0) newShape[inferIdx] = totalIn / known;
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
// GPU Gather kernel
// ═══════════════════════════════════════════════════════════════════════════

static const char* WGSL_GATHER = R"WGSL(
@group(0) @binding(0) var<storage, read> Data: array<u32>;
@group(0) @binding(1) var<storage, read> Indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let sliceSize = _params_[1];  // elements per slice (u32 units)
    let dataStride = _params_[2]; // stride on axis 0

    let total = nIdx * sliceSize;
    let idx = gid.x;
    if (idx >= total) { return; }

    let i = idx / sliceSize;
    let j = idx % sliceSize;
    let dataIdx = u32(Indices[i]) * dataStride + j;
    Out[idx] = Data[dataIdx];
}
)WGSL";

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

    auto& pl = ex.GetPipeline("gather_gpu", WGSL_GATHER, 4);
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
    int64_t axis = n.GetInt("axis", 0);
    std::vector<GpuTensor*> validIn;
    for (auto* t : in) if (t && t->IsValid()) validIn.push_back(t);
    if (validIn.empty()) return;

    int ndim = (int)validIn[0]->shape.size();
    if (axis < 0) axis += ndim;
    auto outShape = validIn[0]->shape;
    int64_t totalOnAxis = 0;
    for (auto* t : validIn)
        if (axis < (int64_t)t->shape.size()) totalOnAxis += t->shape[axis];
    if (axis < (int64_t)outShape.size()) outShape[axis] = totalOnAxis;

    for (auto* t : validIn) ex.EnsureGpu(*t);
    *out[0] = ex.AllocTensor(outShape, validIn[0]->dtype);

    // Queue GPU copies — batched with compute dispatches, no separate submit
    size_t offset = 0;
    for (auto* t : validIn) {
        size_t bytes = t->ByteSize();
        if (bytes > 0 && t->buffer.handle)
            ex.QueueCopy(t->buffer, 0, out[0]->buffer, offset, bytes);
        offset += bytes;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Transpose kernel
// ═══════════════════════════════════════════════════════════════════════════

static const char* WGSL_TRANSPOSE = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
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
    auto* data = in[0]; if (!data || !data->IsValid()) return;
    if (data->isCpuOnly) { *out[0] = *data; return; }
    ex.EnsureGpu(*data);

    std::vector<int64_t> perm;
    if (n.attrIntLists.count("perm")) perm = n.attrIntLists.at("perm");
    else for (int i=(int)data->shape.size()-1; i>=0; i--) perm.push_back(i);

    std::vector<int64_t> outShape(perm.size());
    for (size_t i=0; i<perm.size(); i++) outShape[i] = data->shape[perm[i]];
    int64_t nel = tensorNel(data);
    size_t elemSize = data->DtypeSize();
    int ndim = (int)data->shape.size();

    if (elemSize != 4 && elemSize != 8) {
        // For fp16 (2 bytes), can't directly use u32 transpose. Pass through.
        *out[0] = *data; out[0]->shape = outShape; return;
    }

    *out[0] = ex.AllocTensor(outShape, data->dtype);

    std::vector<int64_t> inStrides(ndim);
    inStrides[ndim-1] = 1;
    for (int i=ndim-2; i>=0; i--) inStrides[i] = inStrides[i+1] * data->shape[i+1];

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

static void opSlice(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) *out[0] = *in[0]; // TODO: proper GPU slice
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

static void opExpand(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) { ex.EnsureGpu(*in[0]); *out[0] = *in[0]; }
}
static void opPad(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) { ex.EnsureGpu(*in[0]); *out[0] = *in[0]; }
}
static void opSplit(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) { ex.EnsureGpu(*in[0]); for (auto* o : out) if (o) *o = *in[0]; }
}
static void opScatterND(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) { ex.EnsureGpu(*in[0]); *out[0] = *in[0]; }
}
static void opMod(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) { ex.EnsureGpu(*in[0]); *out[0] = *in[0]; }
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
