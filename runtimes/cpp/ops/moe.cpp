/**
 * ops/moe.cpp — Mixture of Experts ops for LFM2-style models.
 *
 * All ops run on GPU:
 *   TopK          — GPU kernel for small routing tensors
 *   GatherElements — GPU kernel for indexed gathers
 *   ScatterElements — GPU kernel (copy + scatter passes)
 *   QMoE          — Fused GPU pipeline: per-expert Q4 matmul + SwiGLU + accumulate
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// ─── Helpers ─────────────────────────────────────────────────────────────────

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0;
    return t->ElementCount();
}

static float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

static bool readTensorIntValues(GraphExecutor& ex, const GpuTensor* t,
                                const std::string& name, std::vector<int64_t>& values) {
    values.clear();
    if (!t) return false;
    int64_t nel = tensorNel(t);
    if (nel <= 0) return false;

    const uint8_t* src = nullptr;
    std::vector<uint8_t> gpuReadback;
    if (t->isCpuOnly && !t->cpuData.empty()) {
        src = t->cpuData.data();
    } else if (!t->cpuData.empty()) {
        src = t->cpuData.data();
    } else if (auto* init = ex.GetInitData(name); init && init->data) {
        src = init->data;
    } else if (t->buffer.handle) {
        ex.FlushPendingWork();
        gpuReadback = ex.gpu->readBuffer(t->buffer, (size_t)nel * t->DtypeSize());
        src = gpuReadback.data();
    }
    if (!src) return false;

    values.resize((size_t)nel);
    if (t->dtype == TensorDtype::Int64) {
        memcpy(values.data(), src, (size_t)nel * sizeof(int64_t));
        return true;
    }
    if (t->dtype == TensorDtype::Int32) {
        auto* i32 = reinterpret_cast<const int32_t*>(src);
        for (int64_t i = 0; i < nel; i++) values[(size_t)i] = i32[i];
        return true;
    }
    return false;
}

static bool loadTensorFloats(GraphExecutor& ex, const GpuTensor* t,
                             const std::string& name, std::vector<float>& values) {
    values.clear();
    if (!t) return false;
    int64_t nel = tensorNel(t);
    if (nel <= 0) return false;
    values.resize((size_t)nel);

    const uint8_t* src = nullptr;
    std::vector<uint8_t> gpuReadback;
    if (t->isCpuOnly && !t->cpuData.empty()) {
        src = t->cpuData.data();
    } else if (!t->cpuData.empty()) {
        src = t->cpuData.data();
    } else if (auto* init = ex.GetInitData(name); init && init->data) {
        src = init->data;
    } else if (t->buffer.handle) {
        ex.FlushPendingWork();
        gpuReadback = ex.gpu->readBuffer(t->buffer, (size_t)nel * t->DtypeSize());
        src = gpuReadback.data();
    }
    if (!src) return false;

    if (t->dtype == TensorDtype::Float32) {
        memcpy(values.data(), src, (size_t)nel * sizeof(float));
        return true;
    }
    if (t->dtype == TensorDtype::Float16) {
        auto* fp16 = reinterpret_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < nel; i++) values[(size_t)i] = fp16ToFloat(fp16[i]);
        return true;
    }
    return false;
}

static bool ensureTensorFloat32(GraphExecutor& ex, GpuTensor& tensor, const std::string& name) {
    if (tensor.dtype == TensorDtype::Float32) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }
    if (tensor.dtype != TensorDtype::Float16) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }
    size_t count = (size_t)tensor.ElementCount();
    size_t bytes = count * sizeof(uint16_t);
    const uint8_t* src = nullptr;
    std::vector<uint8_t> raw;
    if (!tensor.cpuData.empty() && tensor.cpuData.size() >= bytes) {
        src = tensor.cpuData.data();
    } else if (auto* init = ex.GetInitData(name); init && init->data && init->size >= bytes) {
        src = init->data;
    } else if (tensor.buffer.handle && tensor.buffer.size >= bytes) {
        ex.FlushPendingWork();
        raw = ex.gpu->readBuffer(tensor.buffer, bytes);
        src = raw.data();
    }
    if (!src) return false;

    std::vector<float> f32(count);
    auto* fp16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < count; i++) f32[i] = fp16ToFloat(fp16[i]);
    tensor.shape = tensor.shape;
    tensor.dtype = TensorDtype::Float32;
    tensor.cpuData.clear();
    tensor.buffer = ex.gpu->createBuffer("moe_f32", f32.size() * 4);
    ex.gpu->writeBuffer(tensor.buffer, f32.data(), f32.size() * 4);
    tensor.isCpuOnly = false;
    return true;
}

// ─── TopK (GPU) ──────────────────────────────────────────────────────────────
// GPU kernel for TopK on fp16 data. For MoE routing: dimSize=32, k=4.

static void opTopK(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    if (!data || !data->IsValid()) return;

    // Read k from input[1] or attribute
    int64_t k = n.GetInt("k", 1);
    if (in.size() > 1 && in[1] && in[1]->IsValid()) {
        std::vector<int64_t> kVals;
        if (readTensorIntValues(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", kVals) && !kVals.empty()) {
            k = kVals[0];
        }
    }

    int64_t axis = n.GetInt("axis", -1);
    int64_t largest = n.GetInt("largest", 1);

    int ndim = (int)data->shape.size();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) axis = ndim - 1;

    int64_t dimSize = data->shape[axis];
    k = std::min(k, dimSize);

    int64_t totalSlices = 1;
    for (int i = 0; i < ndim; i++) {
        if (i != (int)axis) totalSlices *= data->shape[i];
    }

    auto outShape = data->shape;
    outShape[axis] = k;
    int64_t outNel = 1;
    for (auto d : outShape) outNel *= d;

    // GPU path for fp16 data along last axis (the MoE routing case)
    if (data->dtype == TensorDtype::Float16 && axis == ndim - 1 && ex.gpu->supportsShaderF16) {
        ex.EnsureGpu(*data);

        *out[0] = ex.AllocTensor(outShape, TensorDtype::Float16);
        GpuTensor idxTensor = ex.AllocTensor(outShape, TensorDtype::Int32);

        uint32_t params[4] = {(uint32_t)totalSlices, (uint32_t)dimSize, (uint32_t)k, (uint32_t)largest};
        auto paramBuf = ex.gpu->createBuffer("topk_p", 16);
        ex.gpu->writeBuffer(paramBuf, params, 16);

        auto& pl = ex.GetPipeline("topk_f16", WGSL_TOPK_F16, 4);
        auto bg = ex.MakeBindGroup(pl, {
            {0, data->buffer}, {1, out[0]->buffer},
            {2, idxTensor.buffer}, {3, paramBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)totalSlices, 1, 1, "topk_f16"});

        *out[1] = idxTensor;
        out[1]->dtype = TensorDtype::Int32;
        return;
    }

    // GPU path for f32 data along last axis
    if (data->dtype == TensorDtype::Float32 && axis == ndim - 1) {
        ex.EnsureGpu(*data);

        *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);
        GpuTensor idxTensor = ex.AllocTensor(outShape, TensorDtype::Int32);

        uint32_t params[4] = {(uint32_t)totalSlices, (uint32_t)dimSize, (uint32_t)k, (uint32_t)largest};
        auto paramBuf = ex.gpu->createBuffer("topk_p", 16);
        ex.gpu->writeBuffer(paramBuf, params, 16);

        auto& pl = ex.GetPipeline("topk_f32", WGSL_TOPK_F32, 4);
        auto bg = ex.MakeBindGroup(pl, {
            {0, data->buffer}, {1, out[0]->buffer},
            {2, idxTensor.buffer}, {3, paramBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)totalSlices, 1, 1, "topk_f32"});

        *out[1] = idxTensor;
        out[1]->dtype = TensorDtype::Int32;
        return;
    }

    // CPU fallback for non-fp16 or non-last-axis
    std::vector<float> values;
    if (!loadTensorFloats(ex, data, n.inputs.empty() ? "" : n.inputs[0], values) || values.empty()) {
        return;
    }

    int64_t innerSize = 1;
    for (int i = (int)axis + 1; i < ndim; i++) innerSize *= data->shape[i];
    int64_t outerSize = totalSlices / std::max<int64_t>(1, innerSize);

    std::vector<float> outValues(outNel);
    std::vector<int64_t> outIndices(outNel);

    for (int64_t o = 0; o < outerSize; o++) {
        for (int64_t i = 0; i < innerSize; i++) {
            std::vector<std::pair<float, int64_t>> items(dimSize);
            for (int64_t d = 0; d < dimSize; d++) {
                int64_t srcIdx = (o * dimSize + d) * innerSize + i;
                items[d] = {values[srcIdx], d};
            }
            if (largest) {
                std::partial_sort(items.begin(), items.begin() + k, items.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::partial_sort(items.begin(), items.begin() + k, items.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            for (int64_t j = 0; j < k; j++) {
                int64_t dstIdx = (o * k + j) * innerSize + i;
                outValues[dstIdx] = items[j].first;
                outIndices[dstIdx] = items[j].second;
            }
        }
    }

    *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Float32, outValues.data(), outNel * 4);
    *out[1] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, outIndices.data(), outNel * 8);
}

// ─── GatherElements (GPU) ────────────────────────────────────────────────────

static void opGatherElements(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* indices = in.size() > 1 ? in[1] : nullptr;
    if (!data || !indices || !data->IsValid() || !indices->IsValid()) return;

    int64_t axis = n.GetInt("axis", 0);
    int ndim = (int)data->shape.size();
    if (axis < 0) axis += ndim;

    int64_t outNel = tensorNel(indices);

    // GPU path for fp16 data with i32 indices, axis = last dim
    if (data->dtype == TensorDtype::Float16 &&
        indices->dtype == TensorDtype::Int32 &&
        axis == ndim - 1 && ex.gpu->supportsShaderF16) {
        ex.EnsureGpu(*data);
        ex.EnsureGpu(*indices);

        *out[0] = ex.AllocTensor(indices->shape, TensorDtype::Float16);

        uint32_t params[4] = {(uint32_t)outNel, (uint32_t)data->shape[axis],
                               (uint32_t)indices->shape[axis], 0};
        auto paramBuf = ex.gpu->createBuffer("ge_p", 16);
        ex.gpu->writeBuffer(paramBuf, params, 16);

        auto& pl = ex.GetPipeline("gather_elements_f16", WGSL_GATHER_ELEMENTS_F16, 4);
        auto bg = ex.MakeBindGroup(pl, {
            {0, data->buffer}, {1, indices->buffer},
            {2, out[0]->buffer}, {3, paramBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)((outNel + 255) / 256), 1, 1, "gather_elements_f16"});
        return;
    }

    // GPU path for f32 data with i32 indices, axis = last dim
    if (data->dtype == TensorDtype::Float32 &&
        indices->dtype == TensorDtype::Int32 &&
        axis == ndim - 1) {
        ex.EnsureGpu(*data);
        ex.EnsureGpu(*indices);

        *out[0] = ex.AllocTensor(indices->shape, TensorDtype::Float32);

        uint32_t params[4] = {(uint32_t)outNel, (uint32_t)data->shape[axis],
                               (uint32_t)indices->shape[axis], 0};
        auto paramBuf = ex.gpu->createBuffer("ge_p", 16);
        ex.gpu->writeBuffer(paramBuf, params, 16);

        auto& pl = ex.GetPipeline("gather_elements_f32", WGSL_GATHER_ELEMENTS_F32, 4);
        auto bg = ex.MakeBindGroup(pl, {
            {0, data->buffer}, {1, indices->buffer},
            {2, out[0]->buffer}, {3, paramBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)((outNel + 255) / 256), 1, 1, "gather_elements_f32"});
        return;
    }

    // CPU fallback
    std::vector<float> dataVals;
    std::vector<int64_t> idxVals;
    if (!loadTensorFloats(ex, data, n.inputs[0], dataVals)) return;
    if (!readTensorIntValues(ex, indices, n.inputs[1], idxVals)) return;

    std::vector<float> result(outNel);
    std::vector<int64_t> dataStrides(ndim, 1);
    for (int i = ndim - 2; i >= 0; i--) dataStrides[i] = dataStrides[i+1] * data->shape[i+1];
    std::vector<int64_t> idxStrides(ndim, 1);
    for (int i = ndim - 2; i >= 0; i--) idxStrides[i] = idxStrides[i+1] * indices->shape[i+1];

    for (int64_t flat = 0; flat < outNel; flat++) {
        int64_t remaining = flat;
        std::vector<int64_t> multiIdx(ndim);
        for (int d = 0; d < ndim; d++) {
            multiIdx[d] = remaining / idxStrides[d];
            remaining %= idxStrides[d];
        }
        int64_t idx = idxVals[flat];
        if (idx < 0) idx += data->shape[axis];
        multiIdx[axis] = idx;
        int64_t dataFlat = 0;
        for (int d = 0; d < ndim; d++) dataFlat += multiIdx[d] * dataStrides[d];
        result[flat] = dataVals[dataFlat];
    }

    *out[0] = ex.AllocCpuTensor(indices->shape, TensorDtype::Float32, result.data(), outNel * 4);
    ex.EnsureGpu(*out[0]);
}

// ─── ScatterElements (GPU) ───────────────────────────────────────────────────

static void opScatterElements(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* data = in[0];
    auto* indices = in.size() > 1 ? in[1] : nullptr;
    auto* updates = in.size() > 2 ? in[2] : nullptr;
    if (!data || !indices || !updates ||
        !data->IsValid() || !indices->IsValid() || !updates->IsValid()) return;

    int64_t axis = n.GetInt("axis", 0);
    int ndim = (int)data->shape.size();
    if (axis < 0) axis += ndim;

    int64_t dataNel = tensorNel(data);
    int64_t idxNel = tensorNel(indices);

    // GPU path for fp16 data with i32 indices, axis = last dim
    if (data->dtype == TensorDtype::Float16 &&
        indices->dtype == TensorDtype::Int32 &&
        updates->dtype == TensorDtype::Float16 &&
        axis == ndim - 1 && ex.gpu->supportsShaderF16) {
        ex.EnsureGpu(*data);
        ex.EnsureGpu(*indices);
        ex.EnsureGpu(*updates);

        *out[0] = ex.AllocTensor(data->shape, TensorDtype::Float16);

        // Pass 1: copy data → output
        {
            uint32_t params[8] = {(uint32_t)dataNel, (uint32_t)data->shape[axis],
                                   (uint32_t)idxNel, (uint32_t)indices->shape[axis], 0};
            auto paramBuf = ex.gpu->createBuffer("se_p1", 32);
            ex.gpu->writeBuffer(paramBuf, params, 20);

            auto& pl = ex.GetPipeline("scatter_elements_f16", WGSL_SCATTER_ELEMENTS_F16, 5);
            auto bg = ex.MakeBindGroup(pl, {
                {0, data->buffer}, {1, indices->buffer}, {2, updates->buffer},
                {3, out[0]->buffer}, {4, paramBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((dataNel + 255) / 256), 1, 1, "scatter_copy"});
        }
        // Pass 2: scatter updates at indexed positions
        {
            uint32_t params[8] = {(uint32_t)dataNel, (uint32_t)data->shape[axis],
                                   (uint32_t)idxNel, (uint32_t)indices->shape[axis], 1};
            auto paramBuf = ex.gpu->createBuffer("se_p2", 32);
            ex.gpu->writeBuffer(paramBuf, params, 20);

            auto& pl = ex.GetPipeline("scatter_elements_f16", WGSL_SCATTER_ELEMENTS_F16, 5);
            auto bg = ex.MakeBindGroup(pl, {
                {0, data->buffer}, {1, indices->buffer}, {2, updates->buffer},
                {3, out[0]->buffer}, {4, paramBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((idxNel + 255) / 256), 1, 1, "scatter_write"});
        }
        return;
    }

    // GPU path for f32 data with i32 indices, axis = last dim
    if (data->dtype == TensorDtype::Float32 &&
        indices->dtype == TensorDtype::Int32 &&
        updates->dtype == TensorDtype::Float32 &&
        axis == ndim - 1) {
        ex.EnsureGpu(*data);
        ex.EnsureGpu(*indices);
        ex.EnsureGpu(*updates);

        *out[0] = ex.AllocTensor(data->shape, TensorDtype::Float32);

        {
            uint32_t params[8] = {(uint32_t)dataNel, (uint32_t)data->shape[axis],
                                   (uint32_t)idxNel, (uint32_t)indices->shape[axis], 0};
            auto paramBuf = ex.gpu->createBuffer("se_p1", 32);
            ex.gpu->writeBuffer(paramBuf, params, 20);
            auto& pl = ex.GetPipeline("scatter_elements_f32", WGSL_SCATTER_ELEMENTS_F32, 5);
            auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, indices->buffer}, {2, updates->buffer},
                {3, out[0]->buffer}, {4, paramBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((dataNel + 255) / 256), 1, 1, "scatter_copy_f32"});
        }
        {
            uint32_t params[8] = {(uint32_t)dataNel, (uint32_t)data->shape[axis],
                                   (uint32_t)idxNel, (uint32_t)indices->shape[axis], 1};
            auto paramBuf = ex.gpu->createBuffer("se_p2", 32);
            ex.gpu->writeBuffer(paramBuf, params, 20);
            auto& pl = ex.GetPipeline("scatter_elements_f32", WGSL_SCATTER_ELEMENTS_F32, 5);
            auto bg = ex.MakeBindGroup(pl, {{0, data->buffer}, {1, indices->buffer}, {2, updates->buffer},
                {3, out[0]->buffer}, {4, paramBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((idxNel + 255) / 256), 1, 1, "scatter_write_f32"});
        }
        return;
    }

    // CPU fallback
    std::vector<float> dataVals, updateVals;
    std::vector<int64_t> idxVals;
    if (!loadTensorFloats(ex, data, n.inputs[0], dataVals)) return;
    if (!readTensorIntValues(ex, indices, n.inputs[1], idxVals)) return;
    if (!loadTensorFloats(ex, updates, n.inputs[2], updateVals)) return;

    std::vector<float> result = dataVals;
    std::vector<int64_t> dataStrides(ndim, 1);
    for (int i = ndim - 2; i >= 0; i--) dataStrides[i] = dataStrides[i+1] * data->shape[i+1];
    std::vector<int64_t> idxStrides(ndim, 1);
    for (int i = ndim - 2; i >= 0; i--) idxStrides[i] = idxStrides[i+1] * indices->shape[i+1];

    for (int64_t flat = 0; flat < idxNel; flat++) {
        int64_t remaining = flat;
        std::vector<int64_t> multiIdx(ndim);
        for (int d = 0; d < ndim; d++) {
            multiIdx[d] = remaining / idxStrides[d];
            remaining %= idxStrides[d];
        }
        int64_t idx = idxVals[flat];
        if (idx < 0) idx += data->shape[axis];
        multiIdx[axis] = idx;
        int64_t dataFlat = 0;
        for (int d = 0; d < ndim; d++) dataFlat += multiIdx[d] * dataStrides[d];
        if (dataFlat >= 0 && dataFlat < dataNel) {
            result[dataFlat] = updateVals[flat];
        }
    }

    *out[0] = ex.AllocCpuTensor(data->shape, TensorDtype::Float32, result.data(), dataNel * 4);
    ex.EnsureGpu(*out[0]);
}

// ─── QMoE (GPU) ──────────────────────────────────────────────────────────────
// Quantized Mixture of Experts with GPU-dispatched Q4 matmul per expert.
//
// Strategy:
//   1. Read router weights (32 values per token) to CPU to find active experts
//   2. For each active expert dispatch on GPU:
//      gate_up Q4 matmul → SwiGLU → down Q4 matmul → weighted accumulate
//   Only one small CPU readback (32 * fp16 = 64 bytes per token).

static void opQMoE(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* input = in[0];
    auto* routerWeights = in.size() > 1 ? in[1] : nullptr;
    auto* gateUpW = in.size() > 2 ? in[2] : nullptr;
    auto* gateUpS = in.size() > 3 ? in[3] : nullptr;
    auto* downW = in.size() > 5 ? in[5] : nullptr;
    auto* downS = in.size() > 6 ? in[6] : nullptr;

    if (!input || !routerWeights || !gateUpW || !gateUpS || !downW || !downS ||
        !input->IsValid() || !routerWeights->IsValid() ||
        !gateUpW->IsValid() || !gateUpS->IsValid() ||
        !downW->IsValid() || !downS->IsValid()) {
        fprintf(stderr, "QMoE: missing inputs\n");
        return;
    }

    int64_t blockSize = n.GetInt("block_size", 32);
    int64_t normRouting = n.GetInt("normalize_routing_weights", 1);
    int64_t k = n.GetInt("k", 4);

    int64_t hiddenSize = input->shape.back();
    int64_t numExperts = gateUpW->shape[0];
    int64_t N_gu = gateUpW->shape[1];
    int64_t moeIntermediate = N_gu / 2;
    int64_t N_dn = downW->shape[1];
    int64_t blocksPerCol_gu = hiddenSize / blockSize;
    int64_t blocksPerCol_dn = moeIntermediate / blockSize;

    // Ensure input is f32 on GPU
    if (!ensureTensorFloat32(ex, *input, n.inputs[0])) {
        fprintf(stderr, "QMoE: cannot convert input to f32\n");
        return;
    }

    // Ensure router weights are f32 on GPU (for the gate kernel)
    if (routerWeights->dtype == TensorDtype::Float16) {
        ensureTensorFloat32(ex, *routerWeights, n.inputs[1]);
    }
    ex.EnsureGpu(*routerWeights);
    ex.EnsureGpu(*gateUpW);
    ex.EnsureGpu(*gateUpS);
    ex.EnsureGpu(*downW);
    ex.EnsureGpu(*downS);

    // Allocate output (zero-filled)
    auto outShape = input->shape;
    outShape.back() = hiddenSize;
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);
    {
        std::vector<float> zeros(hiddenSize, 0.0f);
        ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), zeros.size() * sizeof(float));
    }

    // GPU expert selection: gate kernel produces indices + weights on GPU
    auto expertIdxBuf = ex.gpu->createBuffer("moe_expert_idx", (size_t)k * 4);
    auto expertWtBuf = ex.gpu->createBuffer("moe_expert_wt", (size_t)k * 4);

    {
        uint32_t gateParams[4] = {(uint32_t)numExperts, (uint32_t)k, (uint32_t)normRouting, 0};
        auto gpBuf = ex.gpu->createBuffer("moe_gate_p", 16);
        ex.gpu->writeBuffer(gpBuf, gateParams, 16);
        auto& pl = ex.GetPipeline("moe_gate", WGSL_MOE_GATE, 4);
        auto bg = ex.MakeBindGroup(pl, {
            {0, routerWeights->buffer}, {1, expertIdxBuf}, {2, expertWtBuf}, {3, gpBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg, 1, 1, 1, "moe_gate"});
    }

    // Scratch buffers (reused across expert slots)
    GpuTensor gateUpBuf = ex.AllocTensor({N_gu}, TensorDtype::Float32);
    GpuTensor intermediateBuf = ex.AllocTensor({moeIntermediate}, TensorDtype::Float32);
    GpuTensor downBuf = ex.AllocTensor({N_dn}, TensorDtype::Float32);

    // Dispatch k expert pipelines — each reads its expert index from GPU buffer
    for (uint32_t slot = 0; slot < (uint32_t)k; slot++) {
        // Gate-up Q4 matmul (indirect expert index)
        {
            uint32_t params[4] = {(uint32_t)N_gu, (uint32_t)hiddenSize,
                                   (uint32_t)blocksPerCol_gu, slot};
            auto pBuf = ex.gpu->createBuffer("moe_gu_p", 16);
            ex.gpu->writeBuffer(pBuf, params, 16);
            auto& pl = ex.GetPipeline("matmul_q4_indirect", WGSL_MATMUL_Q4_INDIRECT, 6);
            auto bg = ex.MakeBindGroup(pl, {
                {0, input->buffer}, {1, gateUpW->buffer}, {2, gateUpS->buffer},
                {3, gateUpBuf.buffer}, {4, pBuf}, {5, expertIdxBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((N_gu + 255) / 256), 1, 1, "moe_gateup"});
        }

        // SwiGLU (interleaved layout)
        {
            uint32_t params[4] = {(uint32_t)moeIntermediate, 0, 0, 0};
            auto pBuf = ex.gpu->createBuffer("moe_sg_p", 16);
            ex.gpu->writeBuffer(pBuf, params, 16);
            auto& pl = ex.GetPipeline("swiglu", WGSL_SWIGLU, 3);
            auto bg = ex.MakeBindGroup(pl, {
                {0, gateUpBuf.buffer}, {1, intermediateBuf.buffer}, {2, pBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((moeIntermediate + 255) / 256), 1, 1, "moe_swiglu"});
        }

        // Down Q4 matmul (indirect expert index)
        {
            uint32_t params[4] = {(uint32_t)N_dn, (uint32_t)moeIntermediate,
                                   (uint32_t)blocksPerCol_dn, slot};
            auto pBuf = ex.gpu->createBuffer("moe_dn_p", 16);
            ex.gpu->writeBuffer(pBuf, params, 16);
            auto& pl = ex.GetPipeline("matmul_q4_indirect", WGSL_MATMUL_Q4_INDIRECT, 6);
            auto bg = ex.MakeBindGroup(pl, {
                {0, intermediateBuf.buffer}, {1, downW->buffer}, {2, downS->buffer},
                {3, downBuf.buffer}, {4, pBuf}, {5, expertIdxBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((N_dn + 255) / 256), 1, 1, "moe_down"});
        }

        // Weighted accumulate (indirect weight from GPU buffer)
        {
            uint32_t params[4] = {(uint32_t)N_dn, slot, 0, 0};
            auto pBuf = ex.gpu->createBuffer("moe_wa_p", 16);
            ex.gpu->writeBuffer(pBuf, params, 16);
            auto& pl = ex.GetPipeline("weighted_add_indirect", WGSL_WEIGHTED_ADD_INDIRECT, 4);
            auto bg = ex.MakeBindGroup(pl, {
                {0, downBuf.buffer}, {1, out[0]->buffer}, {2, pBuf}, {3, expertWtBuf}});
            ex.pendingDispatches_.push_back({pl.pipeline, bg,
                (uint32_t)((N_dn + 255) / 256), 1, 1, "moe_accum"});
        }
    }
}

// ─── Registration ────────────────────────────────────────────────────────────

REGISTER_OP(TopK, opTopK)
REGISTER_OP(GatherElements, opGatherElements)
REGISTER_OP(ScatterElements, opScatterElements)
REGISTER_OP(QMoE, opQMoE)
