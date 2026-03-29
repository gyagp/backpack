/**
 * ops/norm.cpp — Normalization ops using embedded WGSL kernels.
 * SimplifiedLayerNormalization (RMSNorm), SkipSimplifiedLayerNormalization,
 * LayerNormalization, InstanceNormalization.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include "../wgsl_template.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static TensorDtype computeOutDtype(TensorDtype dtype) {
    if (dtype == TensorDtype::Float16) return TensorDtype::Float16;
    if (dtype == TensorDtype::Float32) return TensorDtype::Float32;
    return dtype;
}

static bool isVaeDecoderNode(const std::string& nodeName) {
    return nodeName.rfind("/decoder/", 0) == 0;
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

static bool readTensorFloats(GraphExecutor& ex, const GpuTensor* t,
    const std::string& name, std::vector<float>& out) {
    out.clear();
    if (!t) return false;
    int64_t nel = tensorNel(t);
    if (nel < 0) return false;
    out.resize((size_t)nel);

    auto convertFp16 = [&](const uint8_t* src) {
        for (int64_t i = 0; i < nel; i++) {
            uint16_t h;
            memcpy(&h, src + i * 2, 2);
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) f = (sign << 31) | (mant << 13);
            else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
            else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&out[(size_t)i], &f, sizeof(float));
        }
    };

    auto load = [&](const uint8_t* src, size_t size) -> bool {
        if (!src) return false;
        if (t->dtype == TensorDtype::Float32) {
            if (size < (size_t)nel * sizeof(float)) return false;
            memcpy(out.data(), src, (size_t)nel * sizeof(float));
            return true;
        }
        if (t->dtype == TensorDtype::Float16) {
            if (size < (size_t)nel * sizeof(uint16_t)) return false;
            convertFp16(src);
            return true;
        }
        return false;
    };

    if (!t->cpuData.empty() && load(t->cpuData.data(), t->cpuData.size())) return true;
    if (auto* init = ex.GetInitData(name); init && init->data && load(init->data, init->size)) return true;
    if (t->buffer.handle) {
        ex.FlushPendingWork();
        size_t bytes = (size_t)nel * t->DtypeSize();
        auto raw = ex.gpu->readBuffer(t->buffer, bytes);
        if (load(raw.data(), raw.size())) return true;
    }

    out.clear();
    return false;
}

static bool ensureTensorFloat32(GraphExecutor& ex, GpuTensor& tensor, const std::string& name,
                                bool preferRawInitializer = false) {
    if (auto* init = ex.GetInitData(name); init && init->data && init->dtype == TensorDtype::Float16 &&
        (preferRawInitializer || tensor.cpuData.empty())) {
        int64_t nel = tensorNel(&tensor);
        if (nel > 0 && init->size >= (size_t)nel * sizeof(uint16_t)) {
            std::vector<float> values((size_t)nel, 0.0f);
            auto* src = reinterpret_cast<const uint16_t*>(init->data);
            for (int64_t i = 0; i < nel; i++) values[(size_t)i] = fp16ToFloat(src[(size_t)i]);
            GpuTensor rebuilt;
            rebuilt.shape = tensor.shape;
            rebuilt.dtype = TensorDtype::Float32;
            rebuilt.cpuData.resize(values.size() * sizeof(float));
            memcpy(rebuilt.cpuData.data(), values.data(), rebuilt.cpuData.size());
            rebuilt.buffer = ex.gpu->createBuffer(name.empty() ? "norm_f32_rebuilt" : name,
                                                  rebuilt.cpuData.size());
            ex.gpu->writeBuffer(rebuilt.buffer, rebuilt.cpuData.data(), rebuilt.cpuData.size());
            rebuilt.isCpuOnly = false;
            tensor = std::move(rebuilt);
            return tensor.buffer.handle != nullptr;
        }
    }
    if (tensor.dtype == TensorDtype::Float32) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }
    if (tensor.dtype != TensorDtype::Float16) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }

    std::vector<float> values;
    if (!readTensorFloats(ex, &tensor, name, values)) return false;
    GpuTensor rebuilt;
    rebuilt.shape = tensor.shape;
    rebuilt.dtype = TensorDtype::Float32;
    rebuilt.cpuData.resize(values.size() * sizeof(float));
    memcpy(rebuilt.cpuData.data(), values.data(), rebuilt.cpuData.size());
    rebuilt.buffer = ex.gpu->createBuffer(name.empty() ? "norm_f32_cast" : name,
                                          rebuilt.cpuData.size());
    ex.gpu->writeBuffer(rebuilt.buffer, rebuilt.cpuData.data(), rebuilt.cpuData.size());
    rebuilt.isCpuOnly = false;
    tensor = std::move(rebuilt);
    return tensor.buffer.handle != nullptr;
}

static bool canUseFp16NormWeights(const GraphExecutor& ex, const GpuTensor* x,
                                  const GpuTensor* w, const GpuTensor* b = nullptr) {
    if (!ex.gpu->supportsShaderF16 || !x || !w) return false;
    if (x->dtype != TensorDtype::Float32 || w->dtype != TensorDtype::Float16) return false;
    if (b && b->dtype != TensorDtype::Float16) return false;
    return true;
}

// ─── RMSNorm (SimplifiedLayerNormalization) ──────────────────────────────────
// Uses WGSL_RMSNORM_T template for both f32 and f16 activation paths.
// Falls back to WGSL_RMSNORM_SIMPLE_F16W (kernels/norm/) for mixed f32-data/f16-weight.

static void opSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* W = in.size() > 1 ? in[1] : nullptr;
    if (!X || !X->IsValid() || X->shape.empty()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    if (!W || !W->IsValid()) return;

    *out[0] = ex.AllocTensor(X->shape, X->dtype);  // Match input dtype
    if (out.size() > 1 && out[1]) *out[1] = ex.AllocTensor({nRows}, TensorDtype::Float32);

    // Allocate Rstd output
    GPUBuffer rstdBuf;
    if (out.size() > 1 && out[1] && out[1]->IsValid()) {
        rstdBuf = out[1]->buffer;
    } else {
        rstdBuf = ex.gpu->createBuffer("rstd", std::max((int64_t)4, nRows * 4));
    }

    struct { int32_t stride; int32_t N; float eps; } p;
    p.stride = (int32_t)(nRows * hiddenDim);
    p.N = (int32_t)hiddenDim;
    p.eps = eps;
    auto paramBuf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(paramBuf, &p, 12);

    // Templated path: both X and W are fp16 → use fp16 template
    if (X->dtype == TensorDtype::Float16 && W->dtype == TensorDtype::Float16) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*W);
        auto& pl = ex.GetPipelineT("rmsnorm_t_f16", 5, []() {
            return instantiateTemplate(WGSL_RMSNORM_T, TensorDtype::Float16);
        });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, out[0]->buffer}, {2, W->buffer},
            {3, rstdBuf}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((nRows + 255) / 256), 1, 1, "rmsnorm_f16");
        return;
    }

    // Legacy paths for f32 or mixed dtype
    if (!isVaeDecoderNode(n.name) && canUseFp16NormWeights(ex, X, W)) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*W);
        auto& pl = ex.GetPipelineT("rmsnorm_simple_f16w", 5, []() { return std::string(WGSL_RMSNORM_SIMPLE_F16W); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, out[0]->buffer}, {2, W->buffer},
            {3, rstdBuf}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)nRows, 1, 1, "rmsnorm_f16w");
        return;
    }

    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
        !ensureTensorFloat32(ex, *W, n.inputs.size() > 1 ? n.inputs[1] : std::string())) {
        return;
    }

    auto& pl = ex.GetPipelineT("rmsnorm_simple", 5, []() { return instantiateTemplate(WGSL_RMSNORM_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, out[0]->buffer}, {2, W->buffer},
        {3, rstdBuf}, {4, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "rmsnorm");
}

// ─── SkipSimplifiedLayerNorm (residual add + RMSNorm) ────────────────────────
// Uses WGSL_SKIP_RMSNORM_T template for typed path.
// Uses WGSL_SKIP_RMSNORM_F16W (kernels/norm/) for f32-data + f16-weight.

static void opSkipSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* Skip = in.size() > 1 ? in[1] : nullptr;
    auto* W = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid() || X->shape.empty()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    *out[0] = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));

    if (!Skip || !W || !Skip->IsValid() || !W->IsValid()) return;

    GPUBuffer skipOutBuf;
    if (out.size() > 3 && out[3] && out[3]->IsValid()) {
        skipOutBuf = out[3]->buffer;
    } else {
        auto skipOutTensor = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));
        skipOutBuf = skipOutTensor.buffer;
        if (out.size() > 3 && out[3]) *out[3] = skipOutTensor;
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)hiddenDim, (uint32_t)nRows, eps_u32, 0};
    auto paramBuf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    if (!isVaeDecoderNode(n.name) && canUseFp16NormWeights(ex, X, W) && Skip->dtype == TensorDtype::Float32) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*Skip);
        ex.EnsureGpu(*W);
        auto& pl = ex.GetPipelineT("skip_rmsnorm_f16w", 6, []() { return std::string(WGSL_SKIP_RMSNORM_F16W); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, Skip->buffer}, {2, W->buffer},
            {3, out[0]->buffer}, {4, skipOutBuf}, {5, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((nRows + 255) / 256), 1, 1, "skip_rmsnorm_f16w");
        return;
    }

    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
        !ensureTensorFloat32(ex, *Skip, n.inputs.size() > 1 ? n.inputs[1] : std::string()) ||
        !ensureTensorFloat32(ex, *W, n.inputs.size() > 2 ? n.inputs[2] : std::string())) {
        return;
    }

    auto& pl = ex.GetPipelineT("skip_rmsnorm", 6, []() { return instantiateTemplate(WGSL_SKIP_RMSNORM_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, Skip->buffer}, {2, W->buffer},
        {3, out[0]->buffer}, {4, skipOutBuf}, {5, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "skip_rmsnorm");
}

// ─── LayerNormalization ──────────────────────────────────────────────────────

static void opLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* W = in.size() > 1 ? in[1] : nullptr;
    auto* B = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid() || X->shape.empty()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;
    *out[0] = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));
    if (!W || !W->IsValid()) return;

    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
        !ensureTensorFloat32(ex, *W, n.inputs.size() > 1 ? n.inputs[1] : std::string())) {
        return;
    }
    if (B && B->IsValid() &&
        !ensureTensorFloat32(ex, *B, n.inputs.size() > 2 ? n.inputs[2] : std::string())) {
        return;
    }

    GPUBuffer biasBuf;
    if (B && B->IsValid()) { biasBuf = B->buffer; }
    else {
        std::vector<float> zeros((size_t)hiddenDim, 0.0f);
        biasBuf = ex.gpu->createBuffer("ln_b0", hiddenDim * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), hiddenDim * 4);
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)hiddenDim, (uint32_t)nRows, eps_u32, 0};
    auto paramBuf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    if (B && !isVaeDecoderNode(n.name) && canUseFp16NormWeights(ex, X, W, B)) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*W);
        ex.EnsureGpu(*B);
        auto& pl = ex.GetPipelineT("layer_norm_f16wb", 5, []() { return std::string(WGSL_LAYER_NORM_F16WB); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((nRows + 255) / 256), 1, 1, "layernorm_f16wb");
        return;
    }

    auto& pl = ex.GetPipelineT("layer_norm", 5, []() { return instantiateTemplate(WGSL_LAYER_NORM_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "layernorm");
}

// ─── InstanceNormalization ───────────────────────────────────────────────────

static void opInstanceNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* Scale = in.size() > 1 ? in[1] : nullptr;
    auto* Bias = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t N_batch = (X->shape.size() >= 4) ? X->shape[0] : 1;
    int64_t C = (X->shape.size() >= 3) ? X->shape[X->shape.size()-3] : 1;
    int64_t HW = 1;
    for (size_t i = X->shape.size()-2; i < X->shape.size(); i++) HW *= X->shape[i];
    if (!Scale || !Bias || !Scale->IsValid() || !Bias->IsValid()) return;

    *out[0] = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));

    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
        !ensureTensorFloat32(ex, *Scale, n.inputs.size() > 1 ? n.inputs[1] : std::string(), true) ||
        !ensureTensorFloat32(ex, *Bias, n.inputs.size() > 2 ? n.inputs[2] : std::string(), true)) {
        return;
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)C, (uint32_t)HW, (uint32_t)N_batch, eps_u32};
    auto paramBuf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    if (!isVaeDecoderNode(n.name) && canUseFp16NormWeights(ex, X, Scale, Bias)) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*Scale);
        ex.EnsureGpu(*Bias);
        auto& pl = ex.GetPipelineT("instance_norm_f16wb", 5, []() { return std::string(WGSL_INSTANCE_NORM_F16WB); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, Scale->buffer}, {2, Bias->buffer},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)(N_batch * C), 1, 1, "instancenorm_f16wb");
        return;
    }

    auto& pl = ex.GetPipelineT("instance_norm", 5, []() { return instantiateTemplate(WGSL_INSTANCE_NORM_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, Scale->buffer}, {2, Bias->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)(N_batch * C), 1, 1, "instancenorm");
}

// ─── GroupNormalization ───────────────────────────────────────────────────────

static void opGroupNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* Scale = in.size() > 1 ? in[1] : nullptr;
    auto* Bias = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t numGroups = n.GetInt("num_groups", 32);

    // X shape: [N, C, H, W] or [N, C, *spatial]
    int64_t N_batch = X->shape.size() >= 1 ? X->shape[0] : 1;
    int64_t C = X->shape.size() >= 2 ? X->shape[1] : 1;
    int64_t HW = 1;
    for (size_t i = 2; i < X->shape.size(); i++) HW *= X->shape[i];

    *out[0] = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));
    if (!Scale || !Bias || !Scale->IsValid() || !Bias->IsValid()) return;

    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
        !ensureTensorFloat32(ex, *Scale, n.inputs.size() > 1 ? n.inputs[1] : std::string()) ||
        !ensureTensorFloat32(ex, *Bias, n.inputs.size() > 2 ? n.inputs[2] : std::string())) {
        return;
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)C, (uint32_t)HW, (uint32_t)numGroups, eps_u32};
    auto paramBuf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    if (!isVaeDecoderNode(n.name) && canUseFp16NormWeights(ex, X, Scale, Bias)) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*Scale);
        ex.EnsureGpu(*Bias);
        auto& pl = ex.GetPipelineT("group_norm_f16wb", 5, []() { return std::string(WGSL_GROUP_NORM_F16WB); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, Scale->buffer}, {2, Bias->buffer},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)(N_batch * numGroups), 1, 1, "groupnorm_f16wb");
        return;
    }

    auto& pl = ex.GetPipelineT("group_norm", 5, []() { return instantiateTemplate(WGSL_GROUP_NORM_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, Scale->buffer}, {2, Bias->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)(N_batch * numGroups), 1, 1, "groupnorm");
}

REGISTER_OP(SimplifiedLayerNormalization, opSimplifiedLayerNorm)
REGISTER_OP(SkipSimplifiedLayerNormalization, opSkipSimplifiedLayerNorm)
REGISTER_OP(LayerNormalization, opLayerNorm)
REGISTER_OP(InstanceNormalization, opInstanceNorm)
REGISTER_OP(GroupNorm, opGroupNorm)
