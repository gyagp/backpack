/**
 * ops/conv.cpp — Convolution and resize ops using embedded WGSL kernels.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static constexpr bool kDebugVaeConv = false;

static bool isVaeDecoderNode(const std::string& nodeName) {
    return nodeName.rfind("/decoder/", 0) == 0;
}

static TensorDtype computeOutDtype(TensorDtype dtype) {
    return (dtype == TensorDtype::Float16 || dtype == TensorDtype::Float32)
             ? TensorDtype::Float32
             : dtype;
}

static bool readTensorFloats(GraphExecutor& ex, const GpuTensor& tensor,
                             const std::string& name, std::vector<float>& out) {
    out.clear();
    size_t count = (size_t)tensor.ElementCount();
    out.resize(count);

    auto convertFp16 = [&](const uint8_t* src, size_t bytes) -> bool {
        if (bytes < count * sizeof(uint16_t)) return false;
        auto* srcFp16 = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < count; i++) {
            uint16_t h = srcFp16[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) f = (sign << 31) | (mant << 13);
            else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
            else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&out[i], &f, sizeof(float));
        }
        return true;
    };

    if (tensor.dtype == TensorDtype::Float32) {
        size_t bytes = count * sizeof(float);
        if (!tensor.cpuData.empty() && tensor.cpuData.size() >= bytes) {
            memcpy(out.data(), tensor.cpuData.data(), bytes);
            return true;
        }
        if (auto* init = ex.GetInitData(name); init && init->data && init->size >= bytes) {
            memcpy(out.data(), init->data, bytes);
            return true;
        }
        if (tensor.buffer.handle) {
            ex.FlushPendingWork();
            auto raw = ex.gpu->readBuffer(tensor.buffer, bytes);
            if (raw.size() >= bytes) {
                memcpy(out.data(), raw.data(), bytes);
                return true;
            }
        }
        return false;
    }

    if (tensor.dtype == TensorDtype::Float16) {
        size_t bytes = count * sizeof(uint16_t);
        if (!tensor.cpuData.empty() && convertFp16(tensor.cpuData.data(), tensor.cpuData.size())) return true;
        if (auto* init = ex.GetInitData(name); init && init->data && convertFp16(init->data, init->size)) return true;
        if (tensor.buffer.handle) {
            ex.FlushPendingWork();
            auto raw = ex.gpu->readBuffer(tensor.buffer, bytes);
            if (convertFp16(raw.data(), raw.size())) return true;
        }
    }

    out.clear();
    return false;
}

static bool ensureTensorFloat32(GraphExecutor& ex, GpuTensor& tensor, const std::string& name,
                                bool preferRawInitializer = false) {
    if (auto* init = ex.GetInitData(name); init && init->data && init->dtype == TensorDtype::Float16 &&
        (preferRawInitializer || tensor.cpuData.empty())) {
        size_t count = (size_t)tensor.ElementCount();
        if (init->size >= count * sizeof(uint16_t)) {
            std::vector<float> values(count, 0.0f);
            auto* src = reinterpret_cast<const uint16_t*>(init->data);
            for (size_t i = 0; i < count; i++) {
                uint16_t h = src[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) f = (sign << 31) | (mant << 13);
                else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
                else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&values[i], &f, sizeof(float));
            }
            GpuTensor rebuilt;
            rebuilt.shape = tensor.shape;
            rebuilt.dtype = TensorDtype::Float32;
            rebuilt.cpuData.resize(values.size() * sizeof(float));
            memcpy(rebuilt.cpuData.data(), values.data(), rebuilt.cpuData.size());
            rebuilt.buffer = ex.gpu->createBuffer(name.empty() ? "conv_f32_rebuilt" : name,
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
    if (!readTensorFloats(ex, tensor, name, values)) return false;
    GpuTensor rebuilt;
    rebuilt.shape = tensor.shape;
    rebuilt.dtype = TensorDtype::Float32;
    rebuilt.cpuData.resize(values.size() * sizeof(float));
    memcpy(rebuilt.cpuData.data(), values.data(), rebuilt.cpuData.size());
    rebuilt.buffer = ex.gpu->createBuffer(name.empty() ? "conv_f32_cast" : name,
                                          rebuilt.cpuData.size());
    ex.gpu->writeBuffer(rebuilt.buffer, rebuilt.cpuData.data(), rebuilt.cpuData.size());
    rebuilt.isCpuOnly = false;
    tensor = std::move(rebuilt);
    return tensor.buffer.handle != nullptr;
}

static void debugTensorStats(GraphExecutor& ex, const char* label, const GpuTensor& tensor) {
    if (!tensor.buffer.handle || tensor.dtype != TensorDtype::Float32) return;
    size_t count = (size_t)tensor.ElementCount();
    if (count == 0) return;
    ex.FlushPendingWork();
    auto raw = ex.gpu->readBuffer(tensor.buffer, count * sizeof(float));
    if (raw.size() < count * sizeof(float)) return;
    const float* values = reinterpret_cast<const float*>(raw.data());
    float minV = 1e30f, maxV = -1e30f;
    double sum = 0.0;
    int nanCount = 0;
    for (size_t i = 0; i < count; i++) {
        float v = values[i];
        if (!std::isfinite(v)) {
            nanCount++;
            continue;
        }
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sum += v;
    }
    int valid = (int)count - nanCount;
    fprintf(stderr, "    [dbg] %s: min=%.4f max=%.4f avg=%.4f nan=%d/%zu\n",
            label, minV, maxV, valid > 0 ? sum / valid : 0.0, nanCount, count);
    fflush(stderr);
}

static void debugReadableTensorStats(GraphExecutor& ex, const char* label,
                                     const GpuTensor& tensor, const std::string& name) {
    std::vector<float> values;
    if (!readTensorFloats(ex, tensor, name, values) || values.empty()) return;
    float minV = 1e30f, maxV = -1e30f;
    double sum = 0.0;
    int nanCount = 0;
    for (float v : values) {
        if (!std::isfinite(v)) {
            nanCount++;
            continue;
        }
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sum += v;
    }
    int valid = (int)values.size() - nanCount;
    fprintf(stderr, "    [dbg] %s: min=%.4f max=%.4f avg=%.4f nan=%d/%zu\n",
            label, minV, maxV, valid > 0 ? sum / valid : 0.0, nanCount, values.size());
    fflush(stderr);
}

static void debugInitFp16Stats(GraphExecutor& ex, const char* label, const std::string& name) {
    auto* init = ex.GetInitData(name);
    if (!init || !init->data || init->dtype != TensorDtype::Float16) return;
    size_t count = init->size / sizeof(uint16_t);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(init->data);
    float minV = 1e30f, maxV = -1e30f;
    double sum = 0.0;
    int nanCount = 0;
    for (size_t i = 0; i < count; i++) {
        uint16_t h = src[i];
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) f = (sign << 31) | (mant << 13);
        else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
        else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        float v;
        memcpy(&v, &f, sizeof(float));
        if (!std::isfinite(v)) {
            nanCount++;
            continue;
        }
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sum += v;
    }
    int valid = (int)count - nanCount;
    fprintf(stderr, "    [dbg] %s: min=%.4f max=%.4f avg=%.4f nan=%d/%zu\n",
            label, minV, maxV, valid > 0 ? sum / valid : 0.0, nanCount, count);
    fflush(stderr);
}

// ─── Conv1D / Conv2D ─────────────────────────────────────────────────────────

static void opConv(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; auto* W = in[1];
    auto* B = in.size() > 2 ? in[2] : nullptr;
    if (!X || !W || !X->IsValid() || !W->IsValid()) return;

    // ─── Conv1D path: 3D input [batch, channels, length] ─────────────────
    if (X->shape.size() == 3) {
        bool useFp16Path = ex.gpu->supportsShaderF16 &&
                           X->dtype == TensorDtype::Float16 && W->dtype == TensorDtype::Float16 &&
                           (!B || !B->IsValid() || B->dtype == TensorDtype::Float16);
        if (useFp16Path) {
            ex.EnsureGpu(*X);
            ex.EnsureGpu(*W);
            if (B && B->IsValid()) ex.EnsureGpu(*B);
        } else if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
                   !ensureTensorFloat32(ex, *W, n.inputs.size() > 1 ? n.inputs[1] : std::string(), true) ||
                   (B && B->IsValid() && !ensureTensorFloat32(ex, *B, n.inputs.size() > 2 ? n.inputs[2] : std::string(), true))) {
            return;
        }

        // Reshape 3D → 4D: [batch, C, L] → [batch, C, L, 1]
        // Weight: [C_out, C_in/group, K] → [C_out, C_in/group, K, 1]
        auto origXShape = X->shape;
        auto origWShape = W->shape;
        X->shape = {origXShape[0], origXShape[1], origXShape[2], 1};
        W->shape = {origWShape[0], origWShape[1], origWShape[2], 1};

        int64_t batch = origXShape[0], C_in = origXShape[1], L_in = origXShape[2];
        int64_t C_out = origWShape[0], K = origWShape[2];
        int64_t group = n.GetInt("group", 1);

        std::vector<int64_t> pads_1d = {0, 0};
        std::vector<int64_t> strides_1d = {1};
        std::vector<int64_t> dilations_1d = {1};
        if (n.attrIntLists.count("pads")) pads_1d = n.attrIntLists.at("pads");
        if (n.attrIntLists.count("strides")) strides_1d = n.attrIntLists.at("strides");
        if (n.attrIntLists.count("dilations")) dilations_1d = n.attrIntLists.at("dilations");
        int64_t pad = pads_1d.size() >= 1 ? pads_1d[0] : 0;
        int64_t stride = strides_1d.size() >= 1 ? strides_1d[0] : 1;
        int64_t dilation = dilations_1d.size() >= 1 ? dilations_1d[0] : 1;
        int64_t K_eff = (K - 1) * dilation + 1;
        int64_t L_out = (L_in + 2*pad - K_eff) / stride + 1;

        *out[0] = ex.AllocTensor({batch, C_out, L_out, 1}, computeOutDtype(X->dtype));

        GPUBuffer biasBuf;
        if (B && B->IsValid()) { biasBuf = B->buffer; }
        else {
            if (useFp16Path) {
                std::vector<uint16_t> zeros((size_t)C_out, 0u);
                biasBuf = ex.gpu->createBuffer("conv1d_b0_f16", C_out * 2);
                ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 2);
            } else {
                std::vector<float> zeros((size_t)C_out, 0.0f);
                biasBuf = ex.gpu->createBuffer("conv1d_b0", C_out * 4);
                ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 4);
            }
        }

        // Use Conv2D kernel with H=L, W=1
        uint32_t params[16] = {
            (uint32_t)batch, (uint32_t)C_in, (uint32_t)L_in, 1,   // H_in=L_in, W_in=1
            (uint32_t)C_out, (uint32_t)K, 1,                       // KH=K, KW=1
            (uint32_t)pad, 0, (uint32_t)stride, 1,                 // pad_w=0, stride_w=1
            (uint32_t)L_out, 1, (uint32_t)group,                   // H_out=L_out, W_out=1
            (uint32_t)dilation, 1                                    // dil_h=dilation, dil_w=1
        };
        auto paramBuf = ex.gpu->createBuffer("conv1d_p", 64);
        ex.gpu->writeBuffer(paramBuf, params, 64);

        auto& pl = useFp16Path
            ? ex.GetPipeline("conv2d_f16", WGSL_CONV2D_F16, 5)
            : ex.GetPipeline("conv2d", WGSL_CONV2D, 5);
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
            {3, out[0]->buffer}, {4, paramBuf}});
        int64_t total = batch * C_out * L_out;
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)((total + 255) / 256), 1, 1, "conv1d"});

        // Reshape output back to 3D
        out[0]->shape = {batch, C_out, L_out};
        // Restore original shapes
        X->shape = origXShape;
        W->shape = origWShape;
        return;
    }

    // ─── Conv2D path: 4D input [batch, channels, H, W] ──────────────────
    if (X->shape.size() < 4) return;

    bool useFp16Path = ex.gpu->supportsShaderF16 && !isVaeDecoderNode(n.name) &&
                       X->dtype == TensorDtype::Float16 && W->dtype == TensorDtype::Float16 &&
                       (!B || !B->IsValid() || B->dtype == TensorDtype::Float16);
    if (useFp16Path) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*W);
        if (B && B->IsValid()) ex.EnsureGpu(*B);
    } else if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
               !ensureTensorFloat32(ex, *W, n.inputs.size() > 1 ? n.inputs[1] : std::string(), true) ||
               (B && B->IsValid() && !ensureTensorFloat32(ex, *B, n.inputs.size() > 2 ? n.inputs[2] : std::string(), true))) {
        return;
    }

    bool debugVaeConvIn = kDebugVaeConv && (n.name == "/decoder/conv_in/Conv");
    bool debugVaeConvOut = kDebugVaeConv && (n.name == "/decoder/conv_out/Conv");
    if (debugVaeConvIn || debugVaeConvOut) {
        debugTensorStats(ex, debugVaeConvIn ? "vae conv input" : "vae conv_out input", *X);
        if (debugVaeConvOut) debugInitFp16Stats(ex, "vae conv_out init weight", n.inputs.size() > 1 ? n.inputs[1] : std::string());
        if (debugVaeConvOut) debugReadableTensorStats(ex, "vae conv_out weight readable", *W,
                                                     n.inputs.size() > 1 ? n.inputs[1] : std::string());
        debugTensorStats(ex, debugVaeConvIn ? "vae conv weight" : "vae conv_out weight", *W);
    }

    int64_t batch = X->shape[0], C_in = X->shape[1];
    int64_t H_in = X->shape[2], W_in = X->shape[3];
    int64_t C_out = W->shape[0], KH = W->shape[2], KW = W->shape[3];
    int64_t group = n.GetInt("group", 1);

    std::vector<int64_t> pads = {0,0,0,0}, strides = {1,1}, dilations = {1,1};
    if (n.attrIntLists.count("pads")) pads = n.attrIntLists.at("pads");
    if (n.attrIntLists.count("strides")) strides = n.attrIntLists.at("strides");
    if (n.attrIntLists.count("dilations")) dilations = n.attrIntLists.at("dilations");
    int64_t pad_h = pads.size() >= 1 ? pads[0] : 0;
    int64_t pad_w = pads.size() >= 2 ? pads[1] : 0;
    int64_t stride_h = strides.size() >= 1 ? strides[0] : 1;
    int64_t stride_w = strides.size() >= 2 ? strides[1] : 1;
    int64_t dil_h = dilations.size() >= 1 ? dilations[0] : 1;
    int64_t dil_w = dilations.size() >= 2 ? dilations[1] : 1;
    int64_t KH_eff = (KH - 1) * dil_h + 1;  // effective kernel size with dilation
    int64_t KW_eff = (KW - 1) * dil_w + 1;
    int64_t H_out = (H_in + 2*pad_h - KH_eff) / stride_h + 1;
    int64_t W_out = (W_in + 2*pad_w - KW_eff) / stride_w + 1;

    *out[0] = ex.AllocTensor({batch, C_out, H_out, W_out}, computeOutDtype(X->dtype));

    GPUBuffer biasBuf;
    if (B && B->IsValid()) { biasBuf = B->buffer; }
    else {
        if (useFp16Path) {
            std::vector<uint16_t> zeros((size_t)C_out, 0u);
            biasBuf = ex.gpu->createBuffer("conv_b0_f16", C_out * 2);
            ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 2);
        } else {
            std::vector<float> zeros((size_t)C_out, 0.0f);
            biasBuf = ex.gpu->createBuffer("conv_b0", C_out * 4);
            ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 4);
        }
    }

    uint32_t params[16] = {
        (uint32_t)batch, (uint32_t)C_in, (uint32_t)H_in, (uint32_t)W_in,
        (uint32_t)C_out, (uint32_t)KH, (uint32_t)KW,
        (uint32_t)pad_h, (uint32_t)pad_w, (uint32_t)stride_h, (uint32_t)stride_w,
        (uint32_t)H_out, (uint32_t)W_out, (uint32_t)group,
        (uint32_t)dil_h, (uint32_t)dil_w
    };
    auto paramBuf = ex.gpu->createBuffer("conv_p", 64);
    ex.gpu->writeBuffer(paramBuf, params, 64);

    auto& pl = useFp16Path
        ? ex.GetPipeline("conv2d_f16", WGSL_CONV2D_F16, 5)
        : ex.GetPipeline("conv2d", WGSL_CONV2D, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    int64_t total = batch * C_out * H_out * W_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "conv2d"});
    if (debugVaeConvIn || debugVaeConvOut) {
        debugTensorStats(ex, debugVaeConvIn ? "vae conv output" : "vae conv_out output", *out[0]);
    }
}

// ─── ConvTranspose2D ─────────────────────────────────────────────────────────

static void opConvTranspose(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; auto* W = in[1];
    auto* B = in.size() > 2 ? in[2] : nullptr;
    if (!X || !W || !X->IsValid() || !W->IsValid() || X->shape.size() < 4) return;

    bool useFp16Path = ex.gpu->supportsShaderF16 && X->dtype == TensorDtype::Float16 && W->dtype == TensorDtype::Float16 &&
                       (!B || !B->IsValid() || B->dtype == TensorDtype::Float16);
    if (useFp16Path) {
        ex.EnsureGpu(*X);
        ex.EnsureGpu(*W);
        if (B && B->IsValid()) ex.EnsureGpu(*B);
    } else if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0]) ||
               !ensureTensorFloat32(ex, *W, n.inputs.size() > 1 ? n.inputs[1] : std::string()) ||
               (B && B->IsValid() && !ensureTensorFloat32(ex, *B, n.inputs.size() > 2 ? n.inputs[2] : std::string()))) {
        return;
    }

    int64_t batch = X->shape[0], C_in = X->shape[1];
    int64_t H_in = X->shape[2], W_in = X->shape[3];
    // ConvTranspose weight: [C_in, C_out/group, KH, KW]
    int64_t C_out_per_group = W->shape[1];
    int64_t KH = W->shape[2], KW = W->shape[3];
    int64_t group = n.GetInt("group", 1);
    int64_t C_out = C_out_per_group * group;

    std::vector<int64_t> pads = {0,0,0,0}, strides = {1,1}, output_padding = {0,0};
    if (n.attrIntLists.count("pads")) pads = n.attrIntLists.at("pads");
    if (n.attrIntLists.count("strides")) strides = n.attrIntLists.at("strides");
    if (n.attrIntLists.count("output_padding")) output_padding = n.attrIntLists.at("output_padding");
    int64_t pad_h = pads.size() >= 1 ? pads[0] : 0;
    int64_t pad_w = pads.size() >= 2 ? pads[1] : 0;
    int64_t stride_h = strides.size() >= 1 ? strides[0] : 1;
    int64_t stride_w = strides.size() >= 2 ? strides[1] : 1;
    int64_t out_pad_h = output_padding.size() >= 1 ? output_padding[0] : 0;
    int64_t out_pad_w = output_padding.size() >= 2 ? output_padding[1] : 0;

    // Output size for ConvTranspose
    int64_t H_out = (H_in - 1) * stride_h - 2 * pad_h + KH + out_pad_h;
    int64_t W_out = (W_in - 1) * stride_w - 2 * pad_w + KW + out_pad_w;

    // Override if output_shape is specified
    if (n.attrIntLists.count("output_shape")) {
        auto& os = n.attrIntLists.at("output_shape");
        if (os.size() >= 2) { H_out = os[os.size()-2]; W_out = os[os.size()-1]; }
    }

    *out[0] = ex.AllocTensor({batch, C_out, H_out, W_out}, computeOutDtype(X->dtype));

    GPUBuffer biasBuf;
    if (B && B->IsValid()) { biasBuf = B->buffer; }
    else {
        if (useFp16Path) {
            std::vector<uint16_t> zeros((size_t)C_out, 0u);
            biasBuf = ex.gpu->createBuffer("ct_b0_f16", C_out * 2);
            ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 2);
        } else {
            std::vector<float> zeros((size_t)C_out, 0.0f);
            biasBuf = ex.gpu->createBuffer("ct_b0", C_out * 4);
            ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 4);
        }
    }

    uint32_t params[16] = {
        (uint32_t)batch, (uint32_t)C_in, (uint32_t)H_in, (uint32_t)W_in,
        (uint32_t)C_out, (uint32_t)KH, (uint32_t)KW,
        (uint32_t)pad_h, (uint32_t)pad_w, (uint32_t)stride_h, (uint32_t)stride_w,
        (uint32_t)H_out, (uint32_t)W_out, (uint32_t)group,
        (uint32_t)out_pad_h, (uint32_t)out_pad_w
    };
    auto paramBuf = ex.gpu->createBuffer("ct_p", 64);
    ex.gpu->writeBuffer(paramBuf, params, 64);

    auto& pl = useFp16Path
        ? ex.GetPipeline("conv_transpose2d_f16", WGSL_CONV_TRANSPOSE2D_F16, 5)
        : ex.GetPipeline("conv_transpose2d", WGSL_CONV_TRANSPOSE2D, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    int64_t total = batch * C_out * H_out * W_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "conv_transpose"});
}

// ─── Resize ──────────────────────────────────────────────────────────────────

static void opResize(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    if (!X || !X->IsValid() || X->shape.size() < 4) return;

    int64_t N = X->shape[0], C = X->shape[1], H_in = X->shape[2], W_in = X->shape[3];
    int64_t H_out = H_in, W_out = W_in;

    // Try to read scales from input[2]
    if (in.size() > 2 && in[2] && in[2]->IsValid()) {
        int64_t nScales = tensorNel(in[2]);
        if (nScales >= 4) {
            const float* scales = nullptr;
            std::vector<float> scalesBuf;
            // Prefer CPU data (avoid GPU readback)
            if (in[2]->isCpuOnly && !in[2]->cpuData.empty()) {
                scales = (const float*)in[2]->cpuData.data();
            } else if (!in[2]->cpuData.empty()) {
                scales = (const float*)in[2]->cpuData.data();
            } else if (auto* init = ex.GetInitData(n.inputs[2]); init && init->data) {
                scales = (const float*)init->data;
            }
            if (scales) {
                H_out = (int64_t)(H_in * scales[nScales-2]);
                W_out = (int64_t)(W_in * scales[nScales-1]);
            }
        }
    }

    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0])) return;

    *out[0] = ex.AllocTensor({N, C, H_out, W_out}, computeOutDtype(X->dtype));

    uint32_t params[8] = {(uint32_t)N, (uint32_t)C, (uint32_t)H_in, (uint32_t)W_in,
                           (uint32_t)H_out, (uint32_t)W_out, 0, 0};
    auto paramBuf = ex.gpu->createBuffer("resize_p", 32);
    ex.gpu->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipeline("resize_nearest", WGSL_RESIZE_NEAREST, 3);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    int64_t total = N * C * H_out * W_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "resize"});
}

REGISTER_OP(Conv, opConv)
REGISTER_OP(ConvTranspose, opConvTranspose)
REGISTER_OP(Resize, opResize)
