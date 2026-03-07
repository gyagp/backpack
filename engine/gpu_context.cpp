#include "gpu_context.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::string sv_str(WGPUStringView sv) {
    return (sv.data && sv.length > 0) ? std::string(sv.data, sv.length) : "";
}

// ─── GPUContext::init ────────────────────────────────────────────────────────

bool GPUContext::init(WGPUBackendType backend) {
    // Instance with TimedWaitAny
    WGPUInstanceFeatureName instFeat[] = {
        static_cast<WGPUInstanceFeatureName>(0x01) };
    WGPUInstanceDescriptor idesc{};
    idesc.requiredFeatureCount = 1;
    idesc.requiredFeatures = instFeat;
    instance = wgpuCreateInstance(&idesc);
    if (!instance) return false;

    // Adapter
    struct { WGPUAdapter a; bool ok; } st{nullptr, false};
    const char* adapterToggles[] = {
        "allow_unsafe_apis", "vulkan_enable_f16_on_nvidia"};
    WGPUDawnTogglesDescriptor adT{};
    adT.chain.sType = static_cast<WGPUSType>(0x0005000A);
    adT.enabledToggleCount = 2;
    adT.enabledToggles = adapterToggles;

    WGPURequestAdapterOptions opts{};
    opts.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&adT);
    opts.featureLevel = WGPUFeatureLevel_Core;
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    opts.backendType = backend;

    WGPURequestAdapterCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = [](WGPURequestAdapterStatus s, WGPUAdapter a,
                     WGPUStringView, void* ud, void*) {
        auto* p = static_cast<decltype(&st)>(ud);
        p->a = (s == WGPURequestAdapterStatus_Success) ? a : nullptr;
        p->ok = true;
    };
    cb.userdata1 = &st;
    auto fut = wgpuInstanceRequestAdapter(instance, &opts, cb);
    WGPUFutureWaitInfo w{fut, 0};
    wgpuInstanceWaitAny(instance, 1, &w, UINT64_MAX);
    adapter = st.a;
    if (!adapter) return false;

    WGPUAdapterInfo info{};
    wgpuAdapterGetInfo(adapter, &info);
    printf("GPU: %s (%s)\n", sv_str(info.device).c_str(),
           sv_str(info.description).c_str());

    WGPULimits limits{};
    wgpuAdapterGetLimits(adapter, &limits);

    // Device features
    std::vector<WGPUFeatureName> feats;
    auto tryFeat = [&](std::initializer_list<uint32_t> ids) {
        for (auto id : ids)
            if (wgpuAdapterHasFeature(adapter, (WGPUFeatureName)id))
                { feats.push_back((WGPUFeatureName)id); return; }
    };
    tryFeat({0x0B, 0x0A});  // ShaderF16
    tryFeat({0x12, 0x11});  // Subgroups
    tryFeat({0x00050034});  // SubgroupMatrix
    tryFeat({0x09});        // TimestampQuery

    const char* devE[] = {"skip_validation", "disable_robustness",
                          "d3d_disable_ieee_strictness"};
    const char* devD[] = {"lazy_clear_resource_on_first_use",
                          "timestamp_quantization"};
    WGPUDawnTogglesDescriptor devT{};
    devT.chain.sType = static_cast<WGPUSType>(0x0005000A);
    devT.enabledToggleCount = 3;  devT.enabledToggles = devE;
    devT.disabledToggleCount = 2; devT.disabledToggles = devD;

    WGPUDeviceDescriptor ddesc{};
    ddesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&devT);
    ddesc.label = {"engine", 6};
    ddesc.requiredFeatureCount = feats.size();
    ddesc.requiredFeatures = feats.empty() ? nullptr : feats.data();
    ddesc.requiredLimits = &limits;
    ddesc.uncapturedErrorCallbackInfo.callback =
        [](WGPUDevice const*, WGPUErrorType t, WGPUStringView m, void*, void*) {
            const char* n[] = {"OK","Val","OOM","Int","Unk","Lost"};
            fprintf(stderr, "[DAWN %s] %.*s\n", n[std::min((int)t,5)],
                    (int)m.length, m.data);
        };

    device = wgpuAdapterCreateDevice(adapter, &ddesc);
    if (!device) return false;
    queue = wgpuDeviceGetQueue(device);
    return true;
}

void GPUContext::destroy() {
    if (readbackBuf_) wgpuBufferRelease(readbackBuf_);
    for (auto& [_, b] : buffers_) wgpuBufferRelease(b.handle);
    for (auto& [_, p] : pipelines_) {
        wgpuComputePipelineRelease(p.pipeline);
        wgpuShaderModuleRelease(p.shader);
        wgpuBindGroupLayoutRelease(p.bgLayout);
        wgpuPipelineLayoutRelease(p.pplLayout);
    }
    if (queue) wgpuQueueRelease(queue);
    if (device) wgpuDeviceRelease(device);
    if (adapter) wgpuAdapterRelease(adapter);
    if (instance) wgpuInstanceRelease(instance);
}

// ─── Buffers ─────────────────────────────────────────────────────────────────

GPUBuffer GPUContext::createBuffer(const std::string& name, uint64_t size,
                                   uint64_t usage, bool mappedAtCreation) {
    WGPUBufferDescriptor d{};
    d.label = {name.c_str(), name.size()};
    d.usage = usage;
    d.size = size;
    d.mappedAtCreation = mappedAtCreation ? 1u : 0u;
    GPUBuffer buf{wgpuDeviceCreateBuffer(device, &d), size};
    buffers_[name] = buf;
    return buf;
}

GPUBuffer GPUContext::getBuffer(const std::string& name) const {
    auto it = buffers_.find(name);
    return (it != buffers_.end()) ? it->second : GPUBuffer{nullptr, 0};
}

void GPUContext::writeBuffer(GPUBuffer buf, const void* data, uint64_t size,
                             uint64_t offset) {
    wgpuQueueWriteBuffer(queue, buf.handle, offset, data, size);
}

// ─── Pipelines ───────────────────────────────────────────────────────────────

const CompiledPipeline& GPUContext::getOrCreatePipeline(
        const std::string& name, const std::string& wgsl, uint32_t numBindings) {
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) return it->second;

    CompiledPipeline p{};
    p.numBindings = numBindings;

    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = {wgsl.c_str(), wgsl.size()};
    WGPUShaderModuleDescriptor smD{};
    smD.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&src);
    p.shader = wgpuDeviceCreateShaderModule(device, &smD);

    // Use auto layout — let Dawn infer binding types from the WGSL shader.
    // This correctly handles read vs read_write storage buffers.
    WGPUComputePipelineDescriptor cpD{};
    memset(&cpD, 0, sizeof(cpD));
    cpD.label = {name.c_str(), name.size()};
    cpD.layout = nullptr;  // auto layout
    cpD.compute.module = p.shader;
    cpD.compute.entryPoint = {"main", 4};
    p.pipeline = wgpuDeviceCreateComputePipeline(device, &cpD);
    if (!p.pipeline) {
        fprintf(stderr, "FATAL: pipeline creation failed: %s\n", name.c_str());
        exit(1);
    }

    // Get the auto-generated bind group layout
    p.bgLayout = wgpuComputePipelineGetBindGroupLayout(p.pipeline, 0);
    if (!p.bgLayout) {
        fprintf(stderr, "FATAL: GetBindGroupLayout(0) returned null for: %s\n", name.c_str());
        exit(1);
    }
    p.pplLayout = nullptr;

    pipelines_[name] = p;
    return pipelines_[name];
}

// ─── Bind groups ─────────────────────────────────────────────────────────────

WGPUBindGroup GPUContext::createBindGroup(
        const CompiledPipeline& pl,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& entries) {
    std::vector<WGPUBindGroupEntry> bge(entries.size());
    for (size_t i = 0; i < entries.size(); i++) {
        memset(&bge[i], 0, sizeof(WGPUBindGroupEntry));
        bge[i].binding = entries[i].first;
        bge[i].buffer  = entries[i].second.handle;
        bge[i].offset  = 0;
        bge[i].size    = entries[i].second.size;
    }
    WGPUBindGroupDescriptor d{};
    memset(&d, 0, sizeof(d));
    d.layout = pl.bgLayout;
    d.entryCount = (uint32_t)bge.size();
    d.entries = bge.data();
    WGPUBindGroup result = wgpuDeviceCreateBindGroup(device, &d);
    if (!result) {
        fprintf(stderr, "FATAL: wgpuDeviceCreateBindGroup returned null! entries=%u layout=%p\n",
                (unsigned)bge.size(), (void*)pl.bgLayout);
    }
    return result;
}

// ─── Dispatch ────────────────────────────────────────────────────────────────

void GPUContext::submitDispatches(const std::vector<Dispatch>& dispatches) {
    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    WGPUComputePassDescriptor cpD{};
    auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);

    for (auto& d : dispatches) {
        wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
    }

    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);
}

WGPUBuffer GPUContext::getOrCreateReadbackBuf(uint64_t size) {
    if (readbackBuf_ && readbackBufSize_ >= size) return readbackBuf_;
    if (readbackBuf_) wgpuBufferRelease(readbackBuf_);
    WGPUBufferDescriptor d{};
    d.label = {"readback", 8};
    d.usage = BUF_MAP_READ | BUF_COPY_DST;
    d.size = size;
    readbackBuf_ = wgpuDeviceCreateBuffer(device, &d);
    readbackBufSize_ = size;
    return readbackBuf_;
}

std::vector<uint8_t> GPUContext::submitAndReadback(
        const std::vector<Dispatch>& dispatches,
        GPUBuffer src, uint64_t readSize) {
    auto rb = getOrCreateReadbackBuf(readSize);

    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    WGPUComputePassDescriptor cpD{};
    auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
    for (auto& d : dispatches) {
        wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuCommandEncoderCopyBufferToBuffer(enc, src.handle, 0, rb, 0, readSize);

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);

    // Synchronous map
    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(rb, 1 /*READ*/, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(instance, 1, &mw, UINT64_MAX);

    std::vector<uint8_t> out(readSize);
    if (ms.status == 1 /*Success*/) {
        auto ptr = wgpuBufferGetConstMappedRange(rb, 0, readSize);
        memcpy(out.data(), ptr, readSize);
        wgpuBufferUnmap(rb);
    }
    return out;
}

void GPUContext::waitForQueue() {
    struct { bool done; } s{false};
    WGPUBufferMapCallbackInfo cb{};  // same struct layout as QueueWorkDoneCallbackInfo
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* u, void*) {
        static_cast<decltype(&s)>(u)->done = true;
    };
    cb.userdata1 = &s;

    // wgpuQueueOnSubmittedWorkDone has same ABI as the callback info
    auto f = wgpuQueueOnSubmittedWorkDone(queue,
        *reinterpret_cast<WGPUQueueWorkDoneCallbackInfo*>(&cb));
    WGPUFutureWaitInfo w{f, 0};
    wgpuInstanceWaitAny(instance, 1, &w, UINT64_MAX);
}
