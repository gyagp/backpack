#include "gpu_context.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <future>
#include <thread>
#include <chrono>

// ─── GPUProfiler ─────────────────────────────────────────────────────────────

bool GPUProfiler::init(WGPUDevice dev, WGPUInstance inst, WGPUQueue q) {
    device = dev; instance = inst; queue = q;
    nextIndex = 0;
    entries.clear();

    WGPUQuerySetDescriptor qsd{};
    qsd.type = WGPUQueryType_Timestamp;
    qsd.count = MAX_TIMESTAMPS;
    querySet = wgpuDeviceCreateQuerySet(device, &qsd);
    if (!querySet) return false;

    uint64_t bufSize = MAX_TIMESTAMPS * 8;
    WGPUBufferDescriptor bd{};
    bd.usage = BUF_COPY_SRC | BUF_COPY_DST | BUF_STORAGE;
    bd.size = bufSize;
    resolveBuf = wgpuDeviceCreateBuffer(device, &bd);

    WGPUBufferDescriptor rbd{};
    rbd.usage = BUF_MAP_READ | BUF_COPY_DST;
    rbd.size = bufSize;
    readbackBuf = wgpuDeviceCreateBuffer(device, &rbd);

    return true;
}

void GPUProfiler::destroy() {
    if (querySet)    wgpuQuerySetRelease(querySet);
    if (resolveBuf)  wgpuBufferRelease(resolveBuf);
    if (readbackBuf) wgpuBufferRelease(readbackBuf);
    querySet = nullptr; resolveBuf = nullptr; readbackBuf = nullptr;
}

std::pair<uint32_t, uint32_t> GPUProfiler::allocate(const std::string& name) {
    if (nextIndex + 2 > MAX_TIMESTAMPS) return {0, 0};
    uint32_t b = nextIndex, e = nextIndex + 1;
    nextIndex += 2;
    entries.push_back({name, b, e});
    return {b, e};
}

WGPUPassTimestampWrites GPUProfiler::makeTimestampWrites(
        uint32_t beginIdx, uint32_t endIdx) {
    WGPUPassTimestampWrites tw{};
    tw.querySet = querySet;
    tw.beginningOfPassWriteIndex = beginIdx;
    tw.endOfPassWriteIndex = endIdx;
    return tw;
}

void GPUProfiler::resolveAndReport(WGPUCommandEncoder enc) {
    if (nextIndex == 0) return;

    wgpuCommandEncoderResolveQuerySet(enc, querySet, 0, nextIndex,
                                       resolveBuf, 0);
    wgpuCommandEncoderCopyBufferToBuffer(enc, resolveBuf, 0,
                                          readbackBuf, 0, nextIndex * 8);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::string sv_str(WGPUStringView sv) {
    return (sv.data && sv.length > 0) ? std::string(sv.data, sv.length) : "";
}

// ─── GPUContext::init ────────────────────────────────────────────────────────

bool GPUContext::init(WGPUBackendType backend) {
    backendType = backend;
    adapterName.clear();
    adapterDescription.clear();
    adapterLimits = {};
    deviceLimits = {};
    supportsShaderF16 = false;
    supportsSubgroups = false;
    supportsSubgroupMatrix = false;
    supportsTimestampQuery = false;

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
        "allow_unsafe_apis", "vulkan_enable_f16_on_nvidia", "use_dxc"};
    WGPUDawnTogglesDescriptor adT{};
    adT.chain.sType = static_cast<WGPUSType>(0x0005000A);
    adT.enabledToggleCount = 3;
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
    adapterName = sv_str(info.device);
    adapterDescription = sv_str(info.description);
    fprintf(stderr, "GPU: %s (%s)\n", sv_str(info.device).c_str(),
           sv_str(info.description).c_str());

    if (wgpuAdapterGetLimits(adapter, &adapterLimits) != WGPUStatus_Success) {
        fprintf(stderr, "FATAL: wgpuAdapterGetLimits failed; refusing to fall back to WebGPU default limits.\n");
        return false;
    }
    fprintf(stderr, "  maxComputeInvocationsPerWorkgroup: %u\n",
            (unsigned)adapterLimits.maxComputeInvocationsPerWorkgroup);
    fprintf(stderr, "  maxComputeWorkgroupSizeX: %u\n",
            (unsigned)adapterLimits.maxComputeWorkgroupSizeX);
    fprintf(stderr, "  maxComputeWorkgroupStorageSize: %u bytes\n",
            (unsigned)adapterLimits.maxComputeWorkgroupStorageSize);
    fprintf(stderr, "  maxStorageBufferBindingSize: %llu MB\n",
            (unsigned long long)adapterLimits.maxStorageBufferBindingSize / 1048576);
    fprintf(stderr, "  maxBufferSize: %llu MB\n",
            (unsigned long long)adapterLimits.maxBufferSize / 1048576);

    // Device features
    std::vector<WGPUFeatureName> feats;
    // Debug: enumerate all adapter features
    {
        WGPUSupportedFeatures sf{};
        wgpuAdapterGetFeatures(adapter, &sf);
        fprintf(stderr, "  Adapter features (%zu):", sf.featureCount);
        for (size_t i = 0; i < sf.featureCount && i < 30; i++)
            fprintf(stderr, " 0x%x", (unsigned)sf.features[i]);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    auto tryFeat = [&](bool& supported, std::initializer_list<uint32_t> ids) {
        supported = false;
        for (auto id : ids) {
            if (wgpuAdapterHasFeature(adapter, (WGPUFeatureName)id)) {
                feats.push_back((WGPUFeatureName)id);
                supported = true;
                return;
            }
        }
    };
    tryFeat(supportsShaderF16, {0x0B, 0x0A});
    tryFeat(supportsSubgroups, {0x12, 0x11});
    // Force-enable subgroups if adapter doesn't report it but we know GPU supports wave ops
    if (!supportsSubgroups) {
        feats.push_back((WGPUFeatureName)0x12);
        supportsSubgroups = true;
        fprintf(stderr, "  Forcing subgroups feature (0x12)\n");
    }
    tryFeat(supportsSubgroupMatrix, {0x00050034});
    tryFeat(supportsTimestampQuery, {0x09});

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
    ddesc.label = {"runtime", 7};
    ddesc.requiredFeatureCount = feats.size();
    ddesc.requiredFeatures = feats.empty() ? nullptr : feats.data();
    // Request the adapter's actual supported limits instead of a zeroed
    // WGPULimits struct, which would only ask for WebGPU defaults.
    ddesc.requiredLimits = &adapterLimits;  // request full hardware limits
    ddesc.uncapturedErrorCallbackInfo.callback =
        [](WGPUDevice const*, WGPUErrorType t, WGPUStringView m, void* ud, void*) {
            const char* n[] = {"OK","Val","OOM","Int","Unk","Lost"};
            fprintf(stderr, "[DAWN %s] %.*s\n", n[std::min((int)t,5)],
                    (int)m.length, m.data);
            // Flag OOM errors so createBuffer can detect failed allocations
            if (t == WGPUErrorType_OutOfMemory) {
                auto* ctx = static_cast<GPUContext*>(ud);
                ctx->lastAllocFailed = true;
            }
        };
    ddesc.uncapturedErrorCallbackInfo.userdata1 = this;

    // deviceLostCallbackInfo left empty (zero-initialized)

    fprintf(stderr, "  Creating device with %zu features, limits=%p...\n", feats.size(), (void*)ddesc.requiredLimits);
    fflush(stderr);
    device = wgpuAdapterCreateDevice(adapter, &ddesc);
    fprintf(stderr, "  Device created: %p\n", (void*)device);
    fflush(stderr);
    if (!device) return false;

    fprintf(stderr, "  Querying device limits...\n"); fflush(stderr);
    WGPUStatus limStatus = wgpuDeviceGetLimits(device, &deviceLimits);
    fprintf(stderr, "  Device limits status: %d\n", (int)limStatus); fflush(stderr);
    fprintf(stderr, "  Device maxBuf=%llu maxSBB=%llu maxWGS=%u maxInv=%u\n",
            (unsigned long long)deviceLimits.maxBufferSize,
            (unsigned long long)deviceLimits.maxStorageBufferBindingSize,
            (unsigned)deviceLimits.maxComputeWorkgroupStorageSize,
            (unsigned)deviceLimits.maxComputeInvocationsPerWorkgroup);
    fflush(stderr);
    if (limStatus != WGPUStatus_Success) {
        fprintf(stderr, "FATAL: wgpuDeviceGetLimits failed after device creation.\n");
        return false;
    }

    if (deviceLimits.maxStorageBufferBindingSize != adapterLimits.maxStorageBufferBindingSize ||
        deviceLimits.maxBufferSize != adapterLimits.maxBufferSize ||
        deviceLimits.maxComputeWorkgroupStorageSize != adapterLimits.maxComputeWorkgroupStorageSize ||
        deviceLimits.maxComputeInvocationsPerWorkgroup != adapterLimits.maxComputeInvocationsPerWorkgroup) {
        fprintf(stderr,
                "WARNING: device limits differ from adapter limits; device may not be using full hardware limits.\n");
        fprintf(stderr,
                "  adapter: maxStorageBufferBindingSize=%llu maxBufferSize=%llu maxComputeWorkgroupStorageSize=%u maxComputeInvocationsPerWorkgroup=%u\n",
                (unsigned long long)adapterLimits.maxStorageBufferBindingSize,
                (unsigned long long)adapterLimits.maxBufferSize,
                (unsigned)adapterLimits.maxComputeWorkgroupStorageSize,
                (unsigned)adapterLimits.maxComputeInvocationsPerWorkgroup);
        fprintf(stderr,
                "  device : maxStorageBufferBindingSize=%llu maxBufferSize=%llu maxComputeWorkgroupStorageSize=%u maxComputeInvocationsPerWorkgroup=%u\n",
                (unsigned long long)deviceLimits.maxStorageBufferBindingSize,
                (unsigned long long)deviceLimits.maxBufferSize,
                (unsigned)deviceLimits.maxComputeWorkgroupStorageSize,
                (unsigned)deviceLimits.maxComputeInvocationsPerWorkgroup);
    }

    queue = wgpuDeviceGetQueue(device);
    fprintf(stderr, "  Queue: %p\n", (void*)queue); fflush(stderr);
    fprintf(stderr, "  Features: f16=%s subgroups=%s subgroup_matrix=%s timestamps=%s\n",
           supportsShaderF16 ? "yes" : "no",
           supportsSubgroups ? "yes" : "no",
           supportsSubgroupMatrix ? "yes" : "no",
           supportsTimestampQuery ? "yes" : "no");
    fflush(stderr);
    fprintf(stderr, "  Features: f16=%s subgroups=%s subgroup_matrix=%s timestamps=%s\n",
           supportsShaderF16 ? "yes" : "no",
           supportsSubgroups ? "yes" : "no",
           supportsSubgroupMatrix ? "yes" : "no",
           supportsTimestampQuery ? "yes" : "no");
    fflush(stdout);
    fprintf(stderr, "  GPU init complete!\n"); fflush(stderr);
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

int GPUContext::poolBucket(uint64_t size) const {
    if (size <= (1ULL << POOL_MIN_BITS)) return 0;
    int bits = 0;
    uint64_t s = size - 1;
    while (s > 0) { s >>= 1; bits++; }
    return std::min(std::max(bits - POOL_MIN_BITS, 0), POOL_BUCKETS - 1);
}

GPUBuffer GPUContext::createBuffer(const std::string& name, uint64_t size,
                                   uint64_t usage, bool mappedAtCreation) {
    // Try pool first (only for default usage, non-mapped)
    if (bufferPoolEnabled && usage == BUF_DEFAULT && !mappedAtCreation && size > 0) {
        int bucket = poolBucket(size);
        if (!pool_[bucket].empty()) {
            GPUBuffer buf = pool_[bucket].back();
            pool_[bucket].pop_back();
            poolHitCount++;
            createBufferCount++;
            totalAllocatedBytes += buf.size;
            totalAllocCount++;
            if (totalAllocatedBytes > peakAllocatedBytes)
                peakAllocatedBytes = totalAllocatedBytes;
            return buf;
        }
        // Round up to bucket size for new allocation
        uint64_t bucketSize = 1ULL << (bucket + POOL_MIN_BITS);
        size = std::max(size, bucketSize);
    }
    createBufferCount++;

    WGPUBufferDescriptor d{};
    d.label = {name.c_str(), name.size()};
    d.usage = usage;
    d.size = size;
    d.mappedAtCreation = mappedAtCreation ? 1u : 0u;
    GPUBuffer buf{wgpuDeviceCreateBuffer(device, &d), size};
    // Dawn may return a non-null "error" buffer on OOM. Check via the error callback flag.
    if (lastAllocFailed) {
        lastAllocFailed = false;
        if (buf.handle) { wgpuBufferRelease(buf.handle); buf.handle = nullptr; }
    }
    if (!buf.handle) {
        // Allocation failed — flush pool and retry
        fprintf(stderr, "  [gpu] alloc '%s' %llu bytes FAILED, flushing pool...\n",
                name.c_str(), (unsigned long long)size);
        fflush(stderr);
        flushBufferPool();
        lastAllocFailed = false;
        buf = {wgpuDeviceCreateBuffer(device, &d), size};
        if (lastAllocFailed) {
            lastAllocFailed = false;
            if (buf.handle) { wgpuBufferRelease(buf.handle); buf.handle = nullptr; }
        }
        if (!buf.handle) {
            fprintf(stderr, "  [gpu] ALLOC FAILED (after pool flush): name='%s' size=%llu bytes\n",
                    name.c_str(), (unsigned long long)size);
            fflush(stderr);
        }
    }
    if (buf.handle) {
        totalAllocatedBytes += buf.size;
        totalAllocCount++;
        if (totalAllocatedBytes > peakAllocatedBytes)
            peakAllocatedBytes = totalAllocatedBytes;
    }
    // Only register device-level resources (prefixed with '_') in the shared
    // buffers_ map. Per-model buffers (weights, intermediates) are tracked by
    // GraphExecutor::tensorStore_ and would cause name collisions across models.
    if (!name.empty() && name[0] == '_')
        buffers_[name] = buf;
    return buf;
}

void GPUContext::releaseBuffer(GPUBuffer buf) {
    if (!buf.handle) return;
    // Don't release aliased buffer views — only the parent buffer is released.
    if (buf.offset != 0) return;
    if (totalAllocatedBytes >= buf.size)
        totalAllocatedBytes -= buf.size;
    int bucket = poolBucket(buf.size);
    if (pool_[bucket].size() < 256) {
        pool_[bucket].push_back(buf);
    } else {
        wgpuBufferDestroy(buf.handle);
        wgpuBufferRelease(buf.handle);
    }
}

void GPUContext::flushBufferPool() {
    size_t totalFreed = 0;
    for (auto& bucket : pool_) {
        for (auto& buf : bucket) {
            if (buf.handle) {
                totalFreed += buf.size;
                wgpuBufferDestroy(buf.handle);
                wgpuBufferRelease(buf.handle);
            }
        }
        bucket.clear();
    }
    if (totalFreed > 0)
        fprintf(stderr, "  [pool] flushed %llu MB\n", (unsigned long long)totalFreed / 1048576);
}

GPUBuffer GPUContext::getBuffer(const std::string& name) const {
    auto it = buffers_.find(name);
    return (it != buffers_.end()) ? it->second : GPUBuffer{nullptr, 0};
}

void GPUContext::writeBuffer(GPUBuffer buf, const void* data, uint64_t size,
                             uint64_t offset) {
    using hrc = std::chrono::high_resolution_clock;
    auto t0 = hrc::now();
    // Add buffer base offset for aliased/view buffers
    offset += buf.offset;

    // Record for fast decode replay
    if (captureWritesCb_ && size > 0) {
        captureWritesCb_(buf.handle, offset, data, size, captureWritesCtx_);
    }
    // WebGPU requires writeBuffer size to be a multiple of 4
    if (size > 0 && (size & 3)) {
        uint64_t aligned = size & ~(uint64_t)3;
        if (aligned > 0)
            wgpuQueueWriteBuffer(queue, buf.handle, offset, data, aligned);
        // Write remaining bytes padded to 4
        uint8_t padded[4] = {0};
        memcpy(padded, (const uint8_t*)data + aligned, (size_t)(size - aligned));
        wgpuQueueWriteBuffer(queue, buf.handle, offset + aligned, padded, 4);
    } else {
        wgpuQueueWriteBuffer(queue, buf.handle, offset, data, size);
    }
    auto t1 = hrc::now();
    timing.write_buf_ns += (t1 - t0).count();
}

void GPUContext::writeBufferRaw(WGPUBuffer handle, uint64_t offset, const void* data, uint64_t size) {
    wgpuQueueWriteBuffer(queue, handle, offset, data, size);
}

// ─── Pipelines ───────────────────────────────────────────────────────────────

bool GPUContext::hasPipeline(const std::string& name) const {
    return pipelines_.find(name) != pipelines_.end();
}

const CompiledPipeline* GPUContext::findPipeline(const std::string& name) const {
    auto it = pipelines_.find(name);
    return (it != pipelines_.end()) ? &it->second : nullptr;
}

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

int GPUContext::warmupPipelines(
        const std::vector<std::tuple<std::string, std::string, uint32_t>>& specs) {
    // Filter to only uncached specs
    std::vector<size_t> uncached;
    for (size_t i = 0; i < specs.size(); i++) {
        if (!hasPipeline(std::get<0>(specs[i])))
            uncached.push_back(i);
    }
    if (uncached.empty()) return 0;

    // Step 1: Compile shaders in parallel (wgpuDeviceCreateShaderModule is thread-safe in Dawn)
    std::vector<std::future<WGPUShaderModule>> futures(uncached.size());
    for (size_t j = 0; j < uncached.size(); j++) {
        auto& [name, wgsl, nb] = specs[uncached[j]];
        futures[j] = std::async(std::launch::async, [this, &wgsl]() {
            WGPUShaderSourceWGSL src{};
            src.chain.sType = WGPUSType_ShaderSourceWGSL;
            src.code = {wgsl.c_str(), wgsl.size()};
            WGPUShaderModuleDescriptor smD{};
            smD.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&src);
            return wgpuDeviceCreateShaderModule(device, &smD);
        });
    }

    // Step 2: Collect compiled shaders and create pipelines asynchronously
    struct PipelineResult {
        WGPUCreatePipelineAsyncStatus status;
        WGPUComputePipeline pipeline;
        bool completed = false;
    };
    std::vector<PipelineResult> results(uncached.size());
    int asyncPending = 0;

    for (size_t j = 0; j < uncached.size(); j++) {
        auto& [name, wgsl, numBindings] = specs[uncached[j]];
        WGPUShaderModule shader = futures[j].get();
        if (!shader) {
            fprintf(stderr, "warmup: shader compilation failed: %s\n", name.c_str());
            results[j].status = WGPUCreatePipelineAsyncStatus_InternalError;
            results[j].pipeline = nullptr;
            results[j].completed = true;
            continue;
        }
        // Store shader now for later use
        CompiledPipeline p{};
        p.numBindings = numBindings;
        p.shader = shader;
        pipelines_[name] = p;

        WGPUComputePipelineDescriptor cpD{};
        memset(&cpD, 0, sizeof(cpD));
        cpD.label = {name.c_str(), name.size()};
        cpD.layout = nullptr;
        cpD.compute.module = shader;
        cpD.compute.entryPoint = {"main", 4};

        results[j].status = WGPUCreatePipelineAsyncStatus_InternalError;
        results[j].pipeline = nullptr;
        results[j].completed = false;

        WGPUCreateComputePipelineAsyncCallbackInfo asyncCb{};
        asyncCb.mode = WGPUCallbackMode_WaitAnyOnly;
        asyncCb.callback = [](WGPUCreatePipelineAsyncStatus s,
                              WGPUComputePipeline p, WGPUStringView,
                              void* ud, void*) {
            auto* r = static_cast<PipelineResult*>(ud);
            r->status = s;
            r->pipeline = p;
            r->completed = true;
        };
        asyncCb.userdata1 = &results[j];
        wgpuDeviceCreateComputePipelineAsync(device, &cpD, asyncCb);
        asyncPending++;
    }

    // Poll until all async pipeline creations complete
    if (asyncPending > 0) {
        wgpuInstanceProcessEvents(instance);
        // Wait for all to complete
        for (int remaining = asyncPending; remaining > 0; ) {
            wgpuInstanceProcessEvents(instance);
            remaining = 0;
            for (size_t j = 0; j < uncached.size(); j++) {
                if (!results[j].completed)
                    remaining++;
            }
        }
    }

    // Finalize: extract bind group layouts
    for (size_t j = 0; j < uncached.size(); j++) {
        auto& [name, wgsl, numBindings] = specs[uncached[j]];
        if (results[j].status != WGPUCreatePipelineAsyncStatus_Success ||
            !results[j].pipeline) {
            if (results[j].status != WGPUCreatePipelineAsyncStatus_InternalError)
                fprintf(stderr, "warmup: async pipeline creation failed: %s\n", name.c_str());
            pipelines_.erase(name);
            continue;
        }
        auto& p = pipelines_[name];
        p.pipeline = results[j].pipeline;
        p.bgLayout = wgpuComputePipelineGetBindGroupLayout(p.pipeline, 0);
        p.pplLayout = nullptr;
    }
    return (int)uncached.size();
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

void GPUContext::submitOnly(const std::vector<Dispatch>& dispatches,
                            bool singlePass) {
    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    if (singlePass) {
        WGPUComputePassDescriptor cpD{};
        auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
        for (auto& d : dispatches) {
            wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
        }
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    } else {
        for (auto& d : dispatches) {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
    }

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);

    // Release bind groups — they held references to GPU buffers
    for (auto& d : dispatches) {
        if (d.bindGroup) wgpuBindGroupRelease(d.bindGroup);
    }
}

void GPUContext::submitOnlyProfiled(const std::vector<Dispatch>& dispatches,
                                     GPUProfiler& profiler) {
    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    for (auto& d : dispatches) {
        auto [bIdx, eIdx] = profiler.allocate(d.name);
        auto tw = profiler.makeTimestampWrites(bIdx, eIdx);
        WGPUComputePassDescriptor cpD{};
        cpD.timestampWrites = &tw;
        auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
        wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Resolve timestamps in same command buffer
    profiler.resolveAndReport(enc);

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);

    for (auto& d : dispatches) {
        if (d.bindGroup) wgpuBindGroupRelease(d.bindGroup);
    }
}

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

    // Release bind groups
    for (auto& d : dispatches) {
        if (d.bindGroup) wgpuBindGroupRelease(d.bindGroup);
    }
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
        GPUBuffer src, uint64_t readSize,
        bool passPerDispatch) {
    auto rb = getOrCreateReadbackBuf(readSize);

    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    if (passPerDispatch) {
        // Each dispatch gets its own compute pass (Vulkan correctness)
        for (auto& d : dispatches) {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
    } else {
        // Single compute pass — Dawn inserts barriers automatically (D3D12)
        WGPUComputePassDescriptor cpD{};
        auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
        for (auto& d : dispatches) {
            wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
        }
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

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

std::vector<uint8_t> GPUContext::readBuffer(GPUBuffer src, uint64_t readSize) {
    auto rb = getOrCreateReadbackBuf(readSize);

    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src.handle, src.offset, rb, 0, readSize);
    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);

    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(rb, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(instance, 1, &mw, UINT64_MAX);

    std::vector<uint8_t> out(readSize);
    if (ms.status == 1) {
        auto ptr = wgpuBufferGetConstMappedRange(rb, 0, readSize);
        memcpy(out.data(), ptr, readSize);
        wgpuBufferUnmap(rb);
    }
    return out;
}

std::vector<uint8_t> GPUContext::mapReadbackBuffer(uint64_t readSize) {
    auto rb = getOrCreateReadbackBuf(readSize);
    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(rb, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(instance, 1, &mw, UINT64_MAX);
    std::vector<uint8_t> out(readSize);
    if (ms.status == 1) {
        auto ptr = wgpuBufferGetConstMappedRange(rb, 0, readSize);
        memcpy(out.data(), ptr, readSize);
        wgpuBufferUnmap(rb);
    }
    return out;
}

std::vector<uint8_t> GPUContext::submitAndReadbackProfiled(
        const std::vector<Dispatch>& dispatches,
        GPUBuffer src, uint64_t readSize,
        GPUProfiler& profiler) {
    auto rb = getOrCreateReadbackBuf(readSize);

    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    // Each dispatch gets its own compute pass with timestamp writes
    for (auto& d : dispatches) {
        auto [bIdx, eIdx] = profiler.allocate(d.name);
        auto tw = profiler.makeTimestampWrites(bIdx, eIdx);
        WGPUComputePassDescriptor cpD{};
        cpD.timestampWrites = &tw;
        auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
        wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    wgpuCommandEncoderCopyBufferToBuffer(enc, src.handle, 0, rb, 0, readSize);

    // Resolve timestamps into the same command buffer
    profiler.resolveAndReport(enc);

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);

    // Map result buffer
    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(rb, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(instance, 1, &mw, UINT64_MAX);

    std::vector<uint8_t> out(readSize);
    if (ms.status == 1) {
        auto ptr = wgpuBufferGetConstMappedRange(rb, 0, readSize);
        memcpy(out.data(), ptr, readSize);
        wgpuBufferUnmap(rb);
    }
    return out;
}

WGPUFuture GPUContext::submitAndCopyAsync(const std::vector<Dispatch>& dispatches,
                                           GPUBuffer src, uint64_t readSize,
                                           WGPUBuffer stagingBuf,
                                           bool passPerDispatch) {
    using hrc = std::chrono::high_resolution_clock;
    auto t0 = hrc::now();

    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(device, &enD);

    if (passPerDispatch) {
        for (auto& d : dispatches) {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
    } else {
        WGPUComputePassDescriptor cpD{};
        auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
        for (auto& d : dispatches) {
            wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
        }
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    wgpuCommandEncoderCopyBufferToBuffer(enc, src.handle, 0,
                                          stagingBuf, 0, readSize);

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);

    auto t1 = hrc::now();
    timing.encode_ns += (t1 - t0).count();

    wgpuQueueSubmit(queue, 1, &cb);

    auto t2 = hrc::now();
    timing.submit_ns += (t2 - t1).count();

    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);

    // Start async map (non-blocking until WaitAny is called)
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void*) {};
    auto future = wgpuBufferMapAsync(stagingBuf, 1, 0, readSize, mcb);

    auto t3 = hrc::now();
    timing.map_start_ns += (t3 - t2).count();
    timing.count++;

    return future;
}

int32_t GPUContext::completeAsyncMapI32(WGPUBuffer stagingBuf, WGPUFuture future) {
    using hrc = std::chrono::high_resolution_clock;
    auto t0 = hrc::now();

    // Wait for the pending map to complete
    WGPUFutureWaitInfo fw{future, 0};
    wgpuInstanceWaitAny(instance, 1, &fw, UINT64_MAX);

    auto t1 = hrc::now();
    timing.wait_ns += (t1 - t0).count();

    int32_t val = 0;
    auto ptr = wgpuBufferGetConstMappedRange(stagingBuf, 0, 4);
    if (ptr) memcpy(&val, ptr, 4);
    wgpuBufferUnmap(stagingBuf);

    auto t2 = hrc::now();
    timing.unmap_ns += (t2 - t1).count();

    return val;
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
