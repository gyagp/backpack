/**
 * execution_context.cpp — Per-session mutable execution state.
 *
 * Contains method implementations for ExecutionContext (param pool,
 * dispatch batching, fast decode capture/replay, profiling) and
 * OpContext forwarding methods.
 */

#include "execution_context.h"
#include "graph_executor.h"
#include "clock_calibration.h"
#include "profile_html.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>

// Debug label for current op (shared file-scope state — fine for
// single-threaded interleaved sessions, would need thread_local for
// true concurrent threading).
extern std::string g_currentOpLabel;
extern const char* g_currentOp;

// ─── ExecutionContext destructor ─────────────────────────────────────────────

ExecutionContext::~ExecutionContext() {
    if (!gpu) return;

    // Release captured bind groups
    ReleaseCaptured();

    // Release param pool buffers
    for (int b = 0; b < PARAM_POOL_BUCKETS; b++) {
        for (auto& buf : paramPool_[b].buffers) {
            if (buf.handle) gpu->releaseBuffer(buf);
        }
        paramPool_[b].buffers.clear();
    }

    // Release tensor plan buffers
    if (tensorPlanValid_) {
        std::set<WGPUBuffer> released;
        for (auto& [name, alloc] : tensorPlan_) {
            if (alloc.buffer.handle && released.find(alloc.buffer.handle) == released.end()) {
                released.insert(alloc.buffer.handle);
                gpu->releaseBuffer(alloc.buffer);
            }
        }
        tensorPlan_.clear();
    }

    // Release deferred buffers
    for (auto& buf : deferredBufferReleases_) {
        if (buf.handle) gpu->releaseBuffer(buf);
    }
    deferredBufferReleases_.clear();

    // Release intermediate tensors
    {
        std::set<WGPUBuffer> released;
        for (auto& [name, tensor] : tensorStore_) {
            if (tensor.buffer.handle && released.find(tensor.buffer.handle) == released.end()) {
                released.insert(tensor.buffer.handle);
                gpu->releaseBuffer(tensor.buffer);
            }
        }
        tensorStore_.clear();
    }

    // Clean up profiler
    if (gpuProfiler) {
        delete gpuProfiler;
        gpuProfiler = nullptr;
    }
    if (clockCalibration) {
        delete clockCalibration;
        clockCalibration = nullptr;
    }
}

// ─── Param Buffer Pool ──────────────────────────────────────────────────────

GPUBuffer ExecutionContext::getParamBuffer(uint32_t sizeBytes) {
    int bucket;
    if (sizeBytes <= 16) { bucket = 0; sizeBytes = 16; }
    else if (sizeBytes <= 32) { bucket = 1; sizeBytes = 32; }
    else if (sizeBytes <= 48) { bucket = 2; sizeBytes = 48; }
    else { bucket = 3; sizeBytes = 64; }

    // During fast decode capture: larger pool to avoid wrapping
    if (fastDecodeState_ == FastDecodeState::Capturing) {
        auto& pool = paramPool_[bucket];
        if (pool.buffers.empty() || (int)pool.buffers.size() < 2048) {
            int oldSize = (int)pool.buffers.size();
            pool.buffers.resize(2048);
            for (int i = oldSize; i < 2048; i++)
                pool.buffers[i] = gpu->createBuffer("param_pool", sizeBytes);
        }
        GPUBuffer buf = pool.buffers[pool.nextIdx];
        pool.nextIdx = (pool.nextIdx + 1) % (int)pool.buffers.size();
        return buf;
    }

    auto& pool = paramPool_[bucket];
    if (pool.buffers.empty()) {
        pool.buffers.resize(PARAM_POOL_SIZE);
        for (int i = 0; i < PARAM_POOL_SIZE; i++)
            pool.buffers[i] = gpu->createBuffer("param_pool", sizeBytes);
        pool.nextIdx = 0;
    }
    GPUBuffer buf = pool.buffers[pool.nextIdx];
    pool.nextIdx = (pool.nextIdx + 1) % (int)pool.buffers.size();
    return buf;
}

// ─── Dispatch Batching ──────────────────────────────────────────────────────

void ExecutionContext::QueueDispatch(WGPUComputePipeline pipeline, WGPUBindGroup bg,
                                     uint32_t gx, uint32_t gy, uint32_t gz, const char* name) {
    pendingDispatches_.push_back({pipeline, bg, gx, gy, gz, name, {}});
    if (fastDecodeState_ == FastDecodeState::Capturing && !lastCapturedBindings_.empty()) {
        pendingDispatches_.back().capturedBindings = std::move(lastCapturedBindings_);
        lastCapturedBindings_.clear();
    }
}

void ExecutionContext::QueueCopy(GPUBuffer src, uint64_t srcOffset,
                                 GPUBuffer dst, uint64_t dstOffset, uint64_t size) {
    if (!src.handle || !dst.handle || size == 0) return;
    srcOffset += src.offset;
    dstOffset += dst.offset;
    if (srcOffset + size > src.size) size = (src.size > srcOffset) ? src.size - srcOffset : 0;
    if (dstOffset + size > dst.size) size = (dst.size > dstOffset) ? dst.size - dstOffset : 0;
    if (size == 0) return;
    size = size & ~3ULL;
    srcOffset = srcOffset & ~3ULL;
    dstOffset = dstOffset & ~3ULL;
    if (size == 0) return;
    if (src.handle == dst.handle && srcOffset == dstOffset) return;

    if (!pendingCopies_.empty()) {
        auto& last = pendingCopies_.back();
        if (last.src.handle == src.handle && last.dst.handle == dst.handle &&
            last.srcOff + last.size == srcOffset &&
            last.dstOff + last.size == dstOffset) {
            last.size += size;
            return;
        }
    }
    pendingCopies_.push_back({src, srcOffset, dst, dstOffset, size});
}

void ExecutionContext::flushToEncoder() {
    if (pendingDispatches_.empty() && pendingCopies_.empty()) return;

    // Fast Decode Capture: save AND submit
    if (fastDecodeState_ == FastDecodeState::Capturing) {
        CapturedFlush flush;
        for (auto& d : pendingDispatches_) {
            CapturedCommand cmd;
            cmd.pipeline = d.pipeline;
            cmd.gx = d.gx; cmd.gy = d.gy; cmd.gz = d.gz;
            cmd.name = d.name;
            cmd.bindings = std::move(d.capturedBindings);
            if (d.bindGroup) {
                wgpuBindGroupAddRef(d.bindGroup);
                cmd.bindGroup = d.bindGroup;
            }
            flush.dispatches.push_back(std::move(cmd));
        }
        for (auto& c : pendingCopies_)
            flush.copies.push_back({c.src, c.srcOff, c.dst, c.dstOff, c.size});
        capturedFlushes_.push_back(std::move(flush));
    }

    WGPUCommandEncoderDescriptor enD{};
    auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

    if (!pendingDispatches_.empty()) {
        if (gpuProfiler && gpuProfiler->enabled()) {
            for (auto& d : pendingDispatches_) {
                auto [bIdx, eIdx] = gpuProfiler->allocate(d.name);
                auto tw = gpuProfiler->makeTimestampWrites(bIdx, eIdx);
                WGPUComputePassDescriptor cpD{};
                cpD.timestampWrites = &tw;
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
            for (auto& d : pendingDispatches_) {
                wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, d.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, d.gx, d.gy, d.gz);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
    }
    for (auto& c : pendingCopies_)
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            c.src.handle, c.srcOff, c.dst.handle, c.dstOff, c.size);

    WGPUCommandBufferDescriptor cbD{};
    auto cb = wgpuCommandEncoderFinish(enc, &cbD);
    wgpuQueueSubmit(gpu->queue, 1, &cb);
    wgpuCommandEncoderRelease(enc);
    wgpuCommandBufferRelease(cb);
    for (auto& d : pendingDispatches_)
        if (d.bindGroup) wgpuBindGroupRelease(d.bindGroup);
    pendingDispatches_.clear();
    pendingCopies_.clear();
}

void ExecutionContext::FlushPendingWork() {
    bool hadWork = !pendingDispatches_.empty() || !pendingCopies_.empty();
    flushToEncoder();
    gpu->waitForQueue();
    if (hadWork) {
        flushCount_++;
        std::string key = g_currentOp ? g_currentOp : "(no-op-context)";
        flushSources_[key]++;
    }
}

void ExecutionContext::SubmitPending() {
    flushToEncoder();
}

// ─── Fast Decode: Replay & Release ──────────────────────────────────────────

void ExecutionContext::ReplayWrites() {
    for (auto& update : replayParamUpdates_) {
        uint32_t value = 0;
        switch (update.kind) {
            case ReplayParamUpdate::PosOffset: value = replayPosition_; break;
            case ReplayParamUpdate::PastSeq:   value = replayPosition_; break;
            case ReplayParamUpdate::TotalSeq:  value = replayPosition_ + 1; break;
        }
        gpu->writeBufferRaw(update.paramBuf.handle, update.offsetBytes, &value, 4);
    }

    for (auto& tok : capturedTokenIdBufs_) {
        if (!tok.buffer.handle) continue;
        std::vector<int32_t> i32((size_t)tok.nIdx);
        for (int64_t i = 0; i < tok.nIdx; i++)
            i32[(size_t)i] = (int32_t)replayTokenId_;
        gpu->writeBufferRaw(tok.buffer.handle, tok.buffer.offset,
            i32.data(), (size_t)tok.nIdx * 4);
    }
}

void ExecutionContext::ReplayDispatches(bool skipFence) {
    if (capturedFlushes_.empty()) return;
    fastDecodeState_ = FastDecodeState::Replaying;

    bool profiling = gpuProfiler && gpuProfiler->enabled();

    for (auto& flush : capturedFlushes_) {
        if (flush.dispatches.empty() && flush.copies.empty()) continue;

        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

        if (profiling) {
            for (auto& cmd : flush.dispatches) {
                auto [bIdx, eIdx] = gpuProfiler->allocate(cmd.name);
                auto tw = gpuProfiler->makeTimestampWrites(bIdx, eIdx);
                WGPUComputePassDescriptor cpD{};
                cpD.timestampWrites = &tw;
                auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
                wgpuComputePassEncoderSetPipeline(pass, cmd.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, cmd.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, cmd.gx, cmd.gy, cmd.gz);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
            }
        } else if (!flush.dispatches.empty()) {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            for (auto& cmd : flush.dispatches) {
                wgpuComputePassEncoderSetPipeline(pass, cmd.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, cmd.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, cmd.gx, cmd.gy, cmd.gz);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
        for (auto& c : flush.copies)
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                c.src.handle, c.srcOff, c.dst.handle, c.dstOff, c.size);

        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);
    }

    if (!skipFence && readbackSize_ > 0 && readbackSrc_.handle && readbackDst_.handle) {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            readbackSrc_.handle, readbackSrc_.offset,
            readbackDst_.handle, readbackDst_.offset, readbackSize_);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);
        readbackSize_ = 0;
        readbackSrc_ = {}; readbackDst_ = {};
    }
    if (!skipFence) gpu->waitForQueue();
    fastDecodeState_ = FastDecodeState::Off;
}

void ExecutionContext::ReleaseCaptured() {
    for (auto& flush : capturedFlushes_) {
        for (auto& cmd : flush.dispatches) {
            if (cmd.bindGroup) { wgpuBindGroupRelease(cmd.bindGroup); cmd.bindGroup = nullptr; }
        }
    }
    capturedFlushes_.clear();
    capturedWrites_.clear();
    replayParamUpdates_.clear();
    capturedTokenIdBufs_.clear();
}

// ─── Warm Execute Cache ─────────────────────────────────────────────────────

void ExecutionContext::InvalidateWarmCaches() {
    if (tensorPlanValid_) {
        std::set<WGPUBuffer> released;
        for (auto& [name, alloc] : tensorPlan_) {
            if (alloc.buffer.handle && released.find(alloc.buffer.handle) == released.end()) {
                released.insert(alloc.buffer.handle);
                gpu->releaseBuffer(alloc.buffer);
            }
        }
        tensorPlan_.clear();
    }
    tensorPlanValid_ = false;
    shapeCacheValid_ = false;
    shapeCachePopulating_ = false;
    nodeShapeCache_.clear();
}

// ─── GPU Profiling ──────────────────────────────────────────────────────────

void ExecutionContext::enableGpuProfiling() {
    if (!gpu || !gpu->supportsTimestampQuery) {
        fprintf(stderr, "GPU timestamp queries not supported\n");
        return;
    }
    gpuProfiler = new GPUProfiler();
    if (!gpuProfiler->init(gpu->device, gpu->instance, gpu->queue)) {
        fprintf(stderr, "Failed to init GPU profiler\n");
        delete gpuProfiler;
        gpuProfiler = nullptr;
        return;
    }
    auto cal = acquireClockCalibration(gpu->device, gpu->backendType);
    if (cal.valid) {
        clockCalibration = new ClockCalibration(cal);
    }
}

void ExecutionContext::printGpuProfileReport(int nDecodeTokens, double decodeMs,
                                              const std::string& htmlPath) {
    if (!gpuProfiler || !gpuProfiler->enabled() || gpuProfiler->nextIndex == 0) {
        fprintf(stderr, "No GPU profile data\n");
        return;
    }

    {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        gpuProfiler->resolveAndReport(enc);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
    }
    gpu->waitForQueue();

    uint32_t count = gpuProfiler->nextIndex;
    uint64_t readSize = count * 8;
    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(gpuProfiler->readbackBuf, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(gpu->instance, 1, &mw, UINT64_MAX);

    if (ms.status != 1) {
        fprintf(stderr, "Failed to map profiler readback buffer\n");
        return;
    }

    auto ptr = (const uint64_t*)wgpuBufferGetConstMappedRange(
        gpuProfiler->readbackBuf, 0, readSize);

    struct AggEntry { double totalUs = 0; uint32_t count = 0; };
    std::unordered_map<std::string, AggEntry> agg;
    double totalGpuUs = 0;
    for (auto& e : gpuProfiler->entries) {
        uint64_t begin = ptr[e.beginIdx], end = ptr[e.endIdx];
        if (end <= begin || begin == 0) continue;
        double durUs = (double)(end - begin) / 1000.0;
        agg[e.name].totalUs += durUs;
        agg[e.name].count++;
        totalGpuUs += durUs;
    }

    std::vector<std::pair<std::string, AggEntry>> sorted(agg.begin(), agg.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.second.totalUs > b.second.totalUs; });

    fprintf(stderr, "\n--- GPU Profile (hardware timestamps, %d dispatches) ---\n",
            (int)gpuProfiler->entries.size());
    fprintf(stderr, "%-25s %10s %6s %10s %6s\n",
            "Kernel", "Total(ms)", "Count", "Avg(us)", "%%");
    fprintf(stderr, "%-25s %10s %6s %10s %6s\n",
            "-------------------------", "----------", "------", "----------", "------");
    for (auto& [name, e] : sorted) {
        double totalMs = e.totalUs / 1000.0;
        double avgUs = e.totalUs / e.count;
        double pct = totalGpuUs > 0 ? e.totalUs / totalGpuUs * 100.0 : 0;
        fprintf(stderr, "%-25s %10.2f %6u %10.1f %5.1f%%\n",
                name.c_str(), totalMs, e.count, avgUs, pct);
    }
    double totalGpuMs = totalGpuUs / 1000.0;
    double cpuMs = decodeMs / std::max(1, nDecodeTokens);
    fprintf(stderr, "%-25s %10.2f\n", "GPU TOTAL", totalGpuMs);
    fprintf(stderr, "\nGPU HW time: %.1fms/tok   CPU wall time: %.1fms/tok   Bubble: %.0f%%\n",
            totalGpuMs / std::max(1, nDecodeTokens), cpuMs,
            cpuMs > 0 ? (1.0 - totalGpuMs / std::max(1, nDecodeTokens) / cpuMs) * 100 : 0);

    generateProfileHTML(*gpu, *gpuProfiler, clockCalibration, ptr,
                        nDecodeTokens, 0, 0, decodeMs, htmlPath);
    fprintf(stderr, "Profile HTML: %s\n", htmlPath.c_str());

    wgpuBufferUnmap(gpuProfiler->readbackBuf);
    gpuProfiler->destroy();
    delete gpuProfiler;
    gpuProfiler = nullptr;
    if (clockCalibration) { delete clockCalibration; clockCalibration = nullptr; }
}

// ─── OpContext forwarding to GraphExecutor ───────────────────────────────────

GPUContext* OpContext::getGpu() const {
    return graph.gpu;
}

const CompiledPipeline& OpContext::GetPipeline(const std::string& name,
                                                const std::string& wgsl,
                                                uint32_t numBindings) {
    return graph.GetPipeline(name, wgsl, numBindings);
}

WGPUBindGroup OpContext::MakeBindGroup(const CompiledPipeline& pl,
                                        const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings) {
    return graph.MakeBindGroup(pl, bindings, &exec);
}

GpuTensor OpContext::AllocTensor(std::vector<int64_t> shape, TensorDtype dtype) {
    return graph.AllocTensor(std::move(shape), dtype);
}

GpuTensor OpContext::AllocCpuTensor(const std::vector<int64_t>& shape, TensorDtype dtype,
                                     const void* data, size_t bytes) {
    return graph.AllocCpuTensor(shape, dtype, data, bytes);
}

void OpContext::EnsureGpu(GpuTensor& t) {
    graph.EnsureGpu(t);
}

const OnnxInitData* OpContext::GetInitData(const std::string& name) const {
    return graph.GetInitData(name);
}

bool OpContext::IsInitializer(const std::string& name) const {
    return graph.IsInitializer(name);
}

GpuTensor* OpContext::GetTensor(const std::string& name) {
    // Per-session store first, then shared weights
    auto it = exec.tensorStore_.find(name);
    if (it != exec.tensorStore_.end()) return &it->second;
    return graph.GetWeightTensor(name);
}

void OpContext::Submit(const std::vector<Dispatch>& dispatches) {
    graph.Submit(dispatches);
}

void OpContext::SubmitAsync(const std::vector<Dispatch>& dispatches) {
    graph.SubmitAsync(dispatches);
}

void OpContext::Sync() {
    graph.Sync();
}
