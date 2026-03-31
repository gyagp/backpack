#pragma once
/**
 * execution_context.h — Per-session mutable execution state.
 *
 * ExecutionContext holds all state that changes during Execute() and must
 * be independent per session for concurrent session support:
 *   - Intermediate tensor store
 *   - Param buffer pools
 *   - Pending GPU work (dispatches, copies)
 *   - Fast decode capture/replay state
 *   - Warm execute caches (shape cache, tensor plan)
 *   - Profiling accumulators
 *
 * GraphExecutor retains the shared immutable state (graph definition,
 * weights, compiled pipelines) and delegates per-session work to an
 * ExecutionContext.
 *
 * OpContext is a thin facade passed to op implementations, providing
 * unified access to both the shared GraphExecutor and per-session
 * ExecutionContext.
 */

#include "gpu_context.h"
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
struct GpuTensor;
struct GPUProfiler;
struct ClockCalibration;
class GraphExecutor;
struct OnnxInitData;
enum class TensorDtype;

// ─── ExecutionContext ────────────────────────────────────────────────────────

struct ExecutionContext {
    GPUContext* gpu = nullptr;

    ~ExecutionContext();

    // ─── Intermediate Tensor Store ──────────────────────────────────────
    // Using std::map for pointer stability (unordered_map rehashes invalidate refs)
    std::map<std::string, GpuTensor> tensorStore_;

    // ─── Pending GPU Work ───────────────────────────────────────────────
    std::vector<Dispatch> pendingDispatches_;
    struct PendingCopy { GPUBuffer src; uint64_t srcOff; GPUBuffer dst; uint64_t dstOff; uint64_t size; };
    std::vector<PendingCopy> pendingCopies_;

    /// Flush pending dispatches+copies into a command encoder and submit.
    void flushToEncoder();

    /// Flush all pending GPU work and wait for completion.
    void FlushPendingWork();

    /// Submit all pending dispatches+copies without waiting.
    void SubmitPending();

    /// Queue a GPU buffer copy (batched with dispatches).
    void QueueCopy(GPUBuffer src, uint64_t srcOffset,
                   GPUBuffer dst, uint64_t dstOffset, uint64_t size);

    /// Queue a dispatch. During capture, attaches bind group entries for replay.
    void QueueDispatch(WGPUComputePipeline pipeline, WGPUBindGroup bg,
                       uint32_t gx, uint32_t gy, uint32_t gz, const char* name);

    /// Request a readback copy at the end of the next flushToEncoder.
    GPUBuffer readbackSrc_;
    GPUBuffer readbackDst_;
    uint64_t readbackSize_ = 0;
    void RequestReadback(GPUBuffer src, GPUBuffer dst, uint64_t size) {
        readbackSrc_ = src; readbackDst_ = dst; readbackSize_ = size;
    }

    // ─── Param Buffer Pool ──────────────────────────────────────────────
    static constexpr int PARAM_POOL_BUCKETS = 4;
    static constexpr int PARAM_POOL_SIZE = 512;
    struct ParamPool {
        std::vector<GPUBuffer> buffers;
        int nextIdx = 0;
    };
    ParamPool paramPool_[PARAM_POOL_BUCKETS];

    /// Get a reusable param buffer (16/32/48/64 bytes).
    GPUBuffer getParamBuffer(uint32_t sizeBytes);

    // ─── Fast Decode (Capture + Replay) ─────────────────────────────────

    enum class FastDecodeState { Off, Capturing, Replaying };
    FastDecodeState fastDecodeState_ = FastDecodeState::Off;

    struct CapturedCommand {
        WGPUComputePipeline pipeline;
        WGPUBindGroup bindGroup = nullptr;
        std::vector<Dispatch::BindEntry> bindings;
        uint32_t gx, gy, gz;
        std::string name;
    };
    struct CapturedCopy {
        GPUBuffer src; uint64_t srcOff;
        GPUBuffer dst; uint64_t dstOff;
        uint64_t size;
    };
    struct CapturedFlush {
        std::vector<CapturedCommand> dispatches;
        std::vector<CapturedCopy> copies;
    };
    std::vector<CapturedFlush> capturedFlushes_;

    struct CapturedWrite {
        WGPUBuffer handle;
        uint64_t offset;
        std::vector<uint8_t> data;
    };
    std::vector<CapturedWrite> capturedWrites_;

    struct ReplayParamUpdate {
        GPUBuffer paramBuf;
        uint32_t offsetBytes;
        enum Kind { PosOffset, PastSeq, TotalSeq } kind;
    };
    std::vector<ReplayParamUpdate> replayParamUpdates_;

    void RegisterReplayParam(GPUBuffer buf, uint32_t offset, ReplayParamUpdate::Kind kind) {
        if (fastDecodeState_ == FastDecodeState::Capturing)
            replayParamUpdates_.push_back({buf, offset, kind});
    }

    void CaptureBegin() {
        fastDecodeState_ = FastDecodeState::Capturing;
        capturedFlushes_.clear();
        capturedWrites_.clear();
        replayParamUpdates_.clear();
        replayScalarUpdates_.clear();
        capturedTokenIdBufs_.clear();
    }

    void CaptureEnd() {
        fastDecodeState_ = FastDecodeState::Off;
        gpu->captureWritesCb_ = nullptr;
        gpu->captureWritesCtx_ = nullptr;
    }

    void RecordWrite(GPUBuffer buf, const void* data, uint64_t size, uint64_t offset = 0) {
        if (fastDecodeState_ == FastDecodeState::Capturing) {
            CapturedWrite w;
            w.handle = buf.handle;
            w.offset = offset + buf.offset;
            w.data.resize((size_t)size);
            memcpy(w.data.data(), data, (size_t)size);
            capturedWrites_.push_back(std::move(w));
        }
    }

    uint32_t replayPosition_ = 0;
    int64_t replayTokenId_ = 0;

    struct CapturedTokenIdBuf {
        GPUBuffer buffer;
        int64_t nIdx;
    };
    std::vector<CapturedTokenIdBuf> capturedTokenIdBufs_;

    uint32_t capturePosition_ = 0;
    std::vector<WGPUBuffer> replaySkipBuffers_;

    struct ReplayScalarUpdate {
        WGPUBuffer handle;
        uint64_t offset;
        uint64_t size;
        int64_t captureValue;
        int64_t capturePos;
    };
    std::vector<ReplayScalarUpdate> replayScalarUpdates_;

    void Replay() { ReplayWrites(); ReplayDispatches(); }
    void ReplayWrites();
    void ReplayDispatches(bool skipFence = false);
    void ReleaseCaptured();

    // Temporary storage for bind group entries during capture
    std::vector<Dispatch::BindEntry> lastCapturedBindings_;

    // ─── Shape Cache (Warm Execute) ─────────────────────────────────────
    struct CachedNodeOutput {
        std::vector<int64_t> shape;
        TensorDtype dtype;
        std::vector<uint8_t> cpuData;
        bool hasCpuData = false;
    };
    std::unordered_map<std::string, CachedNodeOutput> nodeShapeCache_;
    bool shapeCacheValid_ = false;
    bool shapeCachePopulating_ = false;

    // ─── Tensor Plan (Buffer Reuse) ─────────────────────────────────────
    struct TensorAlloc {
        GPUBuffer buffer;
        uint64_t size;
        std::vector<int64_t> shape;
        TensorDtype dtype;
    };
    std::unordered_map<std::string, TensorAlloc> tensorPlan_;
    bool tensorPlanValid_ = false;

    /// Invalidate warm execute caches.
    void InvalidateWarmCaches();

    // ─── Deferred Cleanup ───────────────────────────────────────────────
    std::vector<GpuTensor*> pendingIntReadbacks_;
    std::vector<std::string> pendingReleases_;
    std::vector<GPUBuffer> deferredBufferReleases_;

    // ─── Profiling ──────────────────────────────────────────────────────
    bool profilingEnabled = false;
    GPUProfiler* gpuProfiler = nullptr;
    ClockCalibration* clockCalibration = nullptr;

    std::unordered_map<std::string, double> profileData_;
    std::unordered_map<std::string, int> profileCounts_;
    int flushCount_ = 0;
    int intReadbackSyncs_ = 0;
    std::unordered_map<std::string, int> flushSources_;

    void enableGpuProfiling();
    void printGpuProfileReport(int nDecodeTokens, double decodeMs,
                               const std::string& htmlPath = "profile.html");
};

// ─── OpContext ───────────────────────────────────────────────────────────────
// Facade passed to op implementations, providing unified access to both
// the shared GraphExecutor and per-session ExecutionContext.

struct OnnxGraphNode;  // forward
struct CompiledPipeline;

struct OpContext {
    GraphExecutor& graph;
    ExecutionContext& exec;

    // GPU context access (method instead of field to match existing ex.gpu-> pattern
    // when accessed as ex.gpu->...)
    GPUContext* getGpu() const;

    // Forwarding to ExecutionContext (per-session mutable state)
    GPUBuffer getParamBuffer(uint32_t size) { return exec.getParamBuffer(size); }

    void QueueDispatch(WGPUComputePipeline pipeline, WGPUBindGroup bg,
                       uint32_t gx, uint32_t gy, uint32_t gz, const char* name) {
        exec.QueueDispatch(pipeline, bg, gx, gy, gz, name);
    }

    void QueueCopy(GPUBuffer src, uint64_t srcOff, GPUBuffer dst, uint64_t dstOff, uint64_t size) {
        exec.QueueCopy(src, srcOff, dst, dstOff, size);
    }

    void FlushPendingWork() { exec.FlushPendingWork(); }
    void SubmitPending() { exec.SubmitPending(); }

    // Direct GPU submission (bypasses pending work queue)
    void Submit(const std::vector<Dispatch>& dispatches);
    void SubmitAsync(const std::vector<Dispatch>& dispatches);
    void Sync();

    void RequestReadback(GPUBuffer src, GPUBuffer dst, uint64_t size) {
        exec.RequestReadback(src, dst, size);
    }

    void RegisterReplayParam(GPUBuffer buf, uint32_t offset,
                             ExecutionContext::ReplayParamUpdate::Kind kind) {
        exec.RegisterReplayParam(buf, offset, kind);
    }

    void RecordWrite(GPUBuffer buf, const void* data, uint64_t size, uint64_t offset = 0) {
        exec.RecordWrite(buf, data, size, offset);
    }

    ExecutionContext::FastDecodeState fastDecodeState() const { return exec.fastDecodeState_; }

    // Forwarding to GraphExecutor (shared immutable state)
    // These are implemented in execution_context.cpp to avoid circular include
    const CompiledPipeline& GetPipeline(const std::string& name,
                                        const std::string& wgsl,
                                        uint32_t numBindings);

    template<typename F>
    const CompiledPipeline& GetPipelineT(const std::string& name,
                                          uint32_t numBindings,
                                          F&& generator);

    WGPUBindGroup MakeBindGroup(const CompiledPipeline& pl,
                                const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);

    GpuTensor AllocTensor(std::vector<int64_t> shape, TensorDtype dtype);

    GpuTensor AllocCpuTensor(const std::vector<int64_t>& shape, TensorDtype dtype,
                             const void* data, size_t bytes);

    void EnsureGpu(GpuTensor& t);

    const OnnxInitData* GetInitData(const std::string& name) const;
    bool IsInitializer(const std::string& name) const;

    /// Look up tensor: exec.tensorStore_ first, then graph.weightStore_
    GpuTensor* GetTensor(const std::string& name);
};
