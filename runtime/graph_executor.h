#pragma once
/**
 * graph_executor.h — General ONNX graph executor on WebGPU.
 *
 * Walks an ONNX graph in topological order, allocating GPU buffers for
 * intermediates and dispatching WGSL compute kernels for each op.
 *
 * Architecture (concurrent sessions):
 *   GraphExecutor  — shared immutable state (graph, weights, pipelines)
 *   ExecutionContext — per-session mutable state (intermediates, caches, fast decode)
 *   OpContext       — facade passed to op implementations
 *
 * Used by bp::Session::Run() to execute arbitrary ONNX models.
 */

#include "gpu_context.h"
#include "execution_context.h"
#include "mapped_file.h"
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ─── GPU Tensor ──────────────────────────────────────────────────────────────
/// A tensor living on the GPU with shape + type metadata.

enum class TensorDtype { Float32, Float16, Int32, Int64, UInt8, Int8, Bool };

struct GpuTensor {
    GPUBuffer buffer;
    std::vector<int64_t> shape;
    TensorDtype dtype = TensorDtype::Float32;

    // CPU-side data (for small metadata tensors that don't need GPU)
    std::vector<uint8_t> cpuData;
    bool isCpuOnly = false;

    int64_t ElementCount() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    size_t ByteSize() const {
        return (size_t)ElementCount() * DtypeSize();
    }

    size_t DtypeSize() const {
        return DtypeSizeOf(dtype);
    }

    static size_t DtypeSizeOf(TensorDtype dt) {
        switch (dt) {
            case TensorDtype::Float32: case TensorDtype::Int32: return 4;
            case TensorDtype::Float16: return 2;
            case TensorDtype::Int64: return 8;
            case TensorDtype::UInt8: case TensorDtype::Int8: case TensorDtype::Bool: return 1;
        }
        return 4;
    }

    bool IsValid() const { return (buffer.handle != nullptr || isCpuOnly); }
};

// ─── ONNX Node ───────────────────────────────────────────────────────────────

struct OnnxGraphNode {
    std::string opType;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // Attributes (simple key-value)
    std::unordered_map<std::string, int64_t> attrInts;
    std::unordered_map<std::string, float> attrFloats;
    std::unordered_map<std::string, std::string> attrStrings;
    std::unordered_map<std::string, std::vector<int64_t>> attrIntLists;

    int64_t GetInt(const std::string& key, int64_t def = 0) const {
        auto it = attrInts.find(key);
        return (it != attrInts.end()) ? it->second : def;
    }
    float GetFloat(const std::string& key, float def = 0.0f) const {
        auto it = attrFloats.find(key);
        return (it != attrFloats.end()) ? it->second : def;
    }
};

// ─── ONNX Graph ──────────────────────────────────────────────────────────────

struct OnnxGraphInput {
    std::string name;
    TensorDtype dtype;
    std::vector<int64_t> shape;  // may contain -1 for dynamic dims
};

// Initializer data: raw bytes on CPU (from ONNX protobuf)
struct OnnxInitData {
    const uint8_t* data = nullptr;
    size_t size = 0;
    TensorDtype dtype = TensorDtype::Float32;
    std::vector<int64_t> shape;
};

struct OnnxGraph {
    std::vector<OnnxGraphNode> nodes;
    std::vector<OnnxGraphInput> inputs;
    std::vector<OnnxGraphInput> outputs;

    using InitData = OnnxInitData;
    std::unordered_map<std::string, InitData> initializers;
};

// ─── Op Fusion ───────────────────────────────────────────────────────────────

/// Describes a fused group of ONNX nodes replaced by a single GPU dispatch.
/// The fusion pass detects patterns in the graph and generates a combined
/// shader at pipeline creation time (one-time cost).
struct FusedGroup {
    /// Indices into OnnxGraph::nodes for the fused nodes (in execution order).
    std::vector<size_t> nodeIndices;

    /// The fused op dispatches using this pipeline name (cached after first use).
    std::string pipelineName;

    /// Number of shader bindings for the fused kernel.
    uint32_t numBindings = 0;

    /// Input tensor names consumed from outside the group.
    std::vector<std::string> externalInputs;

    /// Output tensor name produced by the group (last node's output).
    std::string outputName;

    /// The shader generator — called once per dtype to produce the fused WGSL.
    /// Captures the op types, opcodes needed for generation.
    std::function<std::string(TensorDtype)> shaderGenerator;
};

// ─── Op Dispatch Function ────────────────────────────────────────────────────

/// Signature for an op implementation:
///   Reads input tensors from the tensor store, writes outputs.
///   OpContext provides access to both shared graph state and per-session
///   execution state.
using OpDispatchFn = std::function<void(
    OpContext& ctx,
    const OnnxGraphNode& node,
    const std::vector<GpuTensor*>& inputs,
    std::vector<GpuTensor*>& outputs)>;

// ─── Graph Executor ──────────────────────────────────────────────────────────
// Shared immutable state: graph definition, weights, compiled pipelines.
// Per-session mutable state lives in ExecutionContext.

class GraphExecutor {
public:
    GPUContext* gpu = nullptr;

    ~GraphExecutor();

    /// Load and parse an ONNX model. Uploads initializer weights to GPU.
    bool Load(GPUContext& gpuCtx, const std::string& onnxPath);

    /// Execute the graph with a per-session ExecutionContext.
    void Execute(
        ExecutionContext& ctx,
        const std::unordered_map<std::string, GpuTensor*>& inputs,
        std::unordered_map<std::string, GpuTensor*>& outputs);

    /// Backward-compat: Execute using the default (internal) ExecutionContext.
    void Execute(
        const std::unordered_map<std::string, GpuTensor*>& inputs,
        std::unordered_map<std::string, GpuTensor*>& outputs) {
        Execute(defaultCtx_, inputs, outputs);
    }

    /// Get graph metadata.
    const OnnxGraph& GetGraph() const { return graph_; }

    /// Pre-compile all GPU pipelines needed by the loaded graph.
    size_t WarmupAllPipelines();

    /// Allocate a GPU tensor.
    GpuTensor AllocTensor(std::vector<int64_t> shape, TensorDtype dtype);

    /// Create a CPU-only tensor (no GPU buffer). Upload to GPU lazily.
    GpuTensor AllocCpuTensor(const std::vector<int64_t>& shape, TensorDtype dtype,
                              const void* data, size_t bytes);

    /// Ensure a tensor has a GPU buffer (uploads CPU data if needed).
    void EnsureGpu(GpuTensor& t);

    /// Get or create a WGSL compute pipeline.
    const CompiledPipeline& GetPipeline(const std::string& name,
                                         const std::string& wgsl,
                                         uint32_t numBindings);

    /// Get or create a pipeline from a generator function.
    template<typename F>
    const CompiledPipeline& GetPipelineT(const std::string& name,
                                          uint32_t numBindings,
                                          F&& generator) {
        if (auto* p = gpu->findPipeline(name)) return *p;
        std::string wgsl = generator();
        return gpu->getOrCreatePipeline(name, wgsl, numBindings);
    }

    /// Create a bind group. Optionally tracks bindings for fast decode capture.
    WGPUBindGroup MakeBindGroup(const CompiledPipeline& pl,
                                 const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings,
                                 ExecutionContext* captureCtx = nullptr);

    /// Detect fuseable patterns in the graph. Called once after Load().
    void DetectFusions();

    /// Submit dispatches and wait.
    void Submit(const std::vector<Dispatch>& dispatches);

    /// Submit dispatches (fire and forget, no sync).
    void SubmitAsync(const std::vector<Dispatch>& dispatches);

    /// Wait for all GPU work to complete.
    void Sync();

    /// Access initializer data on CPU (avoids GPU readback for metadata ops).
    const OnnxGraph::InitData* GetInitData(const std::string& name) const {
        auto it = graph_.initializers.find(name);
        return (it != graph_.initializers.end()) ? &it->second : nullptr;
    }

    /// Check if a tensor name is an initializer (constant weight).
    bool IsInitializer(const std::string& name) const {
        return graph_.initializers.count(name) > 0;
    }

    /// Access a weight tensor by name (persistent tensors only).
    GpuTensor* GetWeightTensor(const std::string& name) {
        auto it = weightStore_.find(name);
        return (it != weightStore_.end()) ? &it->second : nullptr;
    }

    /// Register an op implementation. Called at static init time.
    static void RegisterOp(const std::string& opType, OpDispatchFn fn);

    // ─── Backward Compatibility ─────────────────────────────────────────
    // These methods forward to defaultCtx_ so existing code that accesses
    // the GraphExecutor directly (OnnxLlmContext, LlmContext, tests)
    // continues to work without changes.

    /// Access the default (internal) ExecutionContext.
    ExecutionContext& DefaultCtx() { return defaultCtx_; }

    /// Access a tensor by name: defaultCtx_.tensorStore_ first, then weightStore_.
    GpuTensor* GetTensor(const std::string& name) {
        auto it = defaultCtx_.tensorStore_.find(name);
        if (it != defaultCtx_.tensorStore_.end()) return &it->second;
        auto it2 = weightStore_.find(name);
        return (it2 != weightStore_.end()) ? &it2->second : nullptr;
    }

    // Per-session state forwarding to defaultCtx_ (backward compat)
    bool profilingEnabled = false;  // Forwarded to defaultCtx_ in Execute()

    GPUBuffer getParamBuffer(uint32_t sizeBytes) { return defaultCtx_.getParamBuffer(sizeBytes); }
    void FlushPendingWork() { defaultCtx_.FlushPendingWork(); }
    void SubmitPending() { defaultCtx_.SubmitPending(); }
    void QueueCopy(GPUBuffer src, uint64_t srcOffset, GPUBuffer dst, uint64_t dstOffset, uint64_t size) {
        defaultCtx_.QueueCopy(src, srcOffset, dst, dstOffset, size);
    }
    void QueueDispatch(WGPUComputePipeline pipeline, WGPUBindGroup bg,
                       uint32_t gx, uint32_t gy, uint32_t gz, const char* name) {
        defaultCtx_.QueueDispatch(pipeline, bg, gx, gy, gz, name);
    }
    void RequestReadback(GPUBuffer src, GPUBuffer dst, uint64_t size) {
        defaultCtx_.RequestReadback(src, dst, size);
    }

    // Fast decode forwarding
    ExecutionContext::FastDecodeState fastDecodeState_ = ExecutionContext::FastDecodeState::Off;  // alias

    void CaptureBegin() { defaultCtx_.CaptureBegin(); fastDecodeState_ = defaultCtx_.fastDecodeState_; }
    void CaptureEnd() { defaultCtx_.CaptureEnd(); fastDecodeState_ = defaultCtx_.fastDecodeState_; }
    void Replay() { defaultCtx_.Replay(); }
    void ReplayWrites() { defaultCtx_.ReplayWrites(); }
    void ReplayDispatches(bool skipFence = false) { defaultCtx_.ReplayDispatches(skipFence); }
    void ReleaseCaptured() { defaultCtx_.ReleaseCaptured(); }
    void InvalidateWarmCaches() { defaultCtx_.InvalidateWarmCaches(); }
    void RegisterReplayParam(GPUBuffer buf, uint32_t offset, ExecutionContext::ReplayParamUpdate::Kind kind) {
        defaultCtx_.RegisterReplayParam(buf, offset, kind);
    }
    void RecordWrite(GPUBuffer buf, const void* data, uint64_t size, uint64_t offset = 0) {
        defaultCtx_.RecordWrite(buf, data, size, offset);
    }

    // Expose defaultCtx_ fast decode state for backward compat
    uint32_t& replayPosition() { return defaultCtx_.replayPosition_; }
    int64_t& replayTokenId() { return defaultCtx_.replayTokenId_; }
    uint32_t& capturePosition() { return defaultCtx_.capturePosition_; }
    std::vector<WGPUBuffer>& replaySkipBuffers() { return defaultCtx_.replaySkipBuffers_; }
    std::vector<ExecutionContext::CapturedTokenIdBuf>& capturedTokenIdBufs() { return defaultCtx_.capturedTokenIdBufs_; }
    std::vector<ExecutionContext::ReplayScalarUpdate>& replayScalarUpdates() { return defaultCtx_.replayScalarUpdates_; }
    std::vector<ExecutionContext::CapturedFlush>& capturedFlushes() { return defaultCtx_.capturedFlushes_; }
    std::vector<ExecutionContext::ReplayParamUpdate>& replayParamUpdates() { return defaultCtx_.replayParamUpdates_; }
    std::vector<Dispatch>& pendingDispatches() { return defaultCtx_.pendingDispatches_; }
    std::vector<Dispatch::BindEntry>& lastCapturedBindings() { return defaultCtx_.lastCapturedBindings_; }

    // Readback state forwarding
    GPUBuffer& readbackSrc() { return defaultCtx_.readbackSrc_; }
    GPUBuffer& readbackDst() { return defaultCtx_.readbackDst_; }
    uint64_t& readbackSize() { return defaultCtx_.readbackSize_; }

    // Deferred state forwarding
    std::vector<GpuTensor*>& pendingIntReadbacks() { return defaultCtx_.pendingIntReadbacks_; }
    std::vector<std::string>& pendingReleases() { return defaultCtx_.pendingReleases_; }
    std::vector<GPUBuffer>& deferredBufferReleases() { return defaultCtx_.deferredBufferReleases_; }

    // Profiling forwarding
    GPUProfiler*& gpuProfiler() { return defaultCtx_.gpuProfiler; }
    ClockCalibration*& clockCalibration() { return defaultCtx_.clockCalibration; }
    void enableGpuProfiling() { defaultCtx_.enableGpuProfiling(); }
    void printGpuProfileReport(int nDecodeTokens, double decodeMs,
                               const std::string& htmlPath = "profile.html") {
        defaultCtx_.printGpuProfileReport(nDecodeTokens, decodeMs, htmlPath);
    }

    // Shape cache / tensor plan forwarding
    std::unordered_map<std::string, ExecutionContext::CachedNodeOutput>& nodeShapeCache() { return defaultCtx_.nodeShapeCache_; }
    bool& shapeCacheValid() { return defaultCtx_.shapeCacheValid_; }
    std::unordered_map<std::string, ExecutionContext::TensorAlloc>& tensorPlan() { return defaultCtx_.tensorPlan_; }
    bool& tensorPlanValid() { return defaultCtx_.tensorPlanValid_; }

private:
    OnnxGraph graph_;

    // Weight store: persistent tensors (initializers loaded during Load).
    // Read-only during Execute. Using std::map for pointer stability.
    std::map<std::string, GpuTensor> weightStore_;

    // Names of tensors loaded during LoadGraph (initializers + pre-store constants).
    std::set<std::string> persistentTensors_;

    // Cached topo sort order (reused across Execute calls for the same graph)
    std::vector<size_t> cachedExecOrder_;

    // Fused op groups detected at graph analysis time.
    std::unordered_map<size_t, FusedGroup> fusedGroups_;
    std::set<size_t> fusedNodeIndices_;

    // Default ExecutionContext for backward-compatible single-session use
    ExecutionContext defaultCtx_;

    // Op registry
    static std::unordered_map<std::string, OpDispatchFn>& GetOpRegistry();

    // Memory-mapped data (kept alive for pointer stability)
    MappedFile onnxMapping_;
    MappedFile extDataMapping_;
    std::unordered_map<std::string, MappedFile> extDataMappings_;
};

// ─── Op Registration Macro ───────────────────────────────────────────────────

#define REGISTER_OP(name, fn) \
    static struct _RegOp_##name { \
        _RegOp_##name() { GraphExecutor::RegisterOp(#name, fn); } \
    } _regOp_##name;

// ─── OpContext::GetPipelineT template implementation ─────────────────────────
// Defined here because it needs the full GraphExecutor definition.

template<typename F>
const CompiledPipeline& OpContext::GetPipelineT(const std::string& name,
                                                  uint32_t numBindings,
                                                  F&& generator) {
    return graph.GetPipelineT(name, numBindings, std::forward<F>(generator));
}
