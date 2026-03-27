#pragma once
/**
 * graph_executor.h — General ONNX graph executor on WebGPU.
 *
 * Walks an ONNX graph in topological order, allocating GPU buffers for
 * intermediates and dispatching WGSL compute kernels for each op.
 *
 * Used by bp::Session::Run() to execute arbitrary ONNX models.
 */

#include "gpu_context.h"
#include <cstdint>
#include <functional>
#include <map>
#include <set>
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
        switch (dtype) {
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

struct OnnxGraph {
    std::vector<OnnxGraphNode> nodes;
    std::vector<OnnxGraphInput> inputs;
    std::vector<OnnxGraphInput> outputs;

    // Initializer data: name → raw bytes on CPU (from ONNX protobuf)
    struct InitData {
        const uint8_t* data = nullptr;
        size_t size = 0;
        TensorDtype dtype = TensorDtype::Float32;
        std::vector<int64_t> shape;
    };
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

class GraphExecutor;  // forward

/// Signature for an op implementation:
///   Reads input tensors from the tensor store, writes outputs.
using OpDispatchFn = std::function<void(
    GraphExecutor& executor,
    const OnnxGraphNode& node,
    const std::vector<GpuTensor*>& inputs,
    std::vector<GpuTensor*>& outputs)>;

// ─── Graph Executor ──────────────────────────────────────────────────────────

class GraphExecutor {
public:
    GPUContext* gpu = nullptr;

    ~GraphExecutor() {
        if (gpu) {
            // tensorStore_ may already be cleared by Execute().
            // Release any remaining buffers (from models that were loaded but not executed).
            if (!tensorStore_.empty()) {
                gpu->waitForQueue();
                std::set<WGPUBuffer> released;
                for (auto& [name, tensor] : tensorStore_) {
                    if (tensor.buffer.handle && released.find(tensor.buffer.handle) == released.end()) {
                        released.insert(tensor.buffer.handle);
                        gpu->releaseBuffer(tensor.buffer);
                    }
                    tensor.buffer = {nullptr, 0};
                }
                tensorStore_.clear();
            }
            gpu->flushBufferPool();
        }
    }

    /// Load and parse an ONNX model. Uploads initializer weights to GPU.
    bool Load(GPUContext& gpuCtx, const std::string& onnxPath);

    /// Execute the graph with bound inputs. Writes to bound outputs.
    void Execute(
        const std::unordered_map<std::string, GpuTensor*>& inputs,
        std::unordered_map<std::string, GpuTensor*>& outputs);

    /// Enable per-op profiling. Call before Execute(). Results printed to stderr.
    bool profilingEnabled = false;

    /// Get a reusable param buffer (16/32/48 bytes). Avoids per-dispatch createBuffer.
    /// The buffer is valid until the next Execute() call.
    GPUBuffer getParamBuffer(uint32_t sizeBytes);

    /// GPU hardware timestamp profiler (owned, optional).
    /// Call enableGpuProfiling() to activate. After Execute(), call
    /// printGpuProfileReport() to read timestamps and generate HTML.
    GPUProfiler* gpuProfiler = nullptr;
    struct ClockCalibration* clockCalibration = nullptr;

    /// Enable GPU hardware timestamp profiling. Creates query sets.
    void enableGpuProfiling();

    /// Read back GPU timestamps, print report, generate HTML timeline.
    /// Call after all profiled Execute() runs are done.
    void printGpuProfileReport(int nDecodeTokens, double decodeMs,
                               const std::string& htmlPath = "profile.html");

    /// Get graph metadata.
    const OnnxGraph& GetGraph() const { return graph_; }

    /// Allocate a GPU tensor.
    GpuTensor AllocTensor(std::vector<int64_t> shape, TensorDtype dtype);

    /// Create a CPU-only tensor (no GPU buffer). Upload to GPU lazily.
    GpuTensor AllocCpuTensor(const std::vector<int64_t>& shape, TensorDtype dtype,
                              const void* data, size_t bytes);

    /// Ensure a tensor has a GPU buffer (uploads CPU data if needed).
    void EnsureGpu(GpuTensor& t);

    /// GPU-side fp16→f32 cast. Dispatches a compute shader — no CPU readback.
    /// Replaces the tensor in-place with a new f32 GPU tensor.
    /// Returns true if the tensor is now f32 on GPU.
    bool EnsureFloat32(GpuTensor& t);

    /// Get or create a WGSL compute pipeline.
    const CompiledPipeline& GetPipeline(const std::string& name,
                                         const std::string& wgsl,
                                         uint32_t numBindings);

    /// Get or create a pipeline from a generator function.
    /// The generator is only called on first use (cache miss).
    /// Use for template-generated shaders to avoid per-dispatch string work.
    ///   auto& pl = ex.GetPipelineT("binary_f16", 4, []() {
    ///       return instantiateTemplate(WGSL_BINARY_ELEMENTWISE_T, TensorDtype::Float16);
    ///   });
    template<typename F>
    const CompiledPipeline& GetPipelineT(const std::string& name,
                                          uint32_t numBindings,
                                          F&& generator) {
        // Check cache first (fast path — no string generation)
        if (gpu->hasPipeline(name)) {
            return gpu->getOrCreatePipeline(name, "", numBindings);
        }
        // Cache miss — generate shader once
        std::string wgsl = generator();
        return gpu->getOrCreatePipeline(name, wgsl, numBindings);
    }

    /// Create a bind group.
    WGPUBindGroup MakeBindGroup(const CompiledPipeline& pl,
                                 const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);

    /// Detect fuseable patterns in the graph. Called once after Load().
    /// Populates fusedGroups_ with fusion descriptors.
    void DetectFusions();

    /// Submit dispatches and wait.
    void Submit(const std::vector<Dispatch>& dispatches);

    /// Flush all pending GPU work (dispatches + copies) and wait.
    /// Must be called before any GPU readback to ensure data is written.
    void FlushPendingWork();

    /// Submit dispatches (fire and forget, no sync).
    void SubmitAsync(const std::vector<Dispatch>& dispatches);

    /// Queue a GPU buffer copy (batched with dispatches).
    void QueueCopy(GPUBuffer src, uint64_t srcOffset,
                    GPUBuffer dst, uint64_t dstOffset, uint64_t size);

    /// Wait for all GPU work to complete.
    void Sync();

private:
    OnnxGraph graph_;

public:
    /// Access initializer data on CPU (avoids GPU readback for metadata ops).
    const OnnxGraph::InitData* GetInitData(const std::string& name) const {
        auto it = graph_.initializers.find(name);
        return (it != graph_.initializers.end()) ? &it->second : nullptr;
    }

    /// Check if a tensor name is an initializer (constant weight).
    bool IsInitializer(const std::string& name) const {
        return graph_.initializers.count(name) > 0;
    }

private:
    // Tensor store: all intermediate and initializer tensors by name
    // Using std::map for pointer stability (unordered_map rehashes invalidate references)
    std::map<std::string, GpuTensor> tensorStore_;

    // Names of tensors loaded during LoadGraph (initializers + pre-store constants).
    // These persist across Execute() calls; everything else is cleared.
    std::set<std::string> persistentTensors_;

    // Cached topo sort order (reused across Execute calls for the same graph)
    std::vector<size_t> cachedExecOrder_;

    // Fused op groups detected at graph analysis time.
    // Key: first node index in the group. Value: fusion descriptor.
    std::unordered_map<size_t, FusedGroup> fusedGroups_;
    // Set of node indices that are part of a fusion (skip individual dispatch)
    std::set<size_t> fusedNodeIndices_;

public:
    // Batched dispatches for the current execution (public for op access)
    std::vector<Dispatch> pendingDispatches_;
    struct PendingCopy { GPUBuffer src; uint64_t srcOff; GPUBuffer dst; uint64_t dstOff; uint64_t size; };
    std::vector<PendingCopy> pendingCopies_;

    // Deferred int readbacks — flushed at periodic sync or when needed
    std::vector<GpuTensor*> pendingIntReadbacks_;

    // Deferred buffer releases — flushed at periodic GPU sync points
    std::vector<std::string> pendingReleases_;

    // Param buffer pool: pre-allocated buffers for dispatch params.
    // Indexed by size bucket: [0]=16B, [1]=32B, [2]=48B, [3]=64B
    static constexpr int PARAM_POOL_BUCKETS = 4;
    static constexpr int PARAM_POOL_SIZE = 512;  // max buffers per bucket
    struct ParamPool {
        std::vector<GPUBuffer> buffers;
        int nextIdx = 0;
    };
    ParamPool paramPool_[PARAM_POOL_BUCKETS];

    // Profiling accumulators (per op-type, in ms)
    std::unordered_map<std::string, double> profileData_;
    std::unordered_map<std::string, int> profileCounts_;
    int flushCount_ = 0;  // GPU sync count per Execute
    int intReadbackSyncs_ = 0;  // syncs from int readback only
    std::unordered_map<std::string, int> flushSources_;  // sync count by op

private:

    // Op registry
    static std::unordered_map<std::string, OpDispatchFn>& GetOpRegistry();

    // External data (kept alive for pointer stability)
    std::vector<uint8_t> onnxData_;
    std::vector<uint8_t> externalData_;
    // Multi-file external data: filename → data
    std::unordered_map<std::string, std::vector<uint8_t>> externalDataFiles_;

public:
    /// Register an op implementation. Called at static init time.
    static void RegisterOp(const std::string& opType, OpDispatchFn fn);
};

// ─── Op Registration Macro ───────────────────────────────────────────────────

#define REGISTER_OP(name, fn) \
    static struct _RegOp_##name { \
        _RegOp_##name() { GraphExecutor::RegisterOp(#name, fn); } \
    } _regOp_##name;
