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

    bool IsValid() const { return (buffer.handle != nullptr || isCpuOnly) && !shape.empty(); }
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

    /// Load and parse an ONNX model. Uploads initializer weights to GPU.
    bool Load(GPUContext& gpuCtx, const std::string& onnxPath);

    /// Execute the graph with bound inputs. Writes to bound outputs.
    void Execute(
        const std::unordered_map<std::string, GpuTensor*>& inputs,
        std::unordered_map<std::string, GpuTensor*>& outputs);

    /// Get graph metadata.
    const OnnxGraph& GetGraph() const { return graph_; }

    /// Allocate a GPU tensor.
    GpuTensor AllocTensor(const std::vector<int64_t>& shape, TensorDtype dtype);

    /// Create a CPU-only tensor (no GPU buffer). Upload to GPU lazily.
    GpuTensor AllocCpuTensor(const std::vector<int64_t>& shape, TensorDtype dtype,
                              const void* data, size_t bytes);

    /// Ensure a tensor has a GPU buffer (uploads CPU data if needed).
    void EnsureGpu(GpuTensor& t);

    /// Get or create a WGSL compute pipeline.
    const CompiledPipeline& GetPipeline(const std::string& name,
                                         const std::string& wgsl,
                                         uint32_t numBindings);

    /// Create a bind group.
    WGPUBindGroup MakeBindGroup(const CompiledPipeline& pl,
                                 const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings);

    /// Submit dispatches and wait.
    void Submit(const std::vector<Dispatch>& dispatches);

    /// Submit dispatches (fire and forget, no sync).
    void SubmitAsync(const std::vector<Dispatch>& dispatches);

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
    std::unordered_map<std::string, GpuTensor> tensorStore_;

public:
    // Batched dispatches for the current execution (public for op access)
    std::vector<Dispatch> pendingDispatches_;

private:

    // Op registry
    static std::unordered_map<std::string, OpDispatchFn>& GetOpRegistry();

    // External data (kept alive for pointer stability)
    std::vector<uint8_t> onnxData_;
    std::vector<uint8_t> externalData_;

public:
    /// Register an op implementation. Called at static init time.
    static void RegisterOp(const std::string& opType, OpDispatchFn fn);
};

// ─── Op Registration Macro ───────────────────────────────────────────────────

#define REGISTER_OP(name, fn) \
    static struct _RegOp_##name { \
        _RegOp_##name() { GraphExecutor::RegisterOp(#name, fn); } \
    } _regOp_##name;
