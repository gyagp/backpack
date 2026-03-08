#pragma once
/**
 * gpu_context.h — Common WebGPU runtime for all models.
 *
 * Provides GPU initialization, buffer management, pipeline caching,
 * and dispatch encoding. Model-agnostic: the model-specific decode
 * plan is described in manifest.json and interpreted by the engine.
 */

#include <webgpu/webgpu.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Buffer usage flags (stable WebGPU constants)
constexpr uint64_t BUF_STORAGE   = 0x0080;
constexpr uint64_t BUF_COPY_SRC  = 0x0004;
constexpr uint64_t BUF_COPY_DST  = 0x0008;
constexpr uint64_t BUF_MAP_READ  = 0x0001;
constexpr uint64_t BUF_UNIFORM   = 0x0040;
constexpr uint64_t BUF_DEFAULT   = BUF_STORAGE | BUF_COPY_SRC | BUF_COPY_DST;

/// Compiled compute pipeline with associated layouts.
struct CompiledPipeline {
    WGPUShaderModule    shader       = nullptr;
    WGPUComputePipeline pipeline     = nullptr;
    WGPUBindGroupLayout bgLayout     = nullptr;
    WGPUPipelineLayout  pplLayout    = nullptr;
    uint32_t            numBindings  = 0;
};

/// A single dispatch operation: pipeline + bind group + grid + name.
struct Dispatch {
    WGPUComputePipeline pipeline  = nullptr;
    WGPUBindGroup       bindGroup = nullptr;
    uint32_t gx = 1, gy = 1, gz = 1;
    std::string name;  // for profiling
};

/// GPU buffer with metadata.
struct GPUBuffer {
    WGPUBuffer handle = nullptr;
    uint64_t   size   = 0;
};

/// GPU profiler using WebGPU timestamp queries.
struct GPUProfiler {
    WGPUDevice   device   = nullptr;
    WGPUInstance instance = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPUQuerySet querySet = nullptr;
    WGPUBuffer   resolveBuf = nullptr;
    WGPUBuffer   readbackBuf = nullptr;
    uint32_t     nextIndex = 0;
    static constexpr uint32_t MAX_TIMESTAMPS = 16384;

    struct Entry {
        std::string name;
        uint32_t beginIdx, endIdx;
    };
    std::vector<Entry> entries;

    bool init(WGPUDevice dev, WGPUInstance inst, WGPUQueue q);
    void destroy();

    /// Allocate a timestamp pair. Returns (beginIdx, endIdx).
    std::pair<uint32_t, uint32_t> allocate(const std::string& name);

    /// Fill a WGPUPassTimestampWrites struct for a dispatch.
    WGPUPassTimestampWrites makeTimestampWrites(uint32_t beginIdx, uint32_t endIdx);

    /// Resolve timestamps after GPU work, read back, and print report.
    void resolveAndReport(WGPUCommandEncoder enc);

    bool enabled() const { return querySet != nullptr; }
};

/// Model-agnostic WebGPU context.
struct GPUContext {
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPUBackendType backendType = WGPUBackendType_Vulkan;

    // --- Lifecycle ---
    bool init(WGPUBackendType backend = WGPUBackendType_Vulkan);
    void destroy();

    // --- Buffers ---
    GPUBuffer createBuffer(const std::string& name, uint64_t size,
                           uint64_t usage = BUF_DEFAULT,
                           bool mappedAtCreation = false);
    GPUBuffer getBuffer(const std::string& name) const;
    void      writeBuffer(GPUBuffer buf, const void* data, uint64_t size,
                          uint64_t offset = 0);

    // --- Pipelines ---
    const CompiledPipeline& getOrCreatePipeline(
        const std::string& name, const std::string& wgsl,
        uint32_t numBindings);

    // --- Bind groups ---
    /// Create a bind group from a list of (binding_index, GPUBuffer) pairs.
    WGPUBindGroup createBindGroup(
        const CompiledPipeline& pipeline,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& entries);

    // --- Dispatch ---
    /// Submit a batch of dispatches in a single encoder + single submit.
    /// Returns immediately; call waitForQueue() to synchronize.
    void submitDispatches(const std::vector<Dispatch>& dispatches);

    /// Submit dispatches without readback (fire-and-forget).
    void submitOnly(const std::vector<Dispatch>& dispatches,
                    bool singlePass = true);

    /// Submit dispatches, copy result buffer, and synchronize.
    /// Returns a numpy-like vector of the readback data.
    /// passPerDispatch: if true, each dispatch gets its own compute pass
    ///                  (needed for correctness on some backends).
    std::vector<uint8_t> submitAndReadback(
        const std::vector<Dispatch>& dispatches,
        GPUBuffer src, uint64_t readSize,
        bool passPerDispatch = true);

    /// Read back buffer contents (no dispatches, just copy + map).
    std::vector<uint8_t> readBuffer(GPUBuffer src, uint64_t readSize);

    /// Submit dispatches with profiling: each gets its own timestamped pass.
    std::vector<uint8_t> submitAndReadbackProfiled(
        const std::vector<Dispatch>& dispatches,
        GPUBuffer src, uint64_t readSize,
        GPUProfiler& profiler);

    /// Block until all submitted GPU work completes.
    void waitForQueue();

private:
    std::unordered_map<std::string, CompiledPipeline> pipelines_;
    std::unordered_map<std::string, GPUBuffer>        buffers_;

    // Readback buffer (reused across calls)
    WGPUBuffer readbackBuf_     = nullptr;
    uint64_t   readbackBufSize_ = 0;

    WGPUBuffer getOrCreateReadbackBuf(uint64_t size);
};
