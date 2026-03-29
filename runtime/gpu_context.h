#pragma once
/**
 * gpu_context.h — Common WebGPU runtime for all models.
 *
 * Provides GPU initialization, buffer management, pipeline caching,
 * and dispatch encoding. Model-agnostic: the model-specific decode
 * plan is built from GGUF metadata at load time.
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
constexpr uint64_t BUF_INDIRECT  = 0x0100;
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
    // Bindings for fast decode capture (rebuild bind groups on replay)
    struct BindEntry { uint32_t idx; WGPUBuffer handle; uint64_t offset; uint64_t size; };
    std::vector<BindEntry> capturedBindings;
};

/// GPU buffer with metadata.
struct GPUBuffer {
    WGPUBuffer handle = nullptr;
    uint64_t   size   = 0;
    uint64_t   offset = 0;   // bind offset (for buffer views/aliases)
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
    std::string adapterName;
    std::string adapterDescription;
    WGPULimits adapterLimits{};
    WGPULimits deviceLimits{};
    bool supportsShaderF16 = false;
    bool supportsSubgroups = false;
    bool supportsSubgroupMatrix = false;
    bool supportsTimestampQuery = false;

    // OOM detection: set by Dawn error callback, checked after buffer creation
    bool lastAllocFailed = false;

    // --- Lifecycle ---
    bool init(WGPUBackendType backend = WGPUBackendType_Vulkan);
    void destroy();

    // --- Buffers ---
    int createBufferCount = 0;  // allocation counter for profiling
    int poolHitCount = 0;  // pool reuse counter
    GPUBuffer createBuffer(const std::string& name, uint64_t size,
                           uint64_t usage = BUF_DEFAULT,
                           bool mappedAtCreation = false);
    GPUBuffer getBuffer(const std::string& name) const;
    void      writeBuffer(GPUBuffer buf, const void* data, uint64_t size,
                          uint64_t offset = 0);
    /// Write raw bytes to a WGPUBuffer handle (no offset adjustment for views).
    void      writeBufferRaw(WGPUBuffer handle, uint64_t offset, const void* data, uint64_t size);

    // Fast decode: callback to record writeBuffer calls during capture stage
    using CaptureWritesCb = void(*)(WGPUBuffer handle, uint64_t offset,
                                     const void* data, uint64_t size, void* ctx);
    CaptureWritesCb captureWritesCb_ = nullptr;
    void* captureWritesCtx_ = nullptr;
    /// Return a buffer to the pool for reuse.
    void      releaseBuffer(GPUBuffer buf);
    /// Flush the buffer pool, actually freeing all pooled GPU buffers.
    void      flushBufferPool();

    /// Enable/disable buffer pooling. Enabled by default.
    bool bufferPoolEnabled = true;

    // --- Pipelines ---
    const CompiledPipeline& getOrCreatePipeline(
        const std::string& name, const std::string& wgsl,
        uint32_t numBindings);

    /// Check if a pipeline with the given name is already cached.
    bool hasPipeline(const std::string& name) const;

    /// Find a cached pipeline by name. Returns nullptr on cache miss.
    const CompiledPipeline* findPipeline(const std::string& name) const;

    /// Pre-compile a batch of pipelines. Each spec is (name, wgsl, numBindings).
    /// Compiles shaders in parallel using std::async, then creates pipelines serially.
    /// Returns the number of newly compiled pipelines (already-cached ones are skipped).
    int warmupPipelines(
        const std::vector<std::tuple<std::string, std::string, uint32_t>>& specs);

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

    /// Submit dispatches with per-dispatch timestamp profiling (fire-and-forget).
    /// Each dispatch gets its own compute pass with timestamp writes.
    /// Call profiler.resolveAndReport() separately after all submits.
    void submitOnlyProfiled(const std::vector<Dispatch>& dispatches,
                            GPUProfiler& profiler);

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

    /// Submit dispatches + copy src to a staging buffer in ONE command buffer.
    /// Then start async map on the staging buffer (non-blocking).
    /// Returns the map future; call completeAsyncMapI32() later to get the result.
    /// passPerDispatch: each dispatch gets its own compute pass (allows Dawn
    ///   to elide barriers between passes that don't share writable buffers).
    WGPUFuture submitAndCopyAsync(const std::vector<Dispatch>& dispatches,
                                  GPUBuffer src, uint64_t readSize,
                                  WGPUBuffer stagingBuf,
                                  bool passPerDispatch = false);

    /// Wait for a previously started async map and read the i32 data.
    int32_t completeAsyncMapI32(WGPUBuffer stagingBuf, WGPUFuture future);

    /// Block until all submitted GPU work completes.
    void waitForQueue();

    // Detailed timing (nanoseconds) for benchmarking
    struct {
        int64_t encode_ns = 0;
        int64_t submit_ns = 0;
        int64_t map_start_ns = 0;
        int64_t wait_ns = 0;
        int64_t unmap_ns = 0;
        int64_t write_buf_ns = 0;
        int count = 0;
    } timing;

    /// Get or create a MAP_READ staging buffer of at least 'size' bytes.
    WGPUBuffer getOrCreateReadbackBuf(uint64_t size);

    /// Map the readback buffer and return its contents. Use after the readback
    /// copy was included in a submitted+waited command buffer (no new submit).
    std::vector<uint8_t> mapReadbackBuffer(uint64_t readSize);

private:
    std::unordered_map<std::string, CompiledPipeline> pipelines_;
    std::unordered_map<std::string, GPUBuffer>        buffers_;

    // Readback buffer (reused across calls)
    WGPUBuffer readbackBuf_     = nullptr;
    uint64_t   readbackBufSize_ = 0;

    // Buffer pool: bucketIndex → free list of buffers
    // Bucket i holds buffers of size 2^(i+4) bytes (bucket 0 = 16B, bucket 1 = 32B, ...)
    static constexpr int POOL_MIN_BITS = 4;   // 16 bytes min
    static constexpr int POOL_MAX_BITS = 30;  // 1GB max
    static constexpr int POOL_BUCKETS = POOL_MAX_BITS - POOL_MIN_BITS + 1;
    std::vector<GPUBuffer> pool_[POOL_BUCKETS];
    int poolBucket(uint64_t size) const;
};
