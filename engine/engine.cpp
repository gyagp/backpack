/**
 * Stage 2: C++ Execution Engine for Backpack models.
 *
 * Loads a compiled model bundle (manifest.json + WGSL kernels) and
 * GGUF weights, then runs inference using Dawn's native WebGPU C API.
 *
 * This eliminates all Python/ctypes overhead from the decode loop,
 * providing near-native dispatch performance comparable to llama.cpp.
 *
 * Usage:
 *   backpack_engine \
 *     --bundle build/qwen3-1.7B \
 *     --gguf-file weights/Qwen3-1.7B-Q8_0.gguf \
 *     --prompt "Hello world" \
 *     --max-tokens 100
 */

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_loader.h"

// ─── JSON mini-parser (enough for manifest.json) ─────────────────────────────

// We only need to read flat objects from manifest.json. For a real
// implementation, use nlohmann/json. This is a minimal inline parser
// that handles the specific manifest structure we generate.

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    return {std::istreambuf_iterator<char>(f), {}};
}

// ─── GPU Context ─────────────────────────────────────────────────────────────

struct GPUContext {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;

    // Cached pipeline/bind-group layout per WGSL hash
    struct Pipeline {
        WGPUShaderModule shader;
        WGPUComputePipeline pipeline;
        WGPUBindGroupLayout layout;
        WGPUPipelineLayout pipelineLayout;
    };
    std::unordered_map<std::string, Pipeline> pipelines;

    // Buffer pool
    std::unordered_map<std::string, WGPUBuffer> named_buffers;

    bool init(WGPUBackendType backend = WGPUBackendType_Vulkan);
    void destroy();

    WGPUBuffer createBuffer(const std::string& name, uint64_t size,
                            uint64_t usage, bool mappedAtCreation = false);
    WGPUBuffer getOrCreateBuffer(const std::string& name, uint64_t size,
                                 uint64_t usage);
    void writeBuffer(WGPUBuffer buf, const void* data, uint64_t size);

    Pipeline& getOrCreatePipeline(const std::string& name,
                                   const std::string& wgsl,
                                   uint32_t numBindings);

    // Synchronous GPU fence
    void waitForQueue();
};

// ─── GPUContext implementation ────────────────────────────────────────────────

static std::string sv_to_string(WGPUStringView sv) {
    if (!sv.data || sv.length == 0) return {};
    return {sv.data, sv.length};
}

bool GPUContext::init(WGPUBackendType backend) {
    // Create instance with TimedWaitAny feature
    WGPUInstanceFeatureName instFeatures[] = {
        static_cast<WGPUInstanceFeatureName>(0x00000001)  // TimedWaitAny
    };
    WGPUInstanceDescriptor idesc{};
    idesc.requiredFeatureCount = 1;
    idesc.requiredFeatures = instFeatures;
    instance = wgpuCreateInstance(&idesc);
    if (!instance) return false;

    // Request adapter
    struct { WGPUAdapter adapter; bool done; } state{nullptr, false};

    const char* toggles[] = {"allow_unsafe_apis", "vulkan_enable_f16_on_nvidia"};
    WGPUDawnTogglesDescriptor dawnToggles{};
    dawnToggles.chain.sType = static_cast<WGPUSType>(0x0005000A);
    dawnToggles.enabledToggleCount = 2;
    dawnToggles.enabledToggles = toggles;

    WGPURequestAdapterOptions opts{};
    opts.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&dawnToggles);
    opts.featureLevel = WGPUFeatureLevel_Core;
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    opts.backendType = backend;

    WGPURequestAdapterCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter a,
                     WGPUStringView msg, void* ud1, void*) {
        auto* s = static_cast<decltype(&state)>(ud1);
        s->adapter = (status == WGPURequestAdapterStatus_Success) ? a : nullptr;
        s->done = true;
    };
    cb.userdata1 = &state;

    auto future = wgpuInstanceRequestAdapter(instance, &opts, cb);
    WGPUFutureWaitInfo wait{future, 0};
    wgpuInstanceWaitAny(instance, 1, &wait, UINT64_MAX);
    adapter = state.adapter;
    if (!adapter) return false;

    // Print adapter info
    WGPUAdapterInfo info{};
    wgpuAdapterGetInfo(adapter, &info);
    printf("GPU: %s (%s)\n", sv_to_string(info.device).c_str(),
           sv_to_string(info.description).c_str());

    // Query limits
    WGPULimits limits{};
    wgpuAdapterGetLimits(adapter, &limits);

    // Detect features
    std::vector<WGPUFeatureName> features;
    uint32_t f16_ids[] = {0x0B, 0x0A};
    uint32_t sg_ids[] = {0x12, 0x11};
    for (auto id : f16_ids) if (wgpuAdapterHasFeature(adapter, (WGPUFeatureName)id)) {
        features.push_back((WGPUFeatureName)id); break;
    }
    for (auto id : sg_ids) if (wgpuAdapterHasFeature(adapter, (WGPUFeatureName)id)) {
        features.push_back((WGPUFeatureName)id); break;
    }
    if (wgpuAdapterHasFeature(adapter, (WGPUFeatureName)0x00050034))
        features.push_back((WGPUFeatureName)0x00050034);
    if (wgpuAdapterHasFeature(adapter, (WGPUFeatureName)0x09))
        features.push_back((WGPUFeatureName)0x09);

    // Create device
    const char* devTogglesE[] = {"skip_validation", "disable_robustness",
                                 "d3d_disable_ieee_strictness"};
    const char* devTogglesD[] = {"lazy_clear_resource_on_first_use",
                                 "timestamp_quantization"};
    WGPUDawnTogglesDescriptor devToggles{};
    devToggles.chain.sType = static_cast<WGPUSType>(0x0005000A);
    devToggles.enabledToggleCount = 3;
    devToggles.enabledToggles = devTogglesE;
    devToggles.disabledToggleCount = 2;
    devToggles.disabledToggles = devTogglesD;

    WGPUDeviceDescriptor ddesc{};
    ddesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&devToggles);
    ddesc.label = {"engine", 6};
    ddesc.requiredFeatureCount = features.size();
    ddesc.requiredFeatures = features.empty() ? nullptr : features.data();
    ddesc.requiredLimits = &limits;

    // Error callback
    ddesc.uncapturedErrorCallbackInfo.callback = [](
        WGPUDevice const*, WGPUErrorType type, WGPUStringView msg, void*, void*) {
        const char* names[] = {"None","Validation","OOM","Internal","Unknown","Lost"};
        fprintf(stderr, "[DAWN] %s: %.*s\n", names[std::min((int)type,5)],
                (int)msg.length, msg.data);
    };

    device = wgpuAdapterCreateDevice(adapter, &ddesc);
    if (!device) return false;

    queue = wgpuDeviceGetQueue(device);
    return true;
}

void GPUContext::destroy() {
    // Release cached pipelines, buffers, then device/adapter/instance
    for (auto& [name, buf] : named_buffers) wgpuBufferRelease(buf);
    for (auto& [key, p] : pipelines) {
        wgpuComputePipelineRelease(p.pipeline);
        wgpuShaderModuleRelease(p.shader);
        wgpuBindGroupLayoutRelease(p.layout);
        wgpuPipelineLayoutRelease(p.pipelineLayout);
    }
    if (queue) wgpuQueueRelease(queue);
    if (device) wgpuDeviceRelease(device);
    if (adapter) wgpuAdapterRelease(adapter);
    if (instance) wgpuInstanceRelease(instance);
}

WGPUBuffer GPUContext::createBuffer(const std::string& name, uint64_t size,
                                     uint64_t usage, bool mappedAtCreation) {
    WGPUBufferDescriptor desc{};
    desc.label = {name.c_str(), name.size()};
    desc.usage = usage;
    desc.size = size;
    desc.mappedAtCreation = mappedAtCreation ? 1u : 0u;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
    named_buffers[name] = buf;
    return buf;
}

WGPUBuffer GPUContext::getOrCreateBuffer(const std::string& name, uint64_t size,
                                          uint64_t usage) {
    auto it = named_buffers.find(name);
    if (it != named_buffers.end()) return it->second;
    return createBuffer(name, size, usage);
}

void GPUContext::writeBuffer(WGPUBuffer buf, const void* data, uint64_t size) {
    wgpuQueueWriteBuffer(queue, buf, 0, data, size);
}

GPUContext::Pipeline& GPUContext::getOrCreatePipeline(
        const std::string& name, const std::string& wgsl, uint32_t numBindings) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;

    Pipeline p{};

    // Shader module
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = {wgsl.c_str(), wgsl.size()};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&src);
    p.shader = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Bind group layout: all storage buffers
    std::vector<WGPUBindGroupLayoutEntry> entries(numBindings);
    for (uint32_t i = 0; i < numBindings; i++) {
        entries[i] = {};
        entries[i].binding = i;
        entries[i].visibility = WGPUShaderStage_Compute;
        entries[i].buffer.type = WGPUBufferBindingType_Storage;
    }
    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = numBindings;
    bglDesc.entries = entries.data();
    p.layout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    // Pipeline layout
    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &p.layout;
    p.pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    // Compute pipeline
    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.label = {name.c_str(), name.size()};
    cpDesc.layout = p.pipelineLayout;
    cpDesc.compute.module = p.shader;
    cpDesc.compute.entryPoint = {"main", 4};
    p.pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);

    if (!p.pipeline) {
        fprintf(stderr, "Failed to create pipeline: %s\n", name.c_str());
        exit(1);
    }

    pipelines[name] = p;
    return pipelines[name];
}

void GPUContext::waitForQueue() {
    // Use QueueOnSubmittedWorkDone as GPU fence
    struct { bool done; } s{false};

    // Callback info struct (same layout as BufferMapCallbackInfo)
    struct CBInfo {
        void* nextInChain;
        uint32_t mode;
        void* callback;
        void* ud1;
        void* ud2;
    };

    auto cb = [](uint32_t, WGPUStringView, void* ud1, void*) {
        static_cast<decltype(&s)>(ud1)->done = true;
    };

    // Use WaitAnyOnly mode
    CBInfo cbInfo{nullptr, 1 /*WaitAnyOnly*/,
                  reinterpret_cast<void*>(static_cast<void(*)(uint32_t,WGPUStringView,void*,void*)>(cb)),
                  &s, nullptr};

    // wgpuQueueOnSubmittedWorkDone takes the callback info by value
    typedef WGPUFuture (*PFN_QueueOnSubmittedWorkDone)(WGPUQueue, CBInfo);
    auto fn = reinterpret_cast<PFN_QueueOnSubmittedWorkDone>(
        &wgpuQueueOnSubmittedWorkDone);
    auto future = fn(queue, cbInfo);

    WGPUFutureWaitInfo wait{future, 0};
    wgpuInstanceWaitAny(instance, 1, &wait, UINT64_MAX);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string bundle_dir, gguf_path, prompt = "Hello world";
    int max_tokens = 50;
    std::string backend_str = "vulkan";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bundle" && i+1 < argc) bundle_dir = argv[++i];
        else if (arg == "--gguf-file" && i+1 < argc) gguf_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backend_str = argv[++i];
    }

    if (bundle_dir.empty() || gguf_path.empty()) {
        fprintf(stderr, "Usage: %s --bundle <dir> --gguf-file <path> "
                "[--prompt <text>] [--max-tokens <n>] [--backend vulkan|d3d12]\n",
                argv[0]);
        return 1;
    }

    WGPUBackendType backend = WGPUBackendType_Vulkan;
    if (backend_str == "d3d12") backend = WGPUBackendType_D3D12;
    else if (backend_str == "metal") backend = WGPUBackendType_Metal;

    // Initialize GPU
    GPUContext gpu;
    if (!gpu.init(backend)) {
        fprintf(stderr, "Failed to initialize GPU\n");
        return 1;
    }

    // Load manifest
    printf("Loading bundle: %s\n", bundle_dir.c_str());
    std::string manifest_str = read_file(bundle_dir + "/manifest.json");
    // TODO: parse manifest JSON and load kernels
    // For now, just validate the setup works

    // Load GGUF
    printf("Loading GGUF: %s\n", gguf_path.c_str());
    GGUFFile gguf;
    if (!gguf.open(gguf_path)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path.c_str());
        return 1;
    }
    printf("  %llu tensors, version %u\n",
           (unsigned long long)gguf.n_tensors, gguf.version);

    // Load a test kernel
    std::string wgsl = read_file(bundle_dir + "/kernels/rms_norm.wgsl");
    auto& pl = gpu.getOrCreatePipeline("rms_norm", wgsl, 5);
    printf("  rms_norm pipeline: OK\n");

    // Test: create a buffer and dispatch
    uint64_t BUF_USAGE = 0x0080 | 0x0004 | 0x0008;
    WGPUBuffer test_buf = gpu.createBuffer("test", 4096, BUF_USAGE, false);
    printf("  Test buffer: OK\n");

    printf("\nEngine initialized successfully.\n");
    printf("Bundle: %s\n", bundle_dir.c_str());
    printf("GGUF: %llu tensors\n", (unsigned long long)gguf.n_tensors);
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n", max_tokens);
    printf("\n[TODO] Full inference loop not yet implemented.\n");
    printf("The compiled WGSL kernels are ready for execution.\n");
    printf("Next: implement weight upload, decode dispatch loop, tokenizer.\n");

    gpu.destroy();
    return 0;
}

