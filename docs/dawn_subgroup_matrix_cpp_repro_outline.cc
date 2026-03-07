// Ultra-minimal C++ repro outline for Dawn Vulkan subgroup-matrix failure.
//
// Intent:
// - stay close to Dawn's native test/sample environment
// - avoid Triton / Python / external wrappers
// - reproduce the exact boundary already shown by the raw ctypes repro:
//   1. feature-only shader compiles
//   2. subgroup-matrix type declaration shader fails
//   3. subgroupMatrixLoad/MMA/Store shader fails
//
// This is an outline, not a checked-in build target. The helper functions
// below are intentionally small and explicit so this can be dropped into
// a local Dawn scratch executable or adapted into a Dawn test.

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <dawn/dawn_proc.h>
#include <webgpu/webgpu.h>

namespace {

constexpr char kFeatureOnlyShader[] = R"(
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    Out[lid.x] = 1.0;
}
)";

constexpr char kTypeDeclShader[] = R"(
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var matA: subgroup_matrix_left<f32, 8, 8>;
    var matB: subgroup_matrix_right<f32, 8, 8>;
    var matC: subgroup_matrix_result<f32, 8, 8>;
    Out[lid.x] = 0.0;
}
)";

constexpr char kMmaShader[] = R"(
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

var<workgroup> TileA: array<f32, 64>;
var<workgroup> TileB: array<f32, 64>;
var<workgroup> TileC: array<f32, 64>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = lid.x;
    TileA[idx] = 1.0;
    TileA[idx + 32u] = 1.0;
    TileB[idx] = 1.0;
    TileB[idx + 32u] = 1.0;
    TileC[idx] = 0.0;
    TileC[idx + 32u] = 0.0;
    workgroupBarrier();

    let matA = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(
        &TileA, 0u, false, 8u);
    let matB = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(
        &TileB, 0u, true, 8u);
    var matC: subgroup_matrix_result<f32, 8, 8>;
    matC = subgroupMatrixMultiplyAccumulate(matA, matB, matC);
    subgroupMatrixStore(&TileC, 0u, matC, false, 8u);
    workgroupBarrier();

    Out[idx] = TileC[idx];
    Out[idx + 32u] = TileC[idx + 32u];
}
)";

struct AdapterRequestState {
    WGPUAdapter adapter = nullptr;
    std::string error;
    bool done = false;
};

struct PipelineRequestState {
    WGPUComputePipeline pipeline = nullptr;
    std::string error;
    bool done = false;
};

std::string ToString(WGPUStringView view) {
    if (!view.data || view.length == 0) {
        return {};
    }
    return std::string(view.data, view.length);
}

const char* BackendName(WGPUBackendType backend) {
    switch (backend) {
        case WGPUBackendType_Vulkan:
            return "Vulkan";
        case WGPUBackendType_D3D12:
            return "D3D12";
        case WGPUBackendType_D3D11:
            return "D3D11";
        case WGPUBackendType_Metal:
            return "Metal";
        default:
            return "Unknown";
    }
}

void PrintAdapterInfo(WGPUAdapter adapter) {
    WGPUAdapterInfo info{};
    wgpuAdapterGetInfo(adapter, &info);
    std::cout << "Adapter.vendor=" << ToString(info.vendor) << "\n";
    std::cout << "Adapter.architecture=" << ToString(info.architecture) << "\n";
    std::cout << "Adapter.device=" << ToString(info.device) << "\n";
    std::cout << "Adapter.description=" << ToString(info.description) << "\n";
    std::cout << "Adapter.backend=" << BackendName(static_cast<WGPUBackendType>(info.backendType)) << "\n";
    std::cout << "Adapter.vendorID=" << info.vendorID << "\n";
    std::cout << "Adapter.deviceID=" << info.deviceID << "\n";
}

WGPUInstance CreateInstance() {
    WGPUInstanceDescriptor desc{};
    WGPUInstance instance = wgpuCreateInstance(&desc);
    if (!instance) {
        std::cerr << "Failed to create WGPU instance\n";
        std::exit(1);
    }
    return instance;
}

WGPUAdapter RequestAdapterSync(WGPUInstance instance) {
    AdapterRequestState state;

    std::array<const char*, 2> toggles = {
        "allow_unsafe_apis",
        "vulkan_enable_f16_on_nvidia",
    };
    WGPUDawnTogglesDescriptor dawnToggles{};
    dawnToggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    dawnToggles.enabledToggleCount = toggles.size();
    dawnToggles.enabledToggles = toggles.data();

    WGPURequestAdapterOptions options{};
    options.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&dawnToggles);
    options.featureLevel = WGPUFeatureLevel_Core;
    options.powerPreference = WGPUPowerPreference_HighPerformance;
    options.backendType = WGPUBackendType_Vulkan;

    auto callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView message, void* userdata1, void*) {
        auto* state = static_cast<AdapterRequestState*>(userdata1);
        state->done = true;
        if (status == WGPURequestAdapterStatus_Success) {
            state->adapter = adapter;
        } else {
            state->error = ToString(message);
        }
    };

    WGPURequestAdapterCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = callback;
    cb.userdata1 = &state;

    wgpuInstanceRequestAdapter(instance, &options, cb);
    while (!state.done) {
        wgpuInstanceProcessEvents(instance);
    }

    if (!state.adapter) {
        std::cerr << "RequestAdapter failed: " << state.error << "\n";
        std::exit(1);
    }
    return state.adapter;
}

WGPUDevice CreateDevice(WGPUAdapter adapter) {
    constexpr WGPUFeatureName kFeatureSubgroupsNew = static_cast<WGPUFeatureName>(0x00000012);
    constexpr WGPUFeatureName kFeatureSubgroupsOld = static_cast<WGPUFeatureName>(0x00000011);
    constexpr WGPUFeatureName kFeatureSubgroupMatrix = static_cast<WGPUFeatureName>(0x00050034);

    std::vector<WGPUFeatureName> features;
    if (wgpuAdapterHasFeature(adapter, kFeatureSubgroupsNew)) {
        features.push_back(kFeatureSubgroupsNew);
    } else if (wgpuAdapterHasFeature(adapter, kFeatureSubgroupsOld)) {
        features.push_back(kFeatureSubgroupsOld);
    }
    if (wgpuAdapterHasFeature(adapter, kFeatureSubgroupMatrix)) {
        features.push_back(kFeatureSubgroupMatrix);
    }

    WGPULimits limits{};
    wgpuAdapterGetLimits(adapter, &limits);

    auto errorCallback = [](WGPUDevice const*, WGPUErrorType type,
                            WGPUStringView message, void*, void*) {
        std::cerr << "[DAWN ERROR] type=" << static_cast<int>(type)
                  << " msg=" << ToString(message) << "\n";
    };

    WGPUDeviceDescriptor desc{};
    desc.label = WGPUStringView{"cpp-repro", 9};
    desc.requiredFeatureCount = features.size();
    desc.requiredFeatures = features.empty() ? nullptr : features.data();
    desc.requiredLimits = &limits;
    desc.uncapturedErrorCallbackInfo.callback = errorCallback;

    WGPUDevice device = wgpuAdapterCreateDevice(adapter, &desc);
    if (!device) {
        std::cerr << "Failed to create device\n";
        std::exit(1);
    }
    return device;
}

WGPUBindGroupLayout CreateStorageOnlyBindGroupLayout(WGPUDevice device) {
    WGPUBindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Compute;
    entry.buffer.type = WGPUBufferBindingType_Storage;

    WGPUBindGroupLayoutDescriptor desc{};
    desc.entryCount = 1;
    desc.entries = &entry;
    return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUPipelineLayout CreatePipelineLayout(WGPUDevice device, WGPUBindGroupLayout bgl) {
    WGPUBindGroupLayout layouts[] = {bgl};
    WGPUPipelineLayoutDescriptor desc{};
    desc.bindGroupLayoutCount = 1;
    desc.bindGroupLayouts = layouts;
    return wgpuDeviceCreatePipelineLayout(device, &desc);
}

std::optional<std::string> TryCompilePipelineSync(WGPUInstance instance,
                                                  WGPUDevice device,
                                                  WGPUPipelineLayout pipelineLayout,
                                                  const char* wgsl) {
    PipelineRequestState state;

    WGPUShaderSourceWGSL source{};
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = WGPUStringView{wgsl, std::strlen(wgsl)};

    WGPUShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shaderDesc);

    WGPUComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.label = WGPUStringView{"cpp-repro-pipeline", 18};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute.module = shader;
    pipelineDesc.compute.entryPoint = WGPUStringView{"main", 4};

    auto callback = [](WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline,
                       WGPUStringView message, void* userdata1, void*) {
        auto* state = static_cast<PipelineRequestState*>(userdata1);
        state->done = true;
        if (status == WGPUCreatePipelineAsyncStatus_Success) {
            state->pipeline = pipeline;
        } else {
            state->error = ToString(message);
        }
    };

    WGPUCreateComputePipelineAsyncCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = callback;
    cb.userdata1 = &state;

    wgpuDeviceCreateComputePipelineAsync(device, &pipelineDesc, cb);
    while (!state.done) {
        wgpuInstanceProcessEvents(instance);
    }

    if (state.pipeline) {
        return std::nullopt;
    }
    return state.error.empty() ? std::optional<std::string>("unknown pipeline creation error")
                               : std::optional<std::string>(state.error);
}

}  // namespace

int main() {
    // In a real standalone target, set Dawn proc table here if your build needs it.
    // Example sketch:
    //   dawnProcSetProcs(&dawn::native::GetProcs());

    WGPUInstance instance = CreateInstance();
    WGPUAdapter adapter = RequestAdapterSync(instance);
    PrintAdapterInfo(adapter);
    WGPUDevice device = CreateDevice(adapter);
    WGPUBindGroupLayout bgl = CreateStorageOnlyBindGroupLayout(device);
    WGPUPipelineLayout pipelineLayout = CreatePipelineLayout(device, bgl);

    struct Case {
        const char* name;
        const char* shader;
    };
    const Case cases[] = {
        {"Feature-only shader", kFeatureOnlyShader},
        {"Type declaration shader", kTypeDeclShader},
        {"Load + MMA + Store shader", kMmaShader},
    };

    for (const auto& testCase : cases) {
        std::cout << "\n== " << testCase.name << " ==\n";
        if (auto error = TryCompilePipelineSync(instance, device, pipelineLayout, testCase.shader)) {
            std::cout << "Compile failed:\n" << *error << "\n";
        } else {
            std::cout << "Compiled successfully\n";
        }
    }

    return 0;
}