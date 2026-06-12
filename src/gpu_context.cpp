#include "gpu_context.h"

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

static dawn::native::Instance* get_native_instance() {
    static auto instance = []() {
        dawnProcSetProcs(&dawn::native::GetProcs());
        return std::make_unique<dawn::native::Instance>();
    }();
    return instance.get();
}

static wgpu::BackendType resolve_backend(GpuBackend backend) {
    if (backend == GpuBackend::Default) {
        const char* env = std::getenv("BACKPACK_BACKEND");
        if (env) {
            if (std::strcmp(env, "vulkan") == 0)
                return wgpu::BackendType::Vulkan;
            if (std::strcmp(env, "d3d12") == 0)
                return wgpu::BackendType::D3D12;
            std::fprintf(stderr, "Unknown BACKPACK_BACKEND value: %s (expected vulkan or d3d12)\n", env);
            std::exit(1);
        }
#ifdef _WIN32
        return wgpu::BackendType::D3D12;
#else
        return wgpu::BackendType::Vulkan;
#endif
    }
    if (backend == GpuBackend::D3D12)
        return wgpu::BackendType::D3D12;
    return wgpu::BackendType::Vulkan;
}

GpuContext create_gpu_context(GpuBackend backend) {
    GpuContext ctx{};

    auto* native = get_native_instance();
    ctx.instance = wgpu::Instance(native->Get());

    wgpu::RequestAdapterOptions adapter_opts{};
    adapter_opts.backendType = resolve_backend(backend);

    auto adapters = native->EnumerateAdapters(&adapter_opts);
    if (adapters.empty()) {
        std::fprintf(stderr, "No WebGPU adapters found for requested backend\n");
        std::exit(1);
    }
    ctx.adapter = wgpu::Adapter(adapters[0].Get());

    wgpu::DeviceDescriptor device_desc{};
    device_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Unknown) {
                std::fprintf(stderr, "Device lost: %.*s\n",
                             static_cast<int>(message.length), message.data);
            }
        });
    device_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            std::fprintf(stderr, "WebGPU error: %.*s\n",
                         static_cast<int>(message.length), message.data);
        });

    bool got_device = false;
    ctx.adapter.RequestDevice(
        &device_desc, wgpu::CallbackMode::AllowSpontaneous,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                ctx.device = std::move(result);
                got_device = true;
            }
        });

    if (!got_device) {
        std::fprintf(stderr, "Failed to get WebGPU device\n");
        std::exit(1);
    }

    ctx.queue = ctx.device.GetQueue();
    return ctx;
}
