#pragma once

#include <dawn/webgpu_cpp.h>

enum class GpuBackend {
    Default,
    D3D12,
    Vulkan,
};

struct GpuContext {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
};

GpuContext create_gpu_context(GpuBackend backend = GpuBackend::Default);
