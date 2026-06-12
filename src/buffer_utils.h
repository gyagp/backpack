#pragma once

#include <dawn/webgpu_cpp.h>

#include <cstddef>
#include <cstring>
#include <vector>

inline wgpu::Buffer create_buffer(const wgpu::Device& device,
                                  uint64_t size,
                                  wgpu::BufferUsage usage,
                                  const void* data = nullptr) {
    wgpu::BufferDescriptor desc{};
    desc.size = size;
    desc.usage = usage;
    if (data) {
        desc.mappedAtCreation = true;
    }
    wgpu::Buffer buf = device.CreateBuffer(&desc);
    if (data) {
        std::memcpy(buf.GetMappedRange(), data, size);
        buf.Unmap();
    }
    return buf;
}

inline wgpu::Buffer create_storage_buffer(const wgpu::Device& device,
                                          uint64_t size,
                                          const void* data = nullptr) {
    return create_buffer(device, size,
                         wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                         data);
}

inline wgpu::Buffer create_read_buffer(const wgpu::Device& device, uint64_t size) {
    return create_buffer(device, size,
                         wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
}

inline std::vector<uint8_t> read_buffer(const wgpu::Device& device,
                                        const wgpu::Queue& queue,
                                        const wgpu::Buffer& src,
                                        uint64_t size) {
    wgpu::Buffer staging = create_read_buffer(device, size);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(src, 0, staging, 0, size);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool done = false;
    std::vector<uint8_t> result(size);

    staging.MapAsync(wgpu::MapMode::Read, 0, size,
                     wgpu::CallbackMode::AllowSpontaneous,
                     [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
                         if (status == wgpu::MapAsyncStatus::Success) {
                             std::memcpy(result.data(), staging.GetConstMappedRange(), size);
                         }
                         done = true;
                     });

    while (!done) {
        device.Tick();
    }

    staging.Unmap();
    return result;
}

template <typename T>
std::vector<T> read_buffer_as(const wgpu::Device& device,
                              const wgpu::Queue& queue,
                              const wgpu::Buffer& src,
                              size_t count) {
    auto bytes = read_buffer(device, queue, src, count * sizeof(T));
    std::vector<T> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}
