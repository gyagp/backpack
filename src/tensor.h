#pragma once

#include <cstdint>
#include <numeric>
#include <vector>
#include <dawn/webgpu_cpp.h>

enum class DType {
    f32,
    f16,
    q8_0,
    q4_k_m,
};

inline uint32_t dtype_size_bytes(DType dtype) {
    switch (dtype) {
        case DType::f32:    return 4;
        case DType::f16:    return 2;
        case DType::q8_0:   return 34;   // 32 int8 weights + 1 f16 scale = 34 bytes per block of 32
        case DType::q4_k_m: return 144;  // 256 values per super-block: 2*f16 + 12 packed scales/mins + 128 nibbles
    }
    return 0;
}

inline uint32_t dtype_block_size(DType dtype) {
    switch (dtype) {
        case DType::f32:    return 1;
        case DType::f16:    return 1;
        case DType::q8_0:   return 32;
        case DType::q4_k_m: return 256;
    }
    return 1;
}

inline uint64_t compute_buffer_size(uint32_t element_count, DType dtype) {
    uint32_t block = dtype_block_size(dtype);
    uint32_t num_blocks = (element_count + block - 1) / block;
    return static_cast<uint64_t>(num_blocks) * dtype_size_bytes(dtype);
}

struct Tensor {
    std::vector<uint32_t> shape;
    DType dtype;
    wgpu::Buffer buffer;

    uint32_t element_count() const {
        return std::accumulate(shape.begin(), shape.end(), uint32_t{1}, std::multiplies<uint32_t>());
    }

    static Tensor create(const wgpu::Device& device, std::vector<uint32_t> shape, DType dtype) {
        Tensor t;
        t.shape = std::move(shape);
        t.dtype = dtype;
        uint64_t size = compute_buffer_size(t.element_count(), dtype);
        wgpu::BufferDescriptor desc{};
        desc.size = size;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
        t.buffer = device.CreateBuffer(&desc);
        return t;
    }

    static Tensor from_data(const wgpu::Device& device, const wgpu::Queue& queue,
                            std::vector<uint32_t> shape, DType dtype,
                            const void* data, uint64_t size) {
        Tensor t = create(device, std::move(shape), dtype);
        queue.WriteBuffer(t.buffer, 0, data, size);
        return t;
    }
};
