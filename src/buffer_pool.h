#pragma once

#include <dawn/webgpu_cpp.h>

#include <cstdint>
#include <vector>

struct BufferPoolStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
};

class BufferPool {
public:
    explicit BufferPool(const wgpu::Device& device) : device_(device) {}

    wgpu::Buffer acquire(uint64_t size, wgpu::BufferUsage usage) {
        for (auto it = pool_.begin(); it != pool_.end(); ++it) {
            if (it->size == size && it->usage == usage) {
                wgpu::Buffer buf = std::move(it->buffer);
                pool_.erase(it);
                ++stats_.hits;
                return buf;
            }
        }

        ++stats_.misses;
        wgpu::BufferDescriptor desc{};
        desc.size = size;
        desc.usage = usage;
        return device_.CreateBuffer(&desc);
    }

    void release(wgpu::Buffer buffer, uint64_t size, wgpu::BufferUsage usage) {
        pool_.push_back({std::move(buffer), size, usage});
    }

    BufferPoolStats stats() const { return stats_; }

private:
    struct Entry {
        wgpu::Buffer buffer;
        uint64_t size;
        wgpu::BufferUsage usage;
    };

    wgpu::Device device_;
    std::vector<Entry> pool_;
    BufferPoolStats stats_;
};
