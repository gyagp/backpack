#include <cassert>
#include <cstdio>

#include "../src/gpu_context.h"
#include "../src/buffer_pool.h"

void test_acquire_creates_buffer() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    auto buf = pool.acquire(256, wgpu::BufferUsage::Storage);
    assert(buf != nullptr);
    assert(buf.GetSize() == 256);
    assert(buf.GetUsage() == wgpu::BufferUsage::Storage);
    printf("  PASS: acquire creates buffer\n");
}

void test_stats_initial() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    auto s = pool.stats();
    assert(s.hits == 0);
    assert(s.misses == 0);
    printf("  PASS: stats initial zeros\n");
}

void test_acquire_miss_increments() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    pool.acquire(256, wgpu::BufferUsage::Storage);
    auto s = pool.stats();
    assert(s.hits == 0);
    assert(s.misses == 1);
    printf("  PASS: acquire miss increments\n");
}

void test_release_and_reuse() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    auto buf1 = pool.acquire(512, wgpu::BufferUsage::Uniform);
    pool.release(std::move(buf1), 512, wgpu::BufferUsage::Uniform);

    auto buf2 = pool.acquire(512, wgpu::BufferUsage::Uniform);
    assert(buf2 != nullptr);

    auto s = pool.stats();
    assert(s.hits == 1);
    assert(s.misses == 1);
    printf("  PASS: release and reuse (hit)\n");
}

void test_no_reuse_different_size() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    auto buf = pool.acquire(256, wgpu::BufferUsage::Storage);
    pool.release(std::move(buf), 256, wgpu::BufferUsage::Storage);

    pool.acquire(512, wgpu::BufferUsage::Storage);
    auto s = pool.stats();
    assert(s.hits == 0);
    assert(s.misses == 2);
    printf("  PASS: no reuse different size\n");
}

void test_no_reuse_different_usage() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    auto buf = pool.acquire(256, wgpu::BufferUsage::Storage);
    pool.release(std::move(buf), 256, wgpu::BufferUsage::Storage);

    pool.acquire(256, wgpu::BufferUsage::Uniform);
    auto s = pool.stats();
    assert(s.hits == 0);
    assert(s.misses == 2);
    printf("  PASS: no reuse different usage\n");
}

void test_multiple_releases_reuse_order() {
    auto ctx = create_gpu_context();
    BufferPool pool(ctx.device);

    auto a = pool.acquire(128, wgpu::BufferUsage::Storage);
    auto b = pool.acquire(256, wgpu::BufferUsage::Storage);
    pool.release(std::move(a), 128, wgpu::BufferUsage::Storage);
    pool.release(std::move(b), 256, wgpu::BufferUsage::Storage);

    pool.acquire(256, wgpu::BufferUsage::Storage);
    auto s = pool.stats();
    assert(s.hits == 1);
    assert(s.misses == 2);
    printf("  PASS: multiple releases reuse correct size\n");
}

int main() {
    printf("test_buffer_pool:\n");
    test_acquire_creates_buffer();
    test_stats_initial();
    test_acquire_miss_increments();
    test_release_and_reuse();
    test_no_reuse_different_size();
    test_no_reuse_different_usage();
    test_multiple_releases_reuse_order();
    printf("All buffer pool tests passed.\n");
    return 0;
}
