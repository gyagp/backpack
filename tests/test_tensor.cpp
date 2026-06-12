#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "gpu_context.h"
#include "tensor.h"

void test_dtype_size_bytes() {
    assert(dtype_size_bytes(DType::f32) == 4);
    assert(dtype_size_bytes(DType::f16) == 2);
    assert(dtype_size_bytes(DType::q8_0) == 34);
    assert(dtype_size_bytes(DType::q4_k_m) == 144);
    printf("  dtype_size_bytes: OK\n");
}

void test_dtype_block_size() {
    assert(dtype_block_size(DType::f32) == 1);
    assert(dtype_block_size(DType::f16) == 1);
    assert(dtype_block_size(DType::q8_0) == 32);
    assert(dtype_block_size(DType::q4_k_m) == 256);
    printf("  dtype_block_size: OK\n");
}

void test_compute_buffer_size() {
    // Non-quantized
    assert(compute_buffer_size(24, DType::f32) == 24 * 4);
    assert(compute_buffer_size(24, DType::f16) == 24 * 2);

    // Quantized exact multiple
    assert(compute_buffer_size(32, DType::q8_0) == 1 * 34);
    assert(compute_buffer_size(256, DType::q4_k_m) == 1 * 144);

    // Quantized partial block rounds up
    assert(compute_buffer_size(33, DType::q8_0) == 2 * 34);
    assert(compute_buffer_size(257, DType::q4_k_m) == 2 * 144);

    // Zero elements
    assert(compute_buffer_size(0, DType::f32) == 0);

    // Large element count
    assert(compute_buffer_size(1024, DType::q8_0) == 32 * 34);
    assert(compute_buffer_size(512, DType::q4_k_m) == 2 * 144);

    printf("  compute_buffer_size: OK\n");
}

void test_element_count() {
    Tensor t;
    t.shape = {2, 3, 4};
    t.dtype = DType::f32;
    assert(t.element_count() == 24);

    Tensor scalar;
    scalar.shape = {};
    scalar.dtype = DType::f16;
    assert(scalar.element_count() == 1);

    Tensor vec;
    vec.shape = {10};
    vec.dtype = DType::q8_0;
    assert(vec.element_count() == 10);

    Tensor ones;
    ones.shape = {1, 1, 5};
    ones.dtype = DType::q4_k_m;
    assert(ones.element_count() == 5);

    printf("  element_count: OK\n");
}

void test_tensor_create(const wgpu::Device& device) {
    // f32 tensor
    {
        auto t = Tensor::create(device, {4, 8}, DType::f32);
        assert(t.shape.size() == 2);
        assert(t.shape[0] == 4);
        assert(t.shape[1] == 8);
        assert(t.dtype == DType::f32);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 4 * 8 * 4);
    }

    // f16 tensor
    {
        auto t = Tensor::create(device, {16}, DType::f16);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 16 * 2);
    }

    // q8_0 tensor - exact block multiple
    {
        auto t = Tensor::create(device, {64}, DType::q8_0);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 2 * 34);
    }

    // q8_0 tensor - partial block
    {
        auto t = Tensor::create(device, {33}, DType::q8_0);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 2 * 34);
    }

    // q4_k_m tensor
    {
        auto t = Tensor::create(device, {256}, DType::q4_k_m);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 1 * 144);
    }

    // q4_k_m partial block
    {
        auto t = Tensor::create(device, {257}, DType::q4_k_m);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 2 * 144);
    }

    // Multi-dimensional
    {
        auto t = Tensor::create(device, {2, 3, 4}, DType::f32);
        assert(t.element_count() == 24);
        assert(t.buffer.GetSize() == 24 * 4);
    }

    printf("  Tensor::create: OK\n");
}

void test_tensor_from_data(const wgpu::Device& device, const wgpu::Queue& queue) {
    // Upload f32 data
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto t = Tensor::from_data(device, queue, {4}, DType::f32,
                                   data.data(), data.size() * sizeof(float));
        assert(t.shape.size() == 1);
        assert(t.shape[0] == 4);
        assert(t.dtype == DType::f32);
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 16);
    }

    // Upload f16 data (raw bytes)
    {
        uint16_t data[8] = {0x3C00, 0x4000, 0x4200, 0x4400,
                            0x4500, 0x4600, 0x4700, 0x4800};
        auto t = Tensor::from_data(device, queue, {8}, DType::f16,
                                   data, sizeof(data));
        assert(t.buffer != nullptr);
        assert(t.buffer.GetSize() == 16);
    }

    // Upload 2D f32 data
    {
        std::vector<float> data(12, 1.0f);
        auto t = Tensor::from_data(device, queue, {3, 4}, DType::f32,
                                   data.data(), data.size() * sizeof(float));
        assert(t.element_count() == 12);
        assert(t.buffer.GetSize() == 48);
    }

    printf("  Tensor::from_data: OK\n");
}

int main() {
    printf("Running tensor tests...\n");

    // Pure logic tests (no GPU)
    test_dtype_size_bytes();
    test_dtype_block_size();
    test_compute_buffer_size();
    test_element_count();

    // GPU tests
    printf("Initializing GPU...\n");
    auto ctx = create_gpu_context();

    test_tensor_create(ctx.device);
    test_tensor_from_data(ctx.device, ctx.queue);

    printf("All tensor tests passed.\n");
    return 0;
}
