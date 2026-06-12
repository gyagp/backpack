#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

// Re-declare types without Dawn dependency for standalone testing
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
        case DType::q8_0:   return 34;
        case DType::q4_k_m: return 144;
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

    uint32_t element_count() const {
        return std::accumulate(shape.begin(), shape.end(), uint32_t{1}, std::multiplies<uint32_t>());
    }
};

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
    assert(compute_buffer_size(0, DType::q8_0) == 0);

    // Large counts
    assert(compute_buffer_size(1024, DType::q8_0) == 32 * 34);
    assert(compute_buffer_size(512, DType::q4_k_m) == 2 * 144);

    // Exactly one block
    assert(compute_buffer_size(1, DType::q8_0) == 1 * 34);
    assert(compute_buffer_size(1, DType::q4_k_m) == 1 * 144);

    printf("  compute_buffer_size: OK\n");
}

void test_element_count() {
    // Multi-dimensional
    Tensor t;
    t.shape = {2, 3, 4};
    t.dtype = DType::f32;
    assert(t.element_count() == 24);

    // Scalar (empty shape)
    Tensor scalar;
    scalar.shape = {};
    scalar.dtype = DType::f16;
    assert(scalar.element_count() == 1);

    // 1D
    Tensor vec;
    vec.shape = {10};
    vec.dtype = DType::q8_0;
    assert(vec.element_count() == 10);

    // Shape with 1s
    Tensor ones;
    ones.shape = {1, 1, 5};
    ones.dtype = DType::q4_k_m;
    assert(ones.element_count() == 5);

    printf("  element_count: OK\n");
}

void test_buffer_size_for_shapes() {
    // Verify compute_buffer_size matches expected GPU allocation for various shapes/dtypes
    // This is the core of AC: "Buffer size computed correctly for all dtypes including quantized block sizes"

    // 2D f32: 4x8 = 32 elements, 32*4 = 128 bytes
    assert(compute_buffer_size(4 * 8, DType::f32) == 128);

    // 2D f16: 4x8 = 32 elements, 32*2 = 64 bytes
    assert(compute_buffer_size(4 * 8, DType::f16) == 64);

    // q8_0: 64 elements = 2 blocks of 32, 2*34 = 68 bytes
    assert(compute_buffer_size(64, DType::q8_0) == 68);

    // q4_k_m: 512 elements = 2 blocks of 256, 2*144 = 288 bytes
    assert(compute_buffer_size(512, DType::q4_k_m) == 288);

    // Non-aligned q8_0: 33 elements -> 2 blocks
    assert(compute_buffer_size(33, DType::q8_0) == 68);

    // Non-aligned q4_k_m: 257 elements -> 2 blocks
    assert(compute_buffer_size(257, DType::q4_k_m) == 288);

    printf("  buffer_size_for_shapes: OK\n");
}

int main() {
    printf("Running tensor tests (standalone)...\n");

    test_dtype_size_bytes();
    test_dtype_block_size();
    test_compute_buffer_size();
    test_element_count();
    test_buffer_size_for_shapes();

    printf("All tensor tests passed.\n");
    printf("\nNote: Tensor::create() and Tensor::from_data() require GPU (Dawn).\n");
    printf("Build test_tensor target with VS dev shell for GPU tests.\n");
    return 0;
}
