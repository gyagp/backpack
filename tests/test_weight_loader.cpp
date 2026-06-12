#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gpu_context.h"
#include "weight_loader.h"

class GGUFBuilder {
public:
    std::vector<uint8_t> buf;

    template<typename T> void write(T val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    void write_string(const std::string& s) {
        write<uint64_t>(s.size());
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void write_tensor_info(const std::string& name, const std::vector<uint64_t>& dims, uint32_t dtype, uint64_t offset) {
        write_string(name);
        write<uint32_t>(static_cast<uint32_t>(dims.size()));
        for (auto d : dims) write<uint64_t>(d);
        write<uint32_t>(dtype);
        write<uint64_t>(offset);
    }
};

struct FakeMmap {
    std::vector<uint8_t> storage;
    const uint8_t* data() const { return storage.data(); }
    size_t size() const { return storage.size(); }
};

void test_gguf_dtype_to_dtype() {
    assert(gguf_dtype_to_dtype(GGUFDType::F32) == DType::f32);
    assert(gguf_dtype_to_dtype(GGUFDType::F16) == DType::f16);
    assert(gguf_dtype_to_dtype(GGUFDType::Q8_0) == DType::q8_0);
    assert(gguf_dtype_to_dtype(GGUFDType::Q4_K_M) == DType::q4_k_m);

    bool threw = false;
    try { gguf_dtype_to_dtype(static_cast<GGUFDType>(999)); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);

    printf("  gguf_dtype_to_dtype: OK\n");
}

GGUFFile build_test_gguf(uint64_t data_offset) {
    GGUFFile gguf;
    gguf.version = 3;
    gguf.tensor_count = 4;
    gguf.metadata_kv_count = 0;
    gguf.data_offset = data_offset;

    // F32 tensor: 2x3 = 6 elements, 24 bytes
    GGUFTensorInfo t0;
    t0.name = "weight_f32";
    t0.ndims = 2;
    t0.dimensions = {2, 3};
    t0.dtype = GGUFDType::F32;
    t0.offset = 0;
    gguf.tensors.push_back(t0);

    // F16 tensor: 4 elements, 8 bytes
    GGUFTensorInfo t1;
    t1.name = "weight_f16";
    t1.ndims = 1;
    t1.dimensions = {4};
    t1.dtype = GGUFDType::F16;
    t1.offset = 32;  // after f32 data, aligned

    gguf.tensors.push_back(t1);

    // Q8_0 tensor: 32 elements = 1 block = 34 bytes
    GGUFTensorInfo t2;
    t2.name = "weight_q8";
    t2.ndims = 1;
    t2.dimensions = {32};
    t2.dtype = GGUFDType::Q8_0;
    t2.offset = 64;
    gguf.tensors.push_back(t2);

    // Q4_K_M tensor: 256 elements = 1 super-block = 144 bytes
    GGUFTensorInfo t3;
    t3.name = "weight_q4km";
    t3.ndims = 2;
    t3.dimensions = {16, 16};
    t3.dtype = GGUFDType::Q4_K_M;
    t3.offset = 128;
    gguf.tensors.push_back(t3);

    return gguf;
}

void test_load_weights_basic(const GpuContext& ctx) {
    const uint64_t data_offset = 256;
    GGUFFile gguf = build_test_gguf(data_offset);

    // Build fake mmap: data_offset header + enough tensor data
    size_t total_size = data_offset + 512;
    MmapFile mmap;
    // We can't use MmapFile directly with synthetic data, so test via
    // building a real binary buffer and parsing it.
    // Instead, test the individual components and verify the function signature compiles.

    // Verify function signature: takes GGUFFile + MmapFile + GpuContext, returns map<string, Tensor>
    // This is a compile-time check - the function exists with the right signature.
    using FnType = std::map<std::string, Tensor>(*)(const GGUFFile&, const MmapFile&, const GpuContext&);
    FnType fn = &load_weights;
    (void)fn;

    printf("  load_weights signature: OK\n");
}

void test_load_weights_with_real_data(const GpuContext& ctx) {
    // Build a synthetic GGUF binary in memory, then parse and load
    GGUFBuilder b;

    // Header
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);   // version
    b.write<uint64_t>(2);   // tensor_count
    b.write<uint64_t>(0);   // metadata_kv_count

    // Tensor info: "t_f32" shape [2, 3], dtype F32, offset 0
    b.write_tensor_info("t_f32", {2, 3}, 0, 0);
    // Tensor info: "t_f16" shape [4], dtype F16, offset 32 (aligned past 24 bytes of f32 data)
    b.write_tensor_info("t_f16", {4}, 1, 32);

    // Align to GGUF_ALIGNMENT (32)
    while (b.buf.size() % GGUF_ALIGNMENT != 0)
        b.buf.push_back(0);

    uint64_t data_offset = b.buf.size();

    // Tensor data for t_f32: 6 floats = 24 bytes
    float f32_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const uint8_t* f32_bytes = reinterpret_cast<const uint8_t*>(f32_data);
    b.buf.insert(b.buf.end(), f32_bytes, f32_bytes + 24);

    // Pad to offset 32
    while (b.buf.size() < data_offset + 32)
        b.buf.push_back(0);

    // Tensor data for t_f16: 4 half-floats = 8 bytes
    uint16_t f16_data[] = {0x3C00, 0x4000, 0x4200, 0x4400};
    const uint8_t* f16_bytes = reinterpret_cast<const uint8_t*>(f16_data);
    b.buf.insert(b.buf.end(), f16_bytes, f16_bytes + 8);

    // Parse the GGUF
    GGUFFile gguf = GGUFParser::parse(b.buf.data(), b.buf.size());
    assert(gguf.tensor_count == 2);
    assert(gguf.tensors.size() == 2);
    assert(gguf.data_offset == data_offset);

    // Create a temporary file for MmapFile
    // Write binary to temp file, mmap it, then load weights
    const char* tmppath = "test_weight_loader_tmp.gguf";
    {
        FILE* f = fopen(tmppath, "wb");
        assert(f != nullptr);
        fwrite(b.buf.data(), 1, b.buf.size(), f);
        fclose(f);
    }

    MmapFile mmap(tmppath);
    assert(mmap.is_open());

    auto weights = load_weights(gguf, mmap, ctx);

    // Verify we got 2 tensors
    assert(weights.size() == 2);

    // Check t_f32
    assert(weights.count("t_f32") == 1);
    auto& wf32 = weights.at("t_f32");
    assert(wf32.shape.size() == 2);
    assert(wf32.shape[0] == 2);
    assert(wf32.shape[1] == 3);
    assert(wf32.dtype == DType::f32);
    assert(wf32.buffer != nullptr);
    assert(wf32.buffer.GetSize() == 24);

    // Check t_f16
    assert(weights.count("t_f16") == 1);
    auto& wf16 = weights.at("t_f16");
    assert(wf16.shape.size() == 1);
    assert(wf16.shape[0] == 4);
    assert(wf16.dtype == DType::f16);
    assert(wf16.buffer != nullptr);
    assert(wf16.buffer.GetSize() == 8);

    // Cleanup
    remove(tmppath);

    printf("  load_weights with real data: OK\n");
}

void test_load_weights_q8_0(const GpuContext& ctx) {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(1);
    b.write<uint64_t>(0);

    // Q8_0: 32 elements = 1 block of 34 bytes
    b.write_tensor_info("t_q8", {32}, static_cast<uint32_t>(GGUFDType::Q8_0), 0);

    while (b.buf.size() % GGUF_ALIGNMENT != 0)
        b.buf.push_back(0);

    uint64_t data_offset = b.buf.size();

    // 34 bytes of fake Q8_0 data
    for (int i = 0; i < 34; i++)
        b.buf.push_back(static_cast<uint8_t>(i));

    GGUFFile gguf = GGUFParser::parse(b.buf.data(), b.buf.size());

    const char* tmppath = "test_wl_q8.gguf";
    {
        FILE* f = fopen(tmppath, "wb");
        fwrite(b.buf.data(), 1, b.buf.size(), f);
        fclose(f);
    }

    MmapFile mmap(tmppath);
    auto weights = load_weights(gguf, mmap, ctx);

    assert(weights.size() == 1);
    assert(weights.count("t_q8") == 1);
    auto& t = weights.at("t_q8");
    assert(t.shape.size() == 1);
    assert(t.shape[0] == 32);
    assert(t.dtype == DType::q8_0);
    assert(t.buffer != nullptr);
    assert(t.buffer.GetSize() == 34);

    remove(tmppath);
    printf("  load_weights Q8_0: OK\n");
}

void test_load_weights_q4_k_m(const GpuContext& ctx) {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(1);
    b.write<uint64_t>(0);

    // Q4_K_M: 256 elements = 1 super-block of 144 bytes
    b.write_tensor_info("t_q4km", {16, 16}, static_cast<uint32_t>(GGUFDType::Q4_K_M), 0);

    while (b.buf.size() % GGUF_ALIGNMENT != 0)
        b.buf.push_back(0);

    uint64_t data_offset = b.buf.size();

    // 144 bytes of fake Q4_K_M data
    for (int i = 0; i < 144; i++)
        b.buf.push_back(static_cast<uint8_t>(i % 256));

    GGUFFile gguf = GGUFParser::parse(b.buf.data(), b.buf.size());

    const char* tmppath = "test_wl_q4km.gguf";
    {
        FILE* f = fopen(tmppath, "wb");
        fwrite(b.buf.data(), 1, b.buf.size(), f);
        fclose(f);
    }

    MmapFile mmap(tmppath);
    auto weights = load_weights(gguf, mmap, ctx);

    assert(weights.size() == 1);
    assert(weights.count("t_q4km") == 1);
    auto& t = weights.at("t_q4km");
    assert(t.shape.size() == 2);
    assert(t.shape[0] == 16);
    assert(t.shape[1] == 16);
    assert(t.dtype == DType::q4_k_m);
    assert(t.buffer != nullptr);
    assert(t.buffer.GetSize() == 144);

    remove(tmppath);
    printf("  load_weights Q4_K_M: OK\n");
}

void test_load_weights_empty(const GpuContext& ctx) {
    GGUFBuilder b;
    b.write<uint32_t>(GGUF_MAGIC);
    b.write<uint32_t>(3);
    b.write<uint64_t>(0);  // no tensors
    b.write<uint64_t>(0);

    while (b.buf.size() % GGUF_ALIGNMENT != 0)
        b.buf.push_back(0);

    GGUFFile gguf = GGUFParser::parse(b.buf.data(), b.buf.size());

    const char* tmppath = "test_wl_empty.gguf";
    {
        FILE* f = fopen(tmppath, "wb");
        fwrite(b.buf.data(), 1, b.buf.size(), f);
        fclose(f);
    }

    MmapFile mmap(tmppath);
    auto weights = load_weights(gguf, mmap, ctx);
    assert(weights.empty());

    remove(tmppath);
    printf("  load_weights empty: OK\n");
}

int main() {
    printf("Running weight_loader tests...\n");

    // Pure logic tests
    test_gguf_dtype_to_dtype();

    // GPU tests
    printf("Initializing GPU...\n");
    auto ctx = create_gpu_context();

    test_load_weights_basic(ctx);
    test_load_weights_with_real_data(ctx);
    test_load_weights_q8_0(ctx);
    test_load_weights_q4_k_m(ctx);
    test_load_weights_empty(ctx);

    printf("All weight_loader tests passed.\n");
    return 0;
}
