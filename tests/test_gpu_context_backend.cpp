#include "../src/gpu_context.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void test_default_backend() {
    auto ctx = create_gpu_context();
    assert(ctx.device != nullptr);
    assert(ctx.queue != nullptr);
    std::printf("  create_gpu_context(Default): PASS\n");
}

static void test_explicit_d3d12() {
    auto ctx = create_gpu_context(GpuBackend::D3D12);
    assert(ctx.device != nullptr);
    assert(ctx.queue != nullptr);
    std::printf("  create_gpu_context(D3D12): PASS\n");
}

static void test_explicit_vulkan() {
    auto ctx = create_gpu_context(GpuBackend::Vulkan);
    assert(ctx.device != nullptr);
    assert(ctx.queue != nullptr);
    std::printf("  create_gpu_context(Vulkan): PASS\n");
}

static void test_env_var_d3d12() {
#ifdef _WIN32
    _putenv_s("BACKPACK_BACKEND", "d3d12");
#else
    setenv("BACKPACK_BACKEND", "d3d12", 1);
#endif
    auto ctx = create_gpu_context();
    assert(ctx.device != nullptr);
    assert(ctx.queue != nullptr);
#ifdef _WIN32
    _putenv_s("BACKPACK_BACKEND", "");
#else
    unsetenv("BACKPACK_BACKEND");
#endif
    std::printf("  BACKPACK_BACKEND=d3d12 override: PASS\n");
}

static void test_env_var_vulkan() {
#ifdef _WIN32
    _putenv_s("BACKPACK_BACKEND", "vulkan");
#else
    setenv("BACKPACK_BACKEND", "vulkan", 1);
#endif
    auto ctx = create_gpu_context();
    assert(ctx.device != nullptr);
    assert(ctx.queue != nullptr);
#ifdef _WIN32
    _putenv_s("BACKPACK_BACKEND", "");
#else
    unsetenv("BACKPACK_BACKEND");
#endif
    std::printf("  BACKPACK_BACKEND=vulkan override: PASS\n");
}

static void test_enum_values_exist() {
    GpuBackend b1 = GpuBackend::Default;
    GpuBackend b2 = GpuBackend::D3D12;
    GpuBackend b3 = GpuBackend::Vulkan;
    assert(b1 != b2);
    assert(b1 != b3);
    assert(b2 != b3);
    std::printf("  GpuBackend enum values distinct: PASS\n");
}

static void test_default_parameter() {
    auto ctx = create_gpu_context();
    assert(ctx.device != nullptr);
    std::printf("  create_gpu_context() default param: PASS\n");
}

int main() {
    std::printf("Running gpu_context backend tests...\n");

    test_enum_values_exist();
    test_default_parameter();
    test_default_backend();
    test_explicit_d3d12();
    test_explicit_vulkan();
    test_env_var_d3d12();
    test_env_var_vulkan();

    std::printf("All gpu_context backend tests PASSED.\n");
    return 0;
}
