/**
 * test_runner.cpp — C++ unit tests for WGSL kernels and template kernels.
 *
 * Tests individual WGSL compute shaders by dispatching them on the GPU via
 * GPUContext and comparing outputs against CPU reference implementations.
 *
 * Usage:
 *   backpack_kernel_test [--filter <pattern>]
 *
 * Build:
 *   cd gitignore/runtime/build && cmake ../../../runtime && cmake --build . --config Release
 */

#include "gpu_context.h"
#include "wgsl_template.h"
#include "graph_executor.h"  // for TensorDtype enum

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Lightweight test harness ───────────────────────────────────────────────

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double ms;
};

static std::vector<TestResult> g_results;
static std::string g_filter;

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

static uint32_t f32AsU32(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}

static float u32AsF32(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// ─── Simple deterministic RNG (xoshiro128+) ─────────────────────────────────

struct Rng {
    uint32_t s[4];
    Rng(uint32_t seed = 42) {
        s[0] = seed;
        s[1] = seed ^ 0x9E3779B9;
        s[2] = seed ^ 0x6A09E667;
        s[3] = seed ^ 0xBB67AE85;
        for (int i = 0; i < 8; i++) next();  // warm up
    }
    uint32_t next() {
        uint32_t t = s[1] << 9;
        uint32_t result = s[0] + s[3];
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }
    float uniform(float lo = -1.0f, float hi = 1.0f) {
        float u = (float)(next() >> 8) / (float)(1 << 24);
        return lo + u * (hi - lo);
    }
    float randn() {
        // Box-Muller
        float u1 = uniform(1e-6f, 1.0f);
        float u2 = uniform(0.0f, 6.2831853f);
        return sqrtf(-2.0f * logf(u1)) * cosf(u2);
    }
    std::vector<float> randnVec(int n) {
        std::vector<float> v(n);
        for (int i = 0; i < n; i++) v[i] = randn();
        return v;
    }
    std::vector<float> uniformVec(int n, float lo = -1.0f, float hi = 1.0f) {
        std::vector<float> v(n);
        for (int i = 0; i < n; i++) v[i] = uniform(lo, hi);
        return v;
    }
};

// ─── fp16 pack/unpack ───────────────────────────────────────────────────────

static uint16_t f32ToF16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (u >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | frac);
}

static float f16ToF32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (frac == 0) u = sign;
        else {
            exp = 1;
            while (!(frac & 0x400)) { frac <<= 1; exp--; }
            frac &= 0x3FF;
            u = sign | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000 | (frac << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Pack f32 array into u32 array (2 fp16 per u32)
static std::vector<uint32_t> packF16(const std::vector<float>& arr) {
    int n = (int)arr.size();
    int nu32 = (n + 1) / 2;
    std::vector<uint32_t> result(nu32, 0);
    for (int i = 0; i < n; i++) {
        uint16_t h = f32ToF16(arr[i]);
        if (i % 2 == 0)
            result[i / 2] |= (uint32_t)h;
        else
            result[i / 2] |= ((uint32_t)h) << 16;
    }
    return result;
}

// Unpack u32 array to f32 (2 fp16 per u32)
static std::vector<float> unpackF16(const uint32_t* data, int count) {
    std::vector<float> result(count);
    for (int i = 0; i < count; i++) {
        uint32_t word = data[i / 2];
        uint16_t h = (i % 2 == 0) ? (uint16_t)(word & 0xFFFF) : (uint16_t)(word >> 16);
        result[i] = f16ToF32(h);
    }
    return result;
}

// ─── CPU reference functions ────────────────────────────────────────────────

static float refSigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float refSilu(float x) { return x * refSigmoid(x); }
static float refGelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }
static float refSoftplus(float x) { return logf(1.0f + expf(x)); }

using UnaryFn = float(*)(float);

static float refNeg(float x) { return -x; }
static float refSqrt(float x) { return sqrtf(fabsf(x)); }
static float refSin(float x) { return sinf(x); }
static float refCos(float x) { return cosf(x); }
static float refIdentity(float x) { return x; }
static float refErf(float x) { return erff(x); }
static float refRelu(float x) { return x > 0 ? x : 0; }
static float refExp(float x) { return expf(x); }
static float refLog(float x) { return logf(fabsf(x) + 1e-10f); }
static float refAbs(float x) { return fabsf(x); }
static float refFloor(float x) { return floorf(x); }
static float refCeil(float x) { return ceilf(x); }
static float refRound(float x) { return roundf(x); }
static float refTanh(float x) { return tanhf(x); }

struct UnaryOp { int code; const char* name; UnaryFn fn; };
static const UnaryOp UNARY_OPS[] = {
    {0,  "sigmoid",  refSigmoid},
    {1,  "tanh",     refTanh},
    {2,  "neg",      refNeg},
    {3,  "sqrt",     refSqrt},
    {4,  "sin",      refSin},
    {5,  "cos",      refCos},
    {6,  "identity", refIdentity},
    {7,  "gelu",     refGelu},
    {8,  "silu",     refSilu},
    // Note: opcode 9 (erf) not implemented in the WGSL kernel
    {10, "relu",     refRelu},
    {11, "exp",      refExp},
    {12, "log",      refLog},
    {13, "abs",      refAbs},
    {14, "floor",    refFloor},
    {15, "ceil",     refCeil},
    {16, "round",    refRound},
};

// ─── GPU helpers ────────────────────────────────────────────────────────────

static std::string g_kernelDir;  // set in main()

static std::string loadWgsl(const std::string& category, const std::string& name) {
    // Try the given category first
    std::string path = g_kernelDir + "/" + category + "/" + name + ".wgsl";
    std::ifstream f(path);
    if (!f.is_open()) {
        // Search all subdirectories (kernels were reorganized into categories)
        for (auto& entry : fs::directory_iterator(g_kernelDir)) {
            if (!entry.is_directory()) continue;
            path = entry.path().string() + "/" + name + ".wgsl";
            f.open(path);
            if (f.is_open()) break;
        }
        if (!f.is_open()) {
            fprintf(stderr, "  Cannot find kernel: %s/%s.wgsl\n",
                    category.c_str(), name.c_str());
            return "";
        }
    }
    return std::string{std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>()};
}

// Dispatch a WGSL compute kernel and read back one output buffer.
// bindings: list of {binding_idx, buffer_handle, buffer_size}
static int g_pipelineCounter = 0;

static std::vector<uint8_t> dispatchAndReadback(
    GPUContext& gpu,
    const std::string& wgsl,
    const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings,
    uint32_t gx, uint32_t gy, uint32_t gz,
    GPUBuffer outputBuf, uint64_t readSize,
    uint32_t numBindings = 0)
{
    if (numBindings == 0) numBindings = (uint32_t)bindings.size();
    // Use a unique name per call to avoid pipeline cache collisions
    std::string name = "test_pipeline_" + std::to_string(g_pipelineCounter++);
    auto& pl = gpu.getOrCreatePipeline(name, wgsl, numBindings);
    auto bg = gpu.createBindGroup(pl, bindings);
    Dispatch d{pl.pipeline, bg, gx, gy, gz, "test"};
    return gpu.submitAndReadback({d}, outputBuf, readSize);
}

// Create a GPU buffer and write float data
static GPUBuffer makeBuffer(GPUContext& gpu, const std::string& name,
                             const float* data, int count) {
    uint64_t bytes = (uint64_t)count * 4;
    if (bytes == 0) bytes = 4;
    auto buf = gpu.createBuffer(name, bytes);
    if (data && count > 0) gpu.writeBuffer(buf, data, bytes);
    return buf;
}

// Create a GPU buffer and write u32 data
static GPUBuffer makeBufferU32(GPUContext& gpu, const std::string& name,
                                const uint32_t* data, int count) {
    uint64_t bytes = (uint64_t)count * 4;
    if (bytes == 0) bytes = 4;
    auto buf = gpu.createBuffer(name, bytes);
    if (data && count > 0) gpu.writeBuffer(buf, data, bytes);
    return buf;
}

// Create a GPU buffer and write i32 data
static GPUBuffer makeBufferI32(GPUContext& gpu, const std::string& name,
                                const int32_t* data, int count) {
    uint64_t bytes = (uint64_t)count * 4;
    if (bytes == 0) bytes = 4;
    auto buf = gpu.createBuffer(name, bytes);
    if (data && count > 0) gpu.writeBuffer(buf, data, bytes);
    return buf;
}

// Build a params buffer from u32 values
static GPUBuffer makeParams(GPUContext& gpu, const std::string& name,
                             std::initializer_list<uint32_t> vals) {
    std::vector<uint32_t> v(vals);
    return makeBufferU32(gpu, name, v.data(), (int)v.size());
}

// Check if results are close
struct CheckResult {
    bool ok;
    std::string msg;
};

static CheckResult assertClose(const float* actual, const float* expected,
                                int N, float atol = 1e-5f, float rtol = 1e-4f) {
    float maxDiff = 0;
    int maxIdx = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(actual[i] - expected[i]);
        float tol = atol + rtol * fabsf(expected[i]);
        if (diff > tol && diff > maxDiff) {
            maxDiff = diff;
            maxIdx = i;
        }
    }
    if (maxDiff > 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "idx=%d: got=%.6f expected=%.6f diff=%.6f",
                 maxIdx, actual[maxIdx], expected[maxIdx], maxDiff);
        return {false, buf};
    }
    return {true, ""};
}

static CheckResult assertArrayEqual(const uint32_t* actual, const uint32_t* expected, int N) {
    for (int i = 0; i < N; i++) {
        if (actual[i] != expected[i]) {
            char buf[128];
            snprintf(buf, sizeof(buf), "idx=%d: got=%u expected=%u", i, actual[i], expected[i]);
            return {false, buf};
        }
    }
    return {true, ""};
}

// ─── Test runner macro ──────────────────────────────────────────────────────

using TestFn = std::function<CheckResult(GPUContext&)>;

struct TestEntry {
    std::string name;
    TestFn fn;
};

static std::vector<TestEntry> g_tests;

#define TEST(tname) \
    static CheckResult test_##tname(GPUContext& gpu); \
    static struct _Reg_##tname { \
        _Reg_##tname() { g_tests.push_back({#tname, test_##tname}); } \
    } _reg_##tname; \
    static CheckResult test_##tname(GPUContext& gpu)

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL TESTS — Raw WGSL dispatch (from test_kernels.py)
// ═══════════════════════════════════════════════════════════════════════════

// ── Unary elementwise ───────────────────────────────────────────────────────

static CheckResult testUnaryOp(GPUContext& gpu, int opCode, const char* name, UnaryFn refFn) {
    auto tmpl = loadWgsl("shared", "unary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 64;
    Rng rng(42);
    std::vector<float> A(N);
    for (int i = 0; i < N; i++) {
        if (opCode == 3 || opCode == 12)  // sqrt, log: positive
            A[i] = rng.uniform(0.1f, 10.0f);
        else if (opCode == 11)  // exp: moderate range
            A[i] = rng.uniform(-2.0f, 2.0f);
        else
            A[i] = rng.randn();
    }

    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto params = makeParams(gpu, "params", {(uint32_t)N, (uint32_t)opCode});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufC}, {2, params}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 3);

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = refFn(A[i]);

    // GELU uses tanh approximation in shader, needs wider tolerance
    float tol_atol = 2e-5f, tol_rtol = 1e-4f;
    if (opCode == 7) { tol_atol = 5e-4f; tol_rtol = 5e-4f; }  // gelu
    if (opCode == 9) { tol_atol = 1e-3f; tol_rtol = 1e-3f; }  // erf (shader approximation)

    return assertClose((const float*)result.data(), expected.data(), N, tol_atol, tol_rtol);
}

// Register all unary ops as individual tests
#define UNARY_TEST(opname, code, fn) \
    TEST(unary_##opname) { \
        return testUnaryOp(gpu, code, #opname, fn); \
    }

UNARY_TEST(sigmoid, 0, refSigmoid)
UNARY_TEST(tanh, 1, refTanh)
UNARY_TEST(neg, 2, refNeg)
UNARY_TEST(sqrt, 3, refSqrt)
UNARY_TEST(sin, 4, refSin)
UNARY_TEST(cos, 5, refCos)
UNARY_TEST(identity, 6, refIdentity)
UNARY_TEST(gelu, 7, refGelu)
UNARY_TEST(silu, 8, refSilu)
UNARY_TEST(relu, 10, refRelu)
UNARY_TEST(exp, 11, refExp)
UNARY_TEST(log, 12, refLog)
UNARY_TEST(abs, 13, refAbs)
UNARY_TEST(floor, 14, refFloor)
UNARY_TEST(ceil, 15, refCeil)
UNARY_TEST(round, 16, refRound)
// softplus is opcode 18 in the shader (17 maps to cast which isn't tested)
TEST(unary_softplus) {
    auto tmpl = loadWgsl("shared", "unary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int N = 64;
    Rng rng(42);
    auto A = rng.randnVec(N);
    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto params = makeParams(gpu, "params", {(uint32_t)N, 18u});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufC}, {2, params}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 3);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = refSoftplus(A[i]);
    return assertClose((const float*)result.data(), expected.data(), N, 2e-5f, 1e-4f);
}

// ── Binary elementwise ──────────────────────────────────────────────────────

static CheckResult testBinaryOp(GPUContext& gpu, int opCode, const char* name,
                                 float(*refFn)(float, float)) {
    auto tmpl = loadWgsl("shared", "binary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 64;
    Rng rng(42);
    auto A = rng.randnVec(N);
    auto B = rng.uniformVec(N, 0.5f, 2.0f);  // positive for div
    uint32_t paramsData[] = {(uint32_t)N, (uint32_t)opCode, (uint32_t)N, (uint32_t)N};

    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufB = makeBuffer(gpu, "B", B.data(), N);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufC}, {3, bufP}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 4);

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = refFn(A[i], B[i]);

    return assertClose((const float*)result.data(), expected.data(), N, 1e-5f, 1e-5f);
}

static float refAdd(float a, float b) { return a + b; }
static float refSub(float a, float b) { return a - b; }
static float refMul(float a, float b) { return a * b; }
static float refDiv(float a, float b) { return a / b; }

TEST(binary_add) { return testBinaryOp(gpu, 0, "add", refAdd); }
TEST(binary_sub) { return testBinaryOp(gpu, 1, "sub", refSub); }
TEST(binary_mul) { return testBinaryOp(gpu, 2, "mul", refMul); }
TEST(binary_div) { return testBinaryOp(gpu, 3, "div", refDiv); }

TEST(binary_broadcast) {
    auto tmpl = loadWgsl("shared", "binary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 16;
    std::vector<float> A(N);
    for (int i = 0; i < N; i++) A[i] = (float)i;
    float B[] = {10.0f};
    uint32_t paramsData[] = {(uint32_t)N, 0u, (uint32_t)N, 1u};

    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufB = makeBuffer(gpu, "B", B, 1);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufC}, {3, bufP}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 4);

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = A[i] + 10.0f;
    return assertClose((const float*)result.data(), expected.data(), N, 1e-6f, 0);
}

// ── Softmax ─────────────────────────────────────────────────────────────────

TEST(softmax) {
    auto tmpl = loadWgsl("shared", "softmax");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int rows = 4, cols = 8;
    const int N = rows * cols;
    Rng rng(42);
    auto X = rng.randnVec(N);

    auto bufX = makeBuffer(gpu, "X", X.data(), N);
    auto bufY = makeBuffer(gpu, "Y", nullptr, N);
    auto params = makeParams(gpu, "params", {(uint32_t)rows, (uint32_t)cols});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, params}},
        ceilDiv(rows, 256), 1, 1, bufY, N * 4, 3);

    // CPU reference
    std::vector<float> expected(N);
    for (int r = 0; r < rows; r++) {
        float maxVal = -1e30f;
        for (int c = 0; c < cols; c++)
            maxVal = std::max(maxVal, X[r * cols + c]);
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            expected[r * cols + c] = expf(X[r * cols + c] - maxVal);
            sum += expected[r * cols + c];
        }
        for (int c = 0; c < cols; c++)
            expected[r * cols + c] /= sum;
    }

    return assertClose((const float*)result.data(), expected.data(), N, 1e-5f, 0);
}

// ── Scale ───────────────────────────────────────────────────────────────────

TEST(scale) {
    auto tmpl = loadWgsl("shared", "scale");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 32;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)(i + 1);
    float scaleVal = 0.5f;

    auto bufD = makeBuffer(gpu, "data", data.data(), N);
    auto params = makeParams(gpu, "params", {(uint32_t)N, f32AsU32(scaleVal)});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufD}, {1, params}},
        ceilDiv(N, 512), 1, 1, bufD, N * 4, 2);

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = data[i] * scaleVal;
    return assertClose((const float*)result.data(), expected.data(), N, 1e-6f, 0);
}

// ── MatMul f32 ──────────────────────────────────────────────────────────────

TEST(matmul_f32) {
    auto tmpl = loadWgsl("shared", "matmul");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int M = 4, N = 8, K = 16;
    Rng rng(42);
    auto A = rng.randnVec(M * K);
    auto B = rng.randnVec(K * N);

    auto bufA = makeBuffer(gpu, "A", A.data(), M * K);
    auto bufB = makeBuffer(gpu, "B", B.data(), K * N);
    auto bufC = makeBuffer(gpu, "C", nullptr, M * N);
    auto params = makeParams(gpu, "params", {(uint32_t)M, (uint32_t)N, (uint32_t)K});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufC}, {3, params}},
        ceilDiv(N, 32), ceilDiv(M, 16), 1, bufC, M * N * 4, 4);

    // CPU reference: C = A @ B
    std::vector<float> expected(M * N, 0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                expected[m * N + n] += A[m * K + k] * B[k * N + n];

    return assertClose((const float*)result.data(), expected.data(), M * N, 1e-3f, 1e-4f);
}

// ── Gather ──────────────────────────────────────────────────────────────────

TEST(gather) {
    auto wgsl = loadWgsl("shared", "gather");
    if (wgsl.empty()) return {false, "cannot load kernel"};

    const int nRows = 8, sliceSize = 4, nIdx = 3;
    std::vector<uint32_t> data(nRows * sliceSize);
    for (int i = 0; i < nRows * sliceSize; i++) data[i] = (uint32_t)i;
    int32_t indices[] = {2, 0, 5};
    uint32_t paramsData[] = {(uint32_t)nIdx, (uint32_t)sliceSize, (uint32_t)sliceSize};

    int total = nIdx * sliceSize;
    auto bufData = makeBufferU32(gpu, "Data", data.data(), nRows * sliceSize);
    auto bufIdx = makeBufferI32(gpu, "Indices", indices, nIdx);
    auto bufOut = makeBufferU32(gpu, "Out", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 3);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufData}, {1, bufIdx}, {2, bufOut}, {3, bufP}},
        ceilDiv(total, 256), 1, 1, bufOut, total * 4, 4);

    // Expected: rows [2, 0, 5] concatenated
    std::vector<uint32_t> expected;
    for (int idx : {2, 0, 5})
        for (int j = 0; j < sliceSize; j++)
            expected.push_back(data[idx * sliceSize + j]);

    return assertArrayEqual((const uint32_t*)result.data(), expected.data(), total);
}

// ── Transpose 2D ────────────────────────────────────────────────────────────

TEST(transpose_2d) {
    auto wgsl = loadWgsl("shared", "transpose");
    if (wgsl.empty()) return {false, "cannot load kernel"};

    const int M = 3, N = 4;
    const int total = M * N;
    std::vector<float> X(total);
    for (int i = 0; i < total; i++) X[i] = (float)i;

    // Transpose [3,4] → [4,3]: perm=[1,0]
    // out_strides=[3,1], in_strides=[1,4]
    uint32_t paramsData[] = {(uint32_t)total, 2, 0, 0, 3, 1, 1, 4};

    auto bufX = makeBufferU32(gpu, "X", (const uint32_t*)X.data(), total);
    auto bufY = makeBufferU32(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 8);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 256), 1, 1, bufY, total * 4, 3);

    // Expected: transpose
    std::vector<float> expected(total);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            expected[j * M + i] = X[i * N + j];

    return assertArrayEqual((const uint32_t*)result.data(),
                            (const uint32_t*)expected.data(), total);
}

// ── Slice 1D ────────────────────────────────────────────────────────────────

TEST(slice_1d) {
    auto wgsl = loadWgsl("shared", "slice");
    if (wgsl.empty()) return {false, "cannot load kernel"};

    const int inN = 10, outN = 5;
    std::vector<float> X(inN);
    for (int i = 0; i < inN; i++) X[i] = (float)i;

    // Slice [10] with start=2, step=1, length=5
    uint32_t paramsData[] = {(uint32_t)outN, 1, 0, 0,
                             1,     // out_stride[0]
                             1,     // in_stride[0]
                             2,     // start[0]
                             1};    // step[0]

    auto bufX = makeBufferU32(gpu, "X", (const uint32_t*)X.data(), inN);
    auto bufY = makeBufferU32(gpu, "Y", nullptr, outN);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 8);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(outN, 256), 1, 1, bufY, outN * 4, 3);

    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    return assertArrayEqual((const uint32_t*)result.data(),
                            (const uint32_t*)expected.data(), outN);
}

// ── Expand ──────────────────────────────────────────────────────────────────

TEST(expand_broadcast) {
    auto tmpl = loadWgsl("shared", "expand");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int total = 12;  // [1,4] → [3,4]
    float X[] = {1.0f, 2.0f, 3.0f, 4.0f};
    // out_strides=[4,1], in_dims=[1,4], in_strides=[4,1]
    uint32_t paramsData[] = {(uint32_t)total, 2, 0, 0, 4, 1, 1, 4, 4, 1};

    auto bufX = makeBuffer(gpu, "X", X, 4);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 10);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 512), 1, 1, bufY, total * 4, 3);

    std::vector<float> expected = {1,2,3,4, 1,2,3,4, 1,2,3,4};
    return assertClose((const float*)result.data(), expected.data(), total, 1e-6f, 0);
}

// ── Conv2D simple ───────────────────────────────────────────────────────────

TEST(conv2d_simple) {
    auto tmpl = loadWgsl("shared", "conv2d");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int batch = 1, C_in = 1, H_in = 4, W_in = 4;
    const int C_out = 1, KH = 3, KW = 3;
    const int H_out = 2, W_out = 2;
    const int group = 1;

    Rng rng(42);
    auto X = rng.randnVec(batch * C_in * H_in * W_in);
    auto W = rng.randnVec(C_out * (C_in / group) * KH * KW);
    float bias[] = {0.5f};

    uint32_t paramsData[] = {
        (uint32_t)batch, (uint32_t)C_in, (uint32_t)H_in, (uint32_t)W_in,
        (uint32_t)C_out, (uint32_t)KH, (uint32_t)KW,
        0, 0, 1, 1,  // pad_h, pad_w, stride_h, stride_w
        (uint32_t)H_out, (uint32_t)W_out, (uint32_t)group, 1, 1  // dil_h, dil_w
    };
    int total = batch * C_out * H_out * W_out;

    auto bufX = makeBuffer(gpu, "X", X.data(), (int)X.size());
    auto bufW = makeBuffer(gpu, "W", W.data(), (int)W.size());
    auto bufB = makeBuffer(gpu, "Bias", bias, 1);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 16);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufW}, {2, bufB}, {3, bufY}, {4, bufP}},
        ceilDiv(total, 256), 1, 1, bufY, total * 4, 5);

    // CPU reference
    std::vector<float> expected(total);
    for (int co = 0; co < C_out; co++)
        for (int oh = 0; oh < H_out; oh++)
            for (int ow = 0; ow < W_out; ow++) {
                float val = 0;
                for (int ci = 0; ci < C_in / group; ci++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            val += X[ci * H_in * W_in + (oh + kh) * W_in + (ow + kw)]
                                   * W[co * (C_in / group) * KH * KW + ci * KH * KW + kh * KW + kw];
                expected[co * H_out * W_out + oh * W_out + ow] = val + bias[co];
            }

    return assertClose((const float*)result.data(), expected.data(), total, 1e-4f, 0);
}

// ── Conv2D depthwise ────────────────────────────────────────────────────────

TEST(conv2d_depthwise) {
    auto tmpl = loadWgsl("shared", "conv2d");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int batch = 1, C = 4, H = 6, W = 1;
    const int KH = 3, KW = 1, group = 4;
    const int H_out = 4, W_out = 1;

    Rng rng(42);
    auto X = rng.randnVec(batch * C * H * W);
    auto Wt = rng.randnVec(C * 1 * KH * KW);
    std::vector<float> bias(C, 0.0f);

    uint32_t paramsData[] = {
        (uint32_t)batch, (uint32_t)C, (uint32_t)H, (uint32_t)W,
        (uint32_t)C, (uint32_t)KH, (uint32_t)KW,
        0, 0, 1, 1,
        (uint32_t)H_out, (uint32_t)W_out, (uint32_t)group, 1, 1
    };
    int total = batch * C * H_out * W_out;

    auto bufX = makeBuffer(gpu, "X", X.data(), (int)X.size());
    auto bufW = makeBuffer(gpu, "W", Wt.data(), (int)Wt.size());
    auto bufB = makeBuffer(gpu, "Bias", bias.data(), C);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 16);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufW}, {2, bufB}, {3, bufY}, {4, bufP}},
        ceilDiv(total, 256), 1, 1, bufY, total * 4, 5);

    // CPU reference
    std::vector<float> expected(total, 0);
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < H_out; oh++) {
            float val = 0;
            for (int kh = 0; kh < KH; kh++)
                val += X[c * H * W + (oh + kh) * W] * Wt[c * KH * KW + kh * KW];
            expected[c * H_out * W_out + oh * W_out] = val;
        }

    return assertClose((const float*)result.data(), expected.data(), total, 1e-5f, 0);
}

// ── Rotary embedding ────────────────────────────────────────────────────────

TEST(rotary_embedding) {
    auto tmpl = loadWgsl("shared", "rotary_embedding");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    // Rotary embedding uses ${T} markers
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int headDim = 8, total = headDim;
    Rng rng(42);
    auto X = rng.randnVec(total);

    const int half = headDim / 2;
    std::vector<float> cosCache(half, 1.0f);  // cos=1 for pos 0
    std::vector<float> sinCache(half, 0.0f);  // sin=0 for pos 0
    int32_t posIds[] = {0};

    // params: total, head_dim, interleaved, nPosIds, seqLen, ...
    uint32_t paramsData[] = {(uint32_t)total, (uint32_t)headDim, 0, 1, 1, 0, 0, 0};

    auto bufX = makeBuffer(gpu, "X", X.data(), total);
    auto bufPos = makeBufferI32(gpu, "pos_ids", posIds, 1);
    auto bufCos = makeBuffer(gpu, "cos_cache", cosCache.data(), half);
    auto bufSin = makeBuffer(gpu, "sin_cache", sinCache.data(), half);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 8);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufPos}, {2, bufCos}, {3, bufSin}, {4, bufY}, {5, bufP}},
        ceilDiv(total, 256), 1, 1, bufY, total * 4, 6);

    // At position 0 with cos=1, sin=0, output should equal input
    return assertClose((const float*)result.data(), X.data(), total, 1e-5f, 0);
}

// ── Resize nearest ──────────────────────────────────────────────────────────

TEST(resize_nearest) {
    auto tmpl = loadWgsl("shared", "resize_nearest");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    float X[] = {1, 2, 3, 4};
    const int total = 16;  // 2x2 → 4x4
    uint32_t paramsData[] = {1, 1, 2, 2, 4, 4, 0, 0};

    auto bufX = makeBuffer(gpu, "X", X, 4);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 8);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 512), 1, 1, bufY, total * 4, 3);

    float expected[] = {1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4};
    return assertClose((const float*)result.data(), expected, total, 1e-6f, 0);
}

// ── Where/select ────────────────────────────────────────────────────────────

TEST(where_select) {
    auto tmpl = loadWgsl("shared", "where_select");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 8;
    uint8_t condBytes[] = {1, 0, 1, 0, 1, 1, 0, 0};
    uint32_t condU32[2];
    memcpy(condU32, condBytes, 8);

    std::vector<float> A(N, 10.0f);
    std::vector<float> B(N, 20.0f);
    uint32_t paramsData[] = {(uint32_t)N, (uint32_t)N, (uint32_t)N, (uint32_t)N};

    auto bufCond = makeBufferU32(gpu, "Cond", condU32, 2);
    auto bufA = makeBuffer(gpu, "X", A.data(), N);
    auto bufB = makeBuffer(gpu, "Y", B.data(), N);
    auto bufOut = makeBuffer(gpu, "Out", nullptr, N);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufCond}, {1, bufA}, {2, bufB}, {3, bufOut}, {4, bufP}},
        ceilDiv(N, 512), 1, 1, bufOut, N * 4, 5);

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = condBytes[i] ? 10.0f : 20.0f;
    return assertClose((const float*)result.data(), expected.data(), N, 1e-6f, 0);
}

// ── Equal ───────────────────────────────────────────────────────────────────

TEST(equal_op) {
    auto tmpl = loadWgsl("shared", "equal_op");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    // equal_op may or may not have templates
    std::string wgsl = tmpl;
    if (tmpl.find("${T}") != std::string::npos)
        wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 8;
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float B[] = {1, 0, 3, 0, 5, 0, 7, 0};
    uint32_t paramsData[] = {(uint32_t)N, (uint32_t)N};

    int outSize = (N + 3) / 4;
    auto bufA = makeBuffer(gpu, "A", A, N);
    auto bufB = makeBuffer(gpu, "B", B, N);
    auto bufOut = makeBufferU32(gpu, "Out", nullptr, outSize);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 2);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufOut}, {3, bufP}},
        ceilDiv(N, 256), 1, 1, bufOut, outSize * 4, 4);

    // Unpack: each byte is 0 or 1
    uint8_t* resultBytes = (uint8_t*)result.data();
    uint8_t expectedBytes[] = {1, 0, 1, 0, 1, 0, 1, 0};
    for (int i = 0; i < N; i++) {
        if (resultBytes[i] != expectedBytes[i]) {
            char buf[128];
            snprintf(buf, sizeof(buf), "idx=%d: got=%d expected=%d", i, resultBytes[i], expectedBytes[i]);
            return {false, buf};
        }
    }
    return {true, ""};
}

// ── Gemm ────────────────────────────────────────────────────────────────────

TEST(gemm) {
    auto tmpl = loadWgsl("shared", "gemm");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int M = 2, N = 4, K = 8;
    Rng rng(42);
    auto A = rng.randnVec(M * K);
    auto B = rng.randnVec(K * N);
    auto bias = rng.randnVec(N);

    auto bufA = makeBuffer(gpu, "A", A.data(), M * K);
    auto bufB = makeBuffer(gpu, "B", B.data(), K * N);
    auto bufBias = makeBuffer(gpu, "Bias", bias.data(), N);
    auto bufY = makeBuffer(gpu, "Y", nullptr, M * N);
    auto params = makeParams(gpu, "params", {(uint32_t)M, (uint32_t)N, (uint32_t)K});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufBias}, {3, bufY}, {4, params}},
        ceilDiv(N, 16), ceilDiv(M, 16), 1, bufY, M * N * 4, 5);

    // Y = A @ B + Bias
    std::vector<float> expected(M * N, 0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++)
                expected[m * N + n] += A[m * K + k] * B[k * N + n];
            expected[m * N + n] += bias[n];
        }

    return assertClose((const float*)result.data(), expected.data(), M * N, 1e-3f, 1e-4f);
}

// ── Embed/Gather ────────────────────────────────────────────────────────────

TEST(embed_gather) {
    auto tmpl = loadWgsl("shared", "embed_gather");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int vocab = 16, dim = 8;
    const int tokenId = 3;
    Rng rng(42);
    auto table = rng.randnVec(vocab * dim);
    int32_t ids[] = {tokenId};

    auto bufTable = makeBuffer(gpu, "EmbeddingTable", table.data(), vocab * dim);
    auto bufId = makeBufferI32(gpu, "TokenId", ids, 1);
    auto bufOut = makeBuffer(gpu, "X", nullptr, dim);
    auto params = makeParams(gpu, "params", {(uint32_t)dim});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufTable}, {1, bufId}, {2, bufOut}, {3, params}},
        ceilDiv(dim, 512), 1, 1, bufOut, dim * 4, 4);

    return assertClose((const float*)result.data(),
                       table.data() + tokenId * dim, dim, 1e-6f, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE KERNEL TESTS — f32 and f16 variants (from test_template_kernels.py)
// ═══════════════════════════════════════════════════════════════════════════

// ── Template binary elementwise f32 ─────────────────────────────────────────

static CheckResult testTemplateBinaryF32(GPUContext& gpu, int opCode, const char* name,
                                          float(*refFn)(float, float)) {
    auto tmpl = loadWgsl("shared", "binary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int N = 64;
    Rng rng(42);
    auto A = rng.randnVec(N);
    auto B = rng.uniformVec(N, 0.5f, 2.0f);
    uint32_t paramsData[] = {(uint32_t)N, (uint32_t)opCode, (uint32_t)N, (uint32_t)N};

    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufB = makeBuffer(gpu, "B", B.data(), N);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufC}, {3, bufP}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 4);

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = refFn(A[i], B[i]);
    return assertClose((const float*)result.data(), expected.data(), N, 1e-5f, 0);
}

TEST(tmpl_binary_add_f32) { return testTemplateBinaryF32(gpu, 0, "add", refAdd); }
TEST(tmpl_binary_sub_f32) { return testTemplateBinaryF32(gpu, 1, "sub", refSub); }
TEST(tmpl_binary_mul_f32) { return testTemplateBinaryF32(gpu, 2, "mul", refMul); }
TEST(tmpl_binary_div_f32) { return testTemplateBinaryF32(gpu, 3, "div", refDiv); }

// ── Template binary elementwise f16 ─────────────────────────────────────────

static CheckResult testTemplateBinaryF16(GPUContext& gpu, int opCode, const char* name,
                                          float(*refFn)(float, float)) {
    auto tmpl = loadWgsl("shared", "binary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);

    const int N = 64;
    Rng rng(42);
    auto A = rng.uniformVec(N, -2.0f, 2.0f);
    auto B = rng.uniformVec(N, 0.5f, 2.0f);
    uint32_t paramsData[] = {(uint32_t)N, (uint32_t)opCode, (uint32_t)N, (uint32_t)N};
    int nu32 = ceilDiv(N, 2);

    auto packedA = packF16(A);
    auto packedB = packF16(B);

    auto bufA = makeBufferU32(gpu, "A", packedA.data(), nu32);
    auto bufB = makeBufferU32(gpu, "B", packedB.data(), nu32);
    auto bufC = makeBufferU32(gpu, "C", nullptr, nu32);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufC}, {3, bufP}},
        ceilDiv(N, 512), 1, 1, bufC, nu32 * 4, 4);

    auto actual = unpackF16((const uint32_t*)result.data(), N);

    // Reference in fp16 precision
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) {
        float a16 = f16ToF32(f32ToF16(A[i]));
        float b16 = f16ToF32(f32ToF16(B[i]));
        expected[i] = refFn(a16, b16);
    }
    return assertClose(actual.data(), expected.data(), N, 0.05f, 0.01f);
}

TEST(tmpl_binary_add_f16) { return testTemplateBinaryF16(gpu, 0, "add", refAdd); }
TEST(tmpl_binary_mul_f16) { return testTemplateBinaryF16(gpu, 2, "mul", refMul); }

// ── Template unary elementwise f32 ──────────────────────────────────────────

TEST(tmpl_unary_sigmoid_f32) {
    auto tmpl = loadWgsl("shared", "unary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int N = 64;
    Rng rng(42);
    auto A = rng.randnVec(N);
    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto params = makeParams(gpu, "params", {(uint32_t)N, 0u});  // sigmoid=0
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufC}, {2, params}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 3);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = refSigmoid(A[i]);
    return assertClose((const float*)result.data(), expected.data(), N, 2e-5f, 1e-4f);
}

TEST(tmpl_unary_relu_f32) {
    auto tmpl = loadWgsl("shared", "unary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int N = 64;
    Rng rng(42);
    auto A = rng.randnVec(N);
    auto bufA = makeBuffer(gpu, "A", A.data(), N);
    auto bufC = makeBuffer(gpu, "C", nullptr, N);
    auto params = makeParams(gpu, "params", {(uint32_t)N, 10u});  // relu=10
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufC}, {2, params}},
        ceilDiv(N, 512), 1, 1, bufC, N * 4, 3);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = refRelu(A[i]);
    return assertClose((const float*)result.data(), expected.data(), N, 2e-5f, 1e-4f);
}

// ── Template unary f16 ──────────────────────────────────────────────────────

static CheckResult testTemplateUnaryF16(GPUContext& gpu, int opCode, UnaryFn refFn) {
    auto tmpl = loadWgsl("shared", "unary_elementwise");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);

    const int N = 64;
    Rng rng(42);
    auto A = rng.uniformVec(N, -2.0f, 2.0f);
    int nu32 = ceilDiv(N, 2);
    auto packed = packF16(A);

    auto bufA = makeBufferU32(gpu, "A", packed.data(), nu32);
    auto bufC = makeBufferU32(gpu, "C", nullptr, nu32);
    auto params = makeParams(gpu, "params", {(uint32_t)N, (uint32_t)opCode});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufC}, {2, params}},
        ceilDiv(N, 512), 1, 1, bufC, nu32 * 4, 3);

    auto actual = unpackF16((const uint32_t*)result.data(), N);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) {
        float a16 = f16ToF32(f32ToF16(A[i]));
        expected[i] = refFn(a16);
    }
    return assertClose(actual.data(), expected.data(), N, 0.05f, 0.01f);
}

TEST(tmpl_unary_sigmoid_f16) { return testTemplateUnaryF16(gpu, 0, refSigmoid); }
TEST(tmpl_unary_relu_f16) { return testTemplateUnaryF16(gpu, 10, refRelu); }

// ── Template softmax f32 ────────────────────────────────────────────────────

TEST(tmpl_softmax_f32) {
    auto tmpl = loadWgsl("shared", "softmax");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int rows = 4, cols = 8, N = rows * cols;
    Rng rng(42);
    auto X = rng.randnVec(N);

    auto bufX = makeBuffer(gpu, "X", X.data(), N);
    auto bufY = makeBuffer(gpu, "Y", nullptr, N);
    auto params = makeParams(gpu, "params", {(uint32_t)rows, (uint32_t)cols});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, params}},
        ceilDiv(rows, 256), 1, 1, bufY, N * 4, 3);

    std::vector<float> expected(N);
    for (int r = 0; r < rows; r++) {
        float mx = -1e30f;
        for (int c = 0; c < cols; c++) mx = std::max(mx, X[r * cols + c]);
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            expected[r * cols + c] = expf(X[r * cols + c] - mx);
            sum += expected[r * cols + c];
        }
        for (int c = 0; c < cols; c++) expected[r * cols + c] /= sum;
    }
    return assertClose((const float*)result.data(), expected.data(), N, 1e-5f, 0);
}

// ── Template softmax f16 ────────────────────────────────────────────────────

TEST(tmpl_softmax_f16) {
    auto tmpl = loadWgsl("shared", "softmax");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);

    const int rows = 4, cols = 8, N = rows * cols;
    int nu32 = ceilDiv(N, 2);
    Rng rng(42);
    auto X = rng.randnVec(N);
    auto packed = packF16(X);

    auto bufX = makeBufferU32(gpu, "X", packed.data(), nu32);
    auto bufY = makeBufferU32(gpu, "Y", nullptr, nu32);
    auto params = makeParams(gpu, "params", {(uint32_t)rows, (uint32_t)cols});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, params}},
        ceilDiv(rows, 256), 1, 1, bufY, nu32 * 4, 3);

    auto actual = unpackF16((const uint32_t*)result.data(), N);

    // Reference in fp16 precision
    std::vector<float> expected(N);
    for (int r = 0; r < rows; r++) {
        float mx = -1e30f;
        for (int c = 0; c < cols; c++) {
            float v = f16ToF32(f32ToF16(X[r * cols + c]));
            mx = std::max(mx, v);
        }
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            float v = f16ToF32(f32ToF16(X[r * cols + c]));
            expected[r * cols + c] = expf(v - mx);
            sum += expected[r * cols + c];
        }
        for (int c = 0; c < cols; c++) expected[r * cols + c] /= sum;
    }
    return assertClose(actual.data(), expected.data(), N, 0.05f, 0.02f);
}

// ── Template layer norm f32 ─────────────────────────────────────────────────

TEST(tmpl_layer_norm_f32) {
    auto tmpl = loadWgsl("shared", "layer_norm");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);

    const int nRows = 3, N = 8;
    const int total = nRows * N;
    float eps = 1e-5f;
    Rng rng(42);
    auto X = rng.randnVec(total);
    auto W = rng.randnVec(N);
    auto B = rng.randnVec(N);

    auto bufX = makeBuffer(gpu, "X", X.data(), total);
    auto bufW = makeBuffer(gpu, "W", W.data(), N);
    auto bufB = makeBuffer(gpu, "B", B.data(), N);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto params = makeParams(gpu, "params", {(uint32_t)N, (uint32_t)nRows, f32AsU32(eps)});

    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufW}, {2, bufB}, {3, bufY}, {4, params}},
        ceilDiv(nRows, 256), 1, 1, bufY, total * 4, 5);

    // CPU reference
    std::vector<float> expected(total);
    for (int r = 0; r < nRows; r++) {
        float mean = 0;
        for (int c = 0; c < N; c++) mean += X[r * N + c];
        mean /= N;
        float var = 0;
        for (int c = 0; c < N; c++) var += (X[r * N + c] - mean) * (X[r * N + c] - mean);
        var /= N;
        for (int c = 0; c < N; c++)
            expected[r * N + c] = (X[r * N + c] - mean) / sqrtf(var + eps) * W[c] + B[c];
    }
    return assertClose((const float*)result.data(), expected.data(), total, 1e-4f, 0);
}

// ── Template scale f32 ──────────────────────────────────────────────────────

TEST(tmpl_scale_f32) {
    auto tmpl = loadWgsl("shared", "scale");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int N = 32;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)(i + 1);
    float scaleVal = 0.5f;
    auto bufD = makeBuffer(gpu, "data", data.data(), N);
    auto params = makeParams(gpu, "params", {(uint32_t)N, f32AsU32(scaleVal)});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufD}, {1, params}},
        ceilDiv(N, 512), 1, 1, bufD, N * 4, 2);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = data[i] * scaleVal;
    return assertClose((const float*)result.data(), expected.data(), N, 1e-6f, 0);
}

// ── Template scale f16 ──────────────────────────────────────────────────────

TEST(tmpl_scale_f16) {
    auto tmpl = loadWgsl("shared", "scale");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);
    const int N = 32;
    int nu32 = ceilDiv(N, 2);
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)(i + 1) * 0.1f;
    float scaleVal = 0.5f;
    auto packed = packF16(data);
    auto bufD = makeBufferU32(gpu, "data", packed.data(), nu32);
    auto params = makeParams(gpu, "params", {(uint32_t)N, f32AsU32(scaleVal)});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufD}, {1, params}},
        ceilDiv(N, 512), 1, 1, bufD, nu32 * 4, 2);
    auto actual = unpackF16((const uint32_t*)result.data(), N);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) {
        float d16 = f16ToF32(f32ToF16(data[i]));
        expected[i] = d16 * scaleVal;
    }
    return assertClose(actual.data(), expected.data(), N, 0.01f, 0);
}

// ── Template expand f32 ─────────────────────────────────────────────────────

TEST(tmpl_expand_f32) {
    auto tmpl = loadWgsl("shared", "expand");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    float X[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const int total = 12;
    uint32_t paramsData[] = {(uint32_t)total, 2, 0, 0, 4, 1, 1, 4, 4, 1};
    auto bufX = makeBuffer(gpu, "X", X, 4);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 10);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 512), 1, 1, bufY, total * 4, 3);
    float expected[] = {1,2,3,4, 1,2,3,4, 1,2,3,4};
    return assertClose((const float*)result.data(), expected, total, 1e-6f, 0);
}

// ── Template expand f16 ─────────────────────────────────────────────────────

TEST(tmpl_expand_f16) {
    auto tmpl = loadWgsl("shared", "expand");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);
    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
    const int total = 12;
    int nu32_out = ceilDiv(total, 2);
    int nu32_in = ceilDiv(4, 2);
    uint32_t paramsData[] = {(uint32_t)total, 2, 0, 0, 4, 1, 1, 4, 4, 1};
    auto packedX = packF16(X);
    auto bufX = makeBufferU32(gpu, "X", packedX.data(), nu32_in);
    auto bufY = makeBufferU32(gpu, "Y", nullptr, nu32_out);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 10);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 512), 1, 1, bufY, nu32_out * 4, 3);
    auto actual = unpackF16((const uint32_t*)result.data(), total);
    float expected[] = {1,2,3,4, 1,2,3,4, 1,2,3,4};
    return assertClose(actual.data(), expected, total, 0.01f, 0);
}

// ── Template resize nearest f32 ─────────────────────────────────────────────

TEST(tmpl_resize_nearest_f32) {
    auto tmpl = loadWgsl("shared", "resize_nearest");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    float X[] = {1, 2, 3, 4};
    const int total = 16;
    uint32_t paramsData[] = {1, 1, 2, 2, 4, 4, 0, 0};
    auto bufX = makeBuffer(gpu, "X", X, 4);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 8);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 512), 1, 1, bufY, total * 4, 3);
    float expected[] = {1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4};
    return assertClose((const float*)result.data(), expected, total, 1e-6f, 0);
}

// ── Template resize nearest f16 ─────────────────────────────────────────────

TEST(tmpl_resize_nearest_f16) {
    auto tmpl = loadWgsl("shared", "resize_nearest");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);
    std::vector<float> X = {1, 2, 3, 4};
    const int total = 16;
    int nu32_out = ceilDiv(total, 2);
    uint32_t paramsData[] = {1, 1, 2, 2, 4, 4, 0, 0};
    auto packedX = packF16(X);
    auto bufX = makeBufferU32(gpu, "X", packedX.data(), ceilDiv(4, 2));
    auto bufY = makeBufferU32(gpu, "Y", nullptr, nu32_out);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 8);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufY}, {2, bufP}},
        ceilDiv(total, 512), 1, 1, bufY, nu32_out * 4, 3);
    auto actual = unpackF16((const uint32_t*)result.data(), total);
    float expected[] = {1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4};
    return assertClose(actual.data(), expected, total, 0.01f, 0);
}

// ── Template where select f32 ───────────────────────────────────────────────

TEST(tmpl_where_select_f32) {
    auto tmpl = loadWgsl("shared", "where_select");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int N = 8;
    uint8_t condBytes[] = {1, 0, 1, 0, 1, 1, 0, 0};
    uint32_t condU32[2]; memcpy(condU32, condBytes, 8);
    std::vector<float> A(N, 10.0f), B(N, 20.0f);
    uint32_t paramsData[] = {(uint32_t)N, (uint32_t)N, (uint32_t)N, (uint32_t)N};
    auto bufCond = makeBufferU32(gpu, "Cond", condU32, 2);
    auto bufA = makeBuffer(gpu, "X", A.data(), N);
    auto bufB = makeBuffer(gpu, "Y", B.data(), N);
    auto bufOut = makeBuffer(gpu, "Out", nullptr, N);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufCond}, {1, bufA}, {2, bufB}, {3, bufOut}, {4, bufP}},
        ceilDiv(N, 512), 1, 1, bufOut, N * 4, 5);
    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) expected[i] = condBytes[i] ? 10.0f : 20.0f;
    return assertClose((const float*)result.data(), expected.data(), N, 1e-6f, 0);
}

// ── Template embed gather f32 ───────────────────────────────────────────────

TEST(tmpl_embed_gather_f32) {
    auto tmpl = loadWgsl("shared", "embed_gather");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int vocab = 10, E = 16, tokenId = 3;
    Rng rng(42);
    auto table = rng.randnVec(vocab * E);
    int32_t ids[] = {tokenId};
    auto bufTable = makeBuffer(gpu, "EmbeddingTable", table.data(), vocab * E);
    auto bufId = makeBufferI32(gpu, "TokenId", ids, 1);
    auto bufOut = makeBuffer(gpu, "X", nullptr, E);
    auto params = makeParams(gpu, "params", {(uint32_t)E});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufTable}, {1, bufId}, {2, bufOut}, {3, params}},
        ceilDiv(E, 512), 1, 1, bufOut, E * 4, 4);
    return assertClose((const float*)result.data(),
                       table.data() + tokenId * E, E, 1e-6f, 0);
}

// ── Template embed gather f16 ───────────────────────────────────────────────

TEST(tmpl_embed_gather_f16) {
    auto tmpl = loadWgsl("shared", "embed_gather");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);
    const int vocab = 10, E = 16, tokenId = 3;
    Rng rng(42);
    auto table = rng.randnVec(vocab * E);
    int32_t ids[] = {tokenId};
    int nu32_table = ceilDiv(vocab * E, 2);
    int nu32_out = ceilDiv(E, 2);
    auto packedTable = packF16(table);
    auto bufTable = makeBufferU32(gpu, "EmbeddingTable", packedTable.data(), nu32_table);
    auto bufId = makeBufferI32(gpu, "TokenId", ids, 1);
    auto bufOut = makeBufferU32(gpu, "X", nullptr, nu32_out);
    auto params = makeParams(gpu, "params", {(uint32_t)E});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufTable}, {1, bufId}, {2, bufOut}, {3, params}},
        ceilDiv(E, 512), 1, 1, bufOut, nu32_out * 4, 4);
    auto actual = unpackF16((const uint32_t*)result.data(), E);
    std::vector<float> expected(E);
    for (int i = 0; i < E; i++)
        expected[i] = f16ToF32(f32ToF16(table[tokenId * E + i]));
    return assertClose(actual.data(), expected.data(), E, 0.01f, 0);
}

// ── Template matmul f32 ─────────────────────────────────────────────────────

TEST(tmpl_matmul_f32) {
    auto tmpl = loadWgsl("shared", "matmul");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int M = 4, N = 8, K = 16;
    Rng rng(42);
    auto A = rng.randnVec(M * K);
    auto B = rng.randnVec(K * N);
    auto bufA = makeBuffer(gpu, "A", A.data(), M * K);
    auto bufB = makeBuffer(gpu, "B", B.data(), K * N);
    auto bufC = makeBuffer(gpu, "C", nullptr, M * N);
    auto params = makeParams(gpu, "params", {(uint32_t)M, (uint32_t)N, (uint32_t)K});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufC}, {3, params}},
        ceilDiv(N, 32), ceilDiv(M, 16), 1, bufC, M * N * 4, 4);
    std::vector<float> expected(M * N, 0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                expected[m * N + n] += A[m * K + k] * B[k * N + n];
    return assertClose((const float*)result.data(), expected.data(), M * N, 1e-3f, 0);
}

// ── Template gemm f32 ───────────────────────────────────────────────────────

TEST(tmpl_gemm_f32) {
    auto tmpl = loadWgsl("shared", "gemm");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int M = 4, N = 8, K = 16;
    Rng rng(42);
    auto A = rng.randnVec(M * K);
    auto B = rng.randnVec(N * K);  // B is [N,K] transposed
    auto Bias = rng.randnVec(N);
    uint32_t paramsData[] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 1u};
    auto bufA = makeBuffer(gpu, "A", A.data(), M * K);
    auto bufB = makeBuffer(gpu, "B", B.data(), N * K);
    auto bufBias = makeBuffer(gpu, "Bias", Bias.data(), N);
    auto bufY = makeBuffer(gpu, "Y", nullptr, M * N);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufA}, {1, bufB}, {2, bufBias}, {3, bufY}, {4, bufP}},
        ceilDiv(N, 16), ceilDiv(M, 16), 1, bufY, M * N * 4, 5);
    // Y = A @ B^T + Bias
    std::vector<float> expected(M * N, 0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++)
                expected[m * N + n] += A[m * K + k] * B[n * K + k];
            expected[m * N + n] += Bias[n];
        }
    return assertClose((const float*)result.data(), expected.data(), M * N, 1e-3f, 0);
}

// ── Template conv2d f32 ─────────────────────────────────────────────────────

TEST(tmpl_conv2d_f32) {
    auto tmpl = loadWgsl("shared", "conv2d");
    if (tmpl.empty()) return {false, "cannot load kernel"};
    auto wgsl = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
    const int batch = 1, C_in = 1, H_in = 4, W_in = 4;
    const int C_out = 1, KH = 3, KW = 3, H_out = 2, W_out = 2;
    Rng rng(42);
    auto X = rng.randnVec(batch * C_in * H_in * W_in);
    auto W = rng.randnVec(C_out * C_in * KH * KW);
    float bias[] = {0.5f};
    int total = batch * C_out * H_out * W_out;
    uint32_t paramsData[] = {
        (uint32_t)batch, (uint32_t)C_in, (uint32_t)H_in, (uint32_t)W_in,
        (uint32_t)C_out, (uint32_t)KH, (uint32_t)KW,
        0, 0, 1, 1, (uint32_t)H_out, (uint32_t)W_out, 1, 1, 1
    };
    auto bufX = makeBuffer(gpu, "X", X.data(), (int)X.size());
    auto bufW = makeBuffer(gpu, "W", W.data(), (int)W.size());
    auto bufB = makeBuffer(gpu, "Bias", bias, 1);
    auto bufY = makeBuffer(gpu, "Y", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 16);
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufW}, {2, bufB}, {3, bufY}, {4, bufP}},
        ceilDiv(total, 256), 1, 1, bufY, total * 4, 5);
    std::vector<float> expected(total);
    for (int co = 0; co < C_out; co++)
        for (int oh = 0; oh < H_out; oh++)
            for (int ow = 0; ow < W_out; ow++) {
                float val = 0;
                for (int ci = 0; ci < C_in; ci++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            val += X[ci * H_in * W_in + (oh + kh) * W_in + (ow + kw)]
                                   * W[co * C_in * KH * KW + ci * KH * KW + kh * KW + kw];
                expected[co * H_out * W_out + oh * W_out + ow] = val + bias[co];
            }
    return assertClose((const float*)result.data(), expected.data(), total, 1e-4f, 0);
}

// ── Template marker validation ──────────────────────────────────────────────

static const char* TEMPLATE_NAMES[] = {
    "binary_elementwise", "unary_elementwise", "softmax", "layer_norm",
    "expand", "scale", "resize_nearest", "where_select", "embed_gather",
    "matmul", "conv2d", "gemm",
};

TEST(tmpl_has_markers) {
    for (auto name : TEMPLATE_NAMES) {
        auto tmpl = loadWgsl("shared", name);
        if (tmpl.empty()) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: cannot load", name);
            return {false, buf};
        }
        if (tmpl.find("${T}") == std::string::npos) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: missing ${T} marker", name);
            return {false, buf};
        }
    }
    return {true, ""};
}

TEST(tmpl_f32_instantiation_valid) {
    for (auto name : TEMPLATE_NAMES) {
        auto tmpl = loadWgsl("shared", name);
        if (tmpl.empty()) continue;
        auto f32 = instantiateTemplate(tmpl.c_str(), TensorDtype::Float32);
        if (f32.find("${") != std::string::npos) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: leftover markers in f32", name);
            return {false, buf};
        }
        if (f32.find("array<f32>") == std::string::npos) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: no array<f32> in f32 variant", name);
            return {false, buf};
        }
    }
    return {true, ""};
}

TEST(tmpl_f16_instantiation_valid) {
    for (auto name : TEMPLATE_NAMES) {
        auto tmpl = loadWgsl("shared", name);
        if (tmpl.empty()) continue;
        auto f16 = instantiateTemplate(tmpl.c_str(), TensorDtype::Float16);
        if (f16.find("${") != std::string::npos) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: leftover markers in f16", name);
            return {false, buf};
        }
        if (f16.find("array<u32>") == std::string::npos) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: no array<u32> in f16 variant", name);
            return {false, buf};
        }
        if (f16.find("unpack2x16float") == std::string::npos) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: no unpack2x16float in f16 variant", name);
            return {false, buf};
        }
    }
    return {true, ""};
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--filter" && i + 1 < argc)
            g_filter = argv[++i];
    }

    // Determine kernel directory (relative to executable)
    // Expected layout: gitignore/runtime/build/Release/backpack_kernel_test.exe
    // Kernels at:      runtime/kernels/
    auto exePath = fs::path(argv[0]).parent_path();
    g_kernelDir = (exePath / ".." / ".." / ".." / ".." / "runtime" / "kernels").string();
    if (!fs::exists(g_kernelDir)) {
        // Try in-tree build (runtime/build/Release/)
        g_kernelDir = (exePath / ".." / ".." / "kernels").string();
    }
    if (!fs::exists(g_kernelDir)) {
        fprintf(stderr, "Cannot find kernel directory. Tried:\n  %s\n",
                g_kernelDir.c_str());
        return 1;
    }
    fprintf(stderr, "Kernel dir: %s\n", g_kernelDir.c_str());

    // Init GPU
    GPUContext gpu;
    if (!gpu.init(WGPUBackendType_Vulkan)) {
        if (!gpu.init(WGPUBackendType_D3D12)) {
            fprintf(stderr, "Failed to init GPU\n");
            return 1;
        }
    }
    fprintf(stderr, "GPU: %s (%s)\n", gpu.adapterName.c_str(),
            gpu.adapterDescription.c_str());

    // Run tests
    int passed = 0, failed = 0, skipped = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (auto& test : g_tests) {
        if (!g_filter.empty() && test.name.find(g_filter) == std::string::npos) {
            skipped++;
            continue;
        }

        auto start = std::chrono::steady_clock::now();
        auto result = test.fn(gpu);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (result.ok) {
            passed++;
            fprintf(stderr, "  PASS  %-40s (%.1fms)\n", test.name.c_str(), ms);
        } else {
            failed++;
            fprintf(stderr, "  FAIL  %-40s %s\n", test.name.c_str(), result.msg.c_str());
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    fprintf(stderr, "\n%d passed, %d failed", passed, failed);
    if (skipped > 0) fprintf(stderr, ", %d skipped", skipped);
    fprintf(stderr, " (%.1fs)\n", totalMs / 1000.0);

    return failed > 0 ? 1 : 0;
}
