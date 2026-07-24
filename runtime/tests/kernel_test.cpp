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
#include "gguf_loader.h"
#include "wgsl_shaders.h"

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
static WGPUBackendType g_backend = WGPUBackendType_Vulkan;

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

static std::string useRepeatedQkHeadLayout(const char* source) {
    std::string result(source);
    auto replaceAll=[&](const std::string& grouped,const std::string& repeated){
        size_t pos=0;while((pos=result.find(grouped,pos))!=std::string::npos){
            result.replace(pos,grouped.size(),repeated);pos+=repeated.size();
        }
    };
    replaceAll("head / max(1u, nv / nk)","head % nk");
    replaceAll("head/max(1u,nv/nk)","head%nk");
    return result;
}

static std::string usePortableDeltaScanReduction(const std::string& source) {
    std::string result(source);
    if(auto pos=result.find("enable subgroups;");pos!=std::string::npos)
        result.erase(pos,strlen("enable subgroups;"));
    const auto begin=result.find("var<workgroup>");
    const auto end=begin==std::string::npos?std::string::npos:result.find("@compute",begin);
    if(begin==std::string::npos||end==std::string::npos)return result;
    static const char* portable=R"WGSL(var<workgroup> reduce_scratch: array<f32, 256>;
fn reduce128(x:f32,lane128:u32,pair:u32)->f32 {
    let tid=pair*128u+lane128; reduce_scratch[tid]=x; workgroupBarrier();
    for(var offset=64u;offset>0u;offset=offset/2u){
        if(lane128<offset){reduce_scratch[tid]+=reduce_scratch[tid+offset];}
        workgroupBarrier();
    }
    return reduce_scratch[pair*128u];
}
)WGSL";
    result.replace(begin,end-begin,portable);return result;
}

static std::string q4kOrtRepackedTileTestSource() {
    std::string s(WGSL_ORT_DP4A_MATMUL_EXACT);
    auto all=[&](const std::string&from,const std::string&to){size_t p=0;while((p=s.find(from,p))!=std::string::npos){s.replace(p,from.size(),to);p+=to.size();}};
    auto cut=[&](const std::string&begin,const std::string&end,const std::string&to){auto a=s.find(begin),b=s.find(end,a);if(a==std::string::npos||b==std::string::npos)return false;s.replace(a,b-a,to);return true;};
    all("enable f16;\n","");all("scales_a: array<f16>","scales_a: array<f32>");all("scales_b: array<f16>","scales_b: array<vec2<f32>>");
    all("output: array<vec4<f16>>","output: array<vec4<f32>>");all("alias output_element_t = f16;","alias output_element_t = f32;");all("var<uniform> uniforms: Uniforms;","var<storage, read> uniforms: Uniforms;");
    all("var<workgroup> scale_A : array<output_element_t, tile_size>;","var<workgroup> scale_A : array<vec2<output_element_t>, tile_size>;");all("var<workgroup> scale_B : array<output_element_t, tile_size>;","var<workgroup> scale_B : array<vec2<output_element_t>, tile_size>;");
    all("var own_scale_a: output_element_t","var own_scale_a: vec2<output_element_t>");all("var own_scale_b: output_element_t","var own_scale_b: vec2<output_element_t>");
    all("SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:output_element_t)","SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:vec2<output_element_t>)");all("return output_element_t(mul_precision(local_sum) * mul_precision(scale));","return output_element_t(mul_precision(local_sum) * mul_precision(scale.x) - mul_precision(scale.y));");
    const std::string loaders=R"WGSL(fn sumPacked(v:vec4<u32>)->i32{return dot4I8Packed(v.x,0x01010101u)+dot4I8Packed(v.y,0x01010101u)+dot4I8Packed(v.z,0x01010101u)+dot4I8Packed(v.w,0x01010101u);}
fn loadSHMA(batch:u32,a_global_base:u32,kidx_v:u32,row:u32,col:u32){let ar=a_global_base+row;if(ar>=uniforms.M){return;}let off=ar*uniforms.K16+kidx_v;tile_A[col][row]=input_a[off+col];if(col==0u){let a0=input_a[off];let a1=input_a[off+1u];let xs=scales_a[ar*(uniforms.K/32u)+kidx_v/2u];scale_A[row]=vec2<f32>(xs,xs*f32(sumPacked(a0)+sumPacked(a1)));}}
fn loadSHMB(b_global_base:u32,kidx_v:u32,row:u32,col:u32){let br=b_global_base+row;if(br>=uniforms.N){return;}let v=input_b[br*uniforms.K16+kidx_v+col];tile_B[col][row]=DequantizedFrom4BitsTo8Bits(v,0);if(col==0u){scale_B[row]=scales_b[br*(uniforms.K/32u)+kidx_v/2u];}}

)WGSL";
    if(!cut("fn loadSHMA(","@compute @workgroup_size",loaders))return {};
    return s;
}

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
    uint32_t paramsData[] = {1, (uint32_t)nRows, (uint32_t)sliceSize,
                             (uint32_t)nIdx};

    int total = nIdx * sliceSize;
    auto bufData = makeBufferU32(gpu, "Data", data.data(), nRows * sliceSize);
    auto bufIdx = makeBufferI32(gpu, "Indices", indices, nIdx);
    auto bufOut = makeBufferU32(gpu, "Out", nullptr, total);
    auto bufP = makeBufferU32(gpu, "params", paramsData, 4);

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

TEST(matmul_q4_batched) {
    auto wgsl = loadWgsl("onnx_q4", "matmul_q4_batched");
    if (wgsl.empty()) return {false, "cannot load kernel"};
    const int M = 5, N = 9, K = 256;
    Rng rng(123);
    auto X = rng.randnVec(M * K);
    std::vector<uint32_t> W(N * K / 8);
    for (size_t i = 0; i < W.size(); i++) {
        uint32_t p = 0;
        for (int j = 0; j < 8; j++) p |= uint32_t((i * 7 + j * 3 + 5) & 15) << (4 * j);
        W[i] = p;
    }
    const int blocks = K / 32;
    std::vector<uint16_t> scales16(N * blocks);
    std::vector<uint32_t> scales((scales16.size() + 1) / 2, 0);
    for (size_t i = 0; i < scales16.size(); i++) {
        scales16[i] = f32ToF16(0.01f + 0.001f * float(i % 11));
        scales[i / 2] |= uint32_t(scales16[i]) << (16 * (i & 1));
    }
    auto bufX = makeBuffer(gpu, "X", X.data(), M * K);
    auto bufW = makeBufferU32(gpu, "W", W.data(), (int)W.size());
    auto bufS = makeBufferU32(gpu, "S", scales.data(), (int)scales.size());
    auto bufY = makeBuffer(gpu, "Y", nullptr, M * N);
    auto params = makeParams(gpu, "params", {(uint32_t)M, (uint32_t)N, (uint32_t)K});
    auto result = dispatchAndReadback(gpu, wgsl,
        {{0, bufX}, {1, bufW}, {2, bufS}, {3, bufY}, {4, params}},
        ceilDiv(N, 32), ceilDiv(M, 4), 1, bufY, M * N * 4, 5);
    std::vector<float> expected(M * N, 0.0f);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        for (int b = 0; b < blocks; b++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) amax = std::max(amax, std::abs(X[m*K+b*32+j]));
            float as = amax / 127.0f;
            float ws = f16ToF32(scales16[n * blocks + b]);
            for (int j = 0; j < 32; j++) {
                int aq = as == 0.0f ? 0 : std::max(-127, std::min(127,
                    (int)std::round(X[m*K+b*32+j] / as)));
                int wi = b * 4 + j / 8;
                int wq = int((W[n*(K/8)+wi] >> (4*(j&7))) & 15) - 8;
                expected[m*N+n] += float(aq*wq) * as * ws;
            }
        }
    }
    return assertClose((const float*)result.data(), expected.data(), M*N, 2e-3f, 2e-4f);
}

TEST(qwen35_conv_scan_silu) {
    auto wgsl=loadWgsl("ssm","qwen35_conv_scan_silu");
    if(wgsl.empty()) return {false,"cannot load kernel"};
    const int C=5,K=4,T=7; Rng rng(321);
    auto state=rng.randnVec(C*K), initial=state, x=rng.randnVec(T*C);
    auto w=rng.randnVec(C*K), bias=rng.randnVec(C), expected=std::vector<float>(T*C);
    for(int c=0;c<C;c++) for(int t=0;t<T;t++) {
        float acc=bias[c];
        for(int k=0;k<K-1;k++){state[c*K+k]=state[c*K+k+1];acc+=w[c*K+k]*state[c*K+k];}
        state[c*K+K-1]=x[t*C+c];acc+=w[c*K+K-1]*x[t*C+c];
        expected[t*C+c]=acc/(1.0f+std::exp(-acc));
    }
    auto bs=makeBuffer(gpu,"S",initial.data(),C*K),bx=makeBuffer(gpu,"X",x.data(),T*C);
    auto bw=makeBuffer(gpu,"W",w.data(),C*K),bb=makeBuffer(gpu,"B",bias.data(),C);
    auto by=makeBuffer(gpu,"Y",nullptr,T*C),p=makeParams(gpu,"P",{C,K,T});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bs},{1,bx},{2,bw},{3,bb},{4,by},{5,p}},
        ceilDiv(C,256),1,1,by,T*C*4,6);
    return assertClose((const float*)result.data(),expected.data(),T*C,2e-5f,2e-5f);
}

TEST(delta_net_scan_x2) {
    auto wgsl=loadWgsl("ssm","delta_net_scan_x2");
    if(wgsl.empty()) return {false,"cannot load kernel"};
    if(gpu.adapterName.find("AMD")!=std::string::npos)
        wgsl=usePortableDeltaScanReduction(wgsl);
    // Exercise grouped Q/K sharing: consecutive V heads share one Q/K head.
    const int T=3,NV=2,NK=1,DK=128,DV=2; Rng rng(654);
    auto q=rng.randnVec(T*NK*DK),k=rng.randnVec(T*NK*DK),v=rng.randnVec(T*NV*DV);
    auto beta=rng.randnVec(T*NV),gate=rng.randnVec(T*NV),state=rng.randnVec(NV*DK*DV);
    for(float& b:beta)b=1.0f/(1.0f+std::exp(-b)); for(float& g:gate)g=-std::abs(g)*0.1f;
    auto ref=state;std::vector<float> expected(T*NV*DV);float qs=1.0f/std::sqrt(float(DK));
    for(int t=0;t<T;t++)for(int h=0;h<NV;h++)for(int vi=0;vi<DV;vi++){
        int kh=h/(NV/NK),sb=h*DK*DV;float gh=std::exp(gate[t*NV+h]),pred=0;
        for(int d=0;d<DK;d++)pred+=gh*ref[sb+d*DV+vi]*k[(t*NK+kh)*DK+d];
        float delta=(v[(t*NV+h)*DV+vi]-pred)*beta[t*NV+h],out=0;
        for(int d=0;d<DK;d++){float sn=gh*ref[sb+d*DV+vi]+k[(t*NK+kh)*DK+d]*delta;ref[sb+d*DV+vi]=sn;out+=sn*q[(t*NK+kh)*DK+d]*qs;}
        expected[(t*NV+h)*DV+vi]=out;
    }
    auto bq=makeBuffer(gpu,"Q",q.data(),q.size()),bk=makeBuffer(gpu,"K",k.data(),k.size());
    auto bv=makeBuffer(gpu,"V",v.data(),v.size()),bb=makeBuffer(gpu,"B",beta.data(),beta.size());
    auto bg=makeBuffer(gpu,"G",gate.data(),gate.size()),bs=makeBuffer(gpu,"S",state.data(),state.size());
    auto by=makeBuffer(gpu,"Y",nullptr,T*NV*DV),p=makeParams(gpu,"P",{NV,NK,DK,DV,T});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bq},{1,bk},{2,bv},{3,bb},{4,bg},{5,bs},{6,by},{7,p}},
        NV,ceilDiv(DV,2),1,by,T*NV*DV*4,8);
    return assertClose((const float*)result.data(),expected.data(),T*NV*DV,3e-4f,3e-4f);
}

TEST(delta_net_scan_x2_repeated_heads) {
    // GGUF/llama.cpp repeats Q/K heads as [k0,k1,k0,k1], unlike the ONNX
    // grouped layout [k0,k0,k1,k1]. NV=4,NK=2 distinguishes both mappings.
    auto wgsl=useRepeatedQkHeadLayout(WGSL_DELTA_NET_SCAN_X2);
    if(gpu.adapterName.find("AMD")!=std::string::npos)
        wgsl=usePortableDeltaScanReduction(wgsl);
    const int T=3,NV=4,NK=2,DK=128,DV=2; Rng rng(656);
    auto q=rng.randnVec(T*NK*DK),k=rng.randnVec(T*NK*DK),v=rng.randnVec(T*NV*DV);
    auto beta=rng.randnVec(T*NV),gate=rng.randnVec(T*NV),state=rng.randnVec(NV*DK*DV);
    for(float&b:beta)b=1/(1+std::exp(-b));for(float&g:gate)g=-std::abs(g)*.1f;
    auto ref=state;std::vector<float>expected(T*NV*DV);float qs=1/std::sqrt(float(DK));
    for(int t=0;t<T;t++)for(int h=0;h<NV;h++)for(int vi=0;vi<DV;vi++){
        int kh=h%NK,sb=h*DK*DV;float gh=std::exp(gate[t*NV+h]),pred=0;
        for(int d=0;d<DK;d++)pred+=gh*ref[sb+d*DV+vi]*k[(t*NK+kh)*DK+d];
        float delta=(v[(t*NV+h)*DV+vi]-pred)*beta[t*NV+h],out=0;
        for(int d=0;d<DK;d++){float sn=gh*ref[sb+d*DV+vi]+k[(t*NK+kh)*DK+d]*delta;ref[sb+d*DV+vi]=sn;out+=sn*q[(t*NK+kh)*DK+d]*qs;}
        expected[(t*NV+h)*DV+vi]=out;
    }
    auto bq=makeBuffer(gpu,"Q",q.data(),q.size()),bk=makeBuffer(gpu,"K",k.data(),k.size()),bv=makeBuffer(gpu,"V",v.data(),v.size()),bb=makeBuffer(gpu,"B",beta.data(),beta.size()),bg=makeBuffer(gpu,"G",gate.data(),gate.size()),bs=makeBuffer(gpu,"S",state.data(),state.size()),by=makeBuffer(gpu,"Y",nullptr,T*NV*DV),p=makeParams(gpu,"P",{NV,NK,DK,DV,T});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bq},{1,bk},{2,bv},{3,bb},{4,bg},{5,bs},{6,by},{7,p}},NV,ceilDiv(DV,2),1,by,T*NV*DV*4,8);
    return assertClose((const float*)result.data(),expected.data(),expected.size(),3e-4f,3e-4f);
}

TEST(delta_net_decode_x2_repeated_heads) {
    // This explicit shared-memory reduction is the portable AMD wave64 path.
    auto wgsl=useRepeatedQkHeadLayout(WGSL_DELTA_NET_DECODE_X2);
    const int NV=4,NK=2,DK=128,DV=2; Rng rng(657);
    auto q=rng.randnVec(NK*DK),k=rng.randnVec(NK*DK),v=rng.randnVec(NV*DV);
    auto beta=rng.randnVec(NV),gate=rng.randnVec(NV),state=rng.randnVec(NV*DK*DV);
    for(float&b:beta)b=1/(1+std::exp(-b));for(float&g:gate)g=-std::abs(g)*.1f;
    auto ref=state;std::vector<float>expected(NV*DV);float qs=1/std::sqrt(float(DK));
    for(int h=0;h<NV;h++)for(int vi=0;vi<DV;vi++){
        int kh=h%NK,sb=h*DK*DV;float gh=std::exp(gate[h]),pred=0;
        for(int d=0;d<DK;d++)pred+=gh*ref[sb+d*DV+vi]*k[kh*DK+d];
        float delta=(v[h*DV+vi]-pred)*beta[h],out=0;
        for(int d=0;d<DK;d++){float sn=gh*ref[sb+d*DV+vi]+k[kh*DK+d]*delta;ref[sb+d*DV+vi]=sn;out+=sn*q[kh*DK+d]*qs;}
        expected[h*DV+vi]=out;
    }
    auto bq=makeBuffer(gpu,"Q",q.data(),q.size()),bk=makeBuffer(gpu,"K",k.data(),k.size()),bv=makeBuffer(gpu,"V",v.data(),v.size()),bb=makeBuffer(gpu,"B",beta.data(),beta.size()),bg=makeBuffer(gpu,"G",gate.data(),gate.size()),bs=makeBuffer(gpu,"S",state.data(),state.size()),by=makeBuffer(gpu,"Y",nullptr,NV*DV),p=makeParams(gpu,"P",{NV,NK,DK,DV});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bq},{1,bk},{2,bv},{3,bb},{4,bg},{5,bs},{6,by},{7,p}},NV,ceilDiv(DV,2),1,by,NV*DV*4,8);
    return assertClose((const float*)result.data(),expected.data(),expected.size(),3e-4f,3e-4f);
}

TEST(delta_net_scan_x4) {
    std::string wgsl=WGSL_DELTA_NET_SCAN_X4;const int T=3,NV=1,NK=1,DK=128,DV=5;Rng rng(655);
    if(gpu.adapterName.find("AMD")!=std::string::npos)wgsl=usePortableDeltaScanReduction(wgsl);
    auto q=rng.randnVec(T*NK*DK),k=rng.randnVec(T*NK*DK),v=rng.randnVec(T*NV*DV),beta=rng.randnVec(T*NV),gate=rng.randnVec(T*NV),state=rng.randnVec(NV*DK*DV);
    for(float&b:beta)b=1.0f/(1.0f+std::exp(-b));for(float&g:gate)g=-std::abs(g)*.1f;auto ref=state;std::vector<float>expected(T*NV*DV);float qs=1/std::sqrt(float(DK));
    for(int t=0;t<T;t++)for(int vi=0;vi<DV;vi++){float gh=std::exp(gate[t]),pred=0;for(int d=0;d<DK;d++)pred+=gh*ref[d*DV+vi]*k[t*DK+d];float delta=(v[t*DV+vi]-pred)*beta[t],out=0;for(int d=0;d<DK;d++){float sn=gh*ref[d*DV+vi]+k[t*DK+d]*delta;ref[d*DV+vi]=sn;out+=sn*q[t*DK+d]*qs;}expected[t*DV+vi]=out;}
    auto bq=makeBuffer(gpu,"Q",q.data(),q.size()),bk=makeBuffer(gpu,"K",k.data(),k.size()),bv=makeBuffer(gpu,"V",v.data(),v.size()),bb=makeBuffer(gpu,"B",beta.data(),beta.size()),bg=makeBuffer(gpu,"G",gate.data(),gate.size()),bs=makeBuffer(gpu,"S",state.data(),state.size()),by=makeBuffer(gpu,"Y",nullptr,T*NV*DV),p=makeParams(gpu,"P",{NV,NK,DK,DV,T});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bq},{1,bk},{2,bv},{3,bb},{4,bg},{5,bs},{6,by},{7,p}},NV,ceilDiv(DV,4),1,by,T*NV*DV*4,8);return assertClose((const float*)result.data(),expected.data(),expected.size(),3e-4f,3e-4f);
}

TEST(qwen35_split_qkv_l2_batched) {
    auto wgsl=loadWgsl("ssm","qwen35_split_qkv_l2_batched");if(wgsl.empty())return{false,"cannot load kernel"};
    const int T=2,NK=1,NV=1,DK=128,DV=128,C=2*NK*DK+NV*DV;Rng rng(991);auto in=rng.randnVec(T*C);
    auto bi=makeBuffer(gpu,"I",in.data(),in.size()),bq=makeBuffer(gpu,"Q",nullptr,T*NK*DK),bk=makeBuffer(gpu,"K",nullptr,T*NK*DK),bv=makeBuffer(gpu,"V",nullptr,T*NV*DV);float eps=1e-6f;uint32_t eb;memcpy(&eb,&eps,4);auto p=makeParams(gpu,"P",{NK,NV,DK,DV,eb,T});
    auto qr=dispatchAndReadback(gpu,wgsl,{{0,bi},{1,bq},{2,bk},{3,bv},{4,p}},3,1,T,bq,T*NK*DK*4,5);std::vector<float>e(T*DK);
    for(int t=0;t<T;t++){double ss=0;for(int d=0;d<DK;d++)ss+=double(in[t*C+d])*in[t*C+d];float inv=1.0f/std::max(std::sqrt(float(ss)),eps);for(int d=0;d<DK;d++)e[t*DK+d]=in[t*C+d]*inv;}
    return assertClose((const float*)qr.data(),e.data(),e.size(),2e-5f,2e-5f);
}

TEST(qwen35_conv_scan_split_l2) {
    auto wgsl=loadWgsl("ssm","qwen35_conv_scan_split_l2");if(wgsl.empty())return{false,"cannot load kernel"};
    const int T=3,NK=1,NV=1,DK=128,DV=128,CK=4,C=2*DK+DV;Rng rng(818);auto state=rng.randnVec(C*CK),ref=state,x=rng.randnVec(T*C),w=rng.randnVec(C*CK),bias=rng.randnVec(C);std::vector<float>eq(T*DK);
    for(int t=0;t<T;t++){std::vector<float>q(DK);double ss=0;for(int d=0;d<DK;d++){int c=d;float a=bias[c];for(int j=0;j<CK-1;j++){ref[c*CK+j]=ref[c*CK+j+1];a+=w[c*CK+j]*ref[c*CK+j];}ref[c*CK+CK-1]=x[t*C+c];a+=w[c*CK+CK-1]*x[t*C+c];q[d]=a/(1+std::exp(-a));ss+=double(q[d])*q[d];}float inv=1/std::max(std::sqrt(float(ss)),1e-6f);for(int d=0;d<DK;d++)eq[t*DK+d]=q[d]*inv;}
    auto bst=makeBuffer(gpu,"S",state.data(),state.size()),bx=makeBuffer(gpu,"X",x.data(),x.size()),bw=makeBuffer(gpu,"W",w.data(),w.size()),bb=makeBuffer(gpu,"B",bias.data(),bias.size()),bq=makeBuffer(gpu,"Q",nullptr,T*DK),bk=makeBuffer(gpu,"K",nullptr,T*DK),bv=makeBuffer(gpu,"V",nullptr,T*DV);float eps=1e-6f;uint32_t eb;memcpy(&eb,&eps,4);auto p=makeParams(gpu,"P",{NK,NV,DK,DV,CK,eb,T});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bst},{1,bx},{2,bw},{3,bb},{4,bq},{5,bk},{6,bv},{7,p}},3,1,1,bq,T*DK*4,8);return assertClose((const float*)r.data(),eq.data(),eq.size(),3e-5f,3e-5f);
}

TEST(gemma_sandwich_attn_batched) {
    auto wgsl = loadWgsl("norm", "gemma_sandwich_attn_batched");
    if (wgsl.empty()) return {false, "cannot load kernel"};
    const int M = 3, N = 1536;
    Rng rng(456);
    auto X = rng.randnVec(M*N); auto A = rng.randnVec(M*N);
    auto PW = rng.randnVec(N); auto NW = rng.randnVec(N);
    auto expectedX = X; std::vector<float> expectedY(M*N), rstd(M);
    const float eps = 1e-6f;
    for (int m = 0; m < M; m++) {
        double ss = 0; for (int i = 0; i < N; i++) ss += double(A[m*N+i])*A[m*N+i];
        float ar = 1.0f / std::sqrt(float(ss/N) + eps);
        ss = 0;
        for (int i = 0; i < N; i++) {
            float v = expectedX[m*N+i] + A[m*N+i]*ar*PW[i];
            expectedX[m*N+i] = v; ss += double(v)*v;
        }
        float xr = 1.0f / std::sqrt(float(ss/N) + eps); rstd[m] = xr;
        for (int i = 0; i < N; i++) expectedY[m*N+i] = expectedX[m*N+i]*xr*NW[i];
    }
    auto bx=makeBuffer(gpu,"X",X.data(),M*N), ba=makeBuffer(gpu,"A",A.data(),M*N);
    auto bp=makeBuffer(gpu,"PW",PW.data(),N), bn=makeBuffer(gpu,"NW",NW.data(),N);
    auto by=makeBuffer(gpu,"Y",nullptr,M*N), br=makeBuffer(gpu,"R",nullptr,M);
    uint32_t eb; memcpy(&eb,&eps,4);
    auto pp=makeParams(gpu,"P",{(uint32_t)N,(uint32_t)N,eb});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,ba},{2,bp},{3,bn},{4,by},{5,br},{6,pp}},
        M,1,1,by,M*N*4,7);
    return assertClose((const float*)result.data(),expectedY.data(),M*N,2e-4f,2e-4f);
}

TEST(q4_gather_batched) {
    auto wgsl=loadWgsl("onnx_q4","q4_gather_batched");
    if(wgsl.empty()) return {false,"cannot load kernel"};
    const int M=3,D=256,V=4,blocks=D/32;
    std::vector<uint32_t> W(V*D/8);
    for(size_t i=0;i<W.size();i++) for(int j=0;j<8;j++) W[i]|=uint32_t((i+j*5)&15)<<(4*j);
    std::vector<uint16_t> sh(V*blocks); std::vector<uint32_t> S((sh.size()+1)/2,0);
    for(size_t i=0;i<sh.size();i++){sh[i]=f32ToF16(.02f+.001f*(i%7));S[i/2]|=uint32_t(sh[i])<<(16*(i&1));}
    int32_t toks[M]={2,-1,3}; float outScale=1.25f; uint32_t sb; memcpy(&sb,&outScale,4);
    auto bw=makeBufferU32(gpu,"W",W.data(),(int)W.size()), bs=makeBufferU32(gpu,"S",S.data(),(int)S.size());
    auto bt=makeBufferI32(gpu,"T",toks,M); auto bo=makeBuffer(gpu,"O",nullptr,M*D);
    auto pp=makeParams(gpu,"P",{M,D,V,sb});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bw},{1,bs},{2,bt},{3,bo},{4,pp}},ceilDiv(M*D,256),1,1,bo,M*D*4,5);
    std::vector<float> expected(M*D);
    for(int m=0;m<M;m++){int t=toks[m]>=0&&toks[m]<V?toks[m]:0;for(int i=0;i<D;i++){
        uint32_t p=W[t*(D/8)+i/8]; int q=int((p>>(4*(i&7)))&15)-8;
        expected[m*D+i]=q*f16ToF32(sh[t*blocks+i/32])*outScale;
    }}
    return assertClose((const float*)result.data(),expected.data(),M*D,1e-6f,0);
}

TEST(ple_project_combine_gemma_shape) {
    const char* wgsl = R"(
@group(0) @binding(0) var<storage, read> Proj: array<f32>;
@group(0) @binding(1) var<storage, read> Norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> TokenSignal: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;
var<workgroup> squares: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let D = P[0]; let i = lid.x; let idx = wid.x * D + i;
    let inRange = i < D;
    var v = 0.0;
    if (inRange) { v = Proj[idx]; }
    squares[i] = v * v;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (i < stride) { squares[i] += squares[i + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride /= 2u;
    }
    if (inRange) {
        let rms = inverseSqrt(squares[0] / f32(D) + bitcast<f32>(P[2]));
        TokenSignal[idx] = (v * rms * Norm[i] + TokenSignal[idx]) * 0.7071067811865476;
    }
}
)";
    const int D=256,L=3,N=D*L; Rng rng(0x504c45u);
    auto proj=rng.uniformVec(N,-3.0f,3.0f), norm=rng.uniformVec(D,0.25f,1.75f);
    auto signal=rng.uniformVec(N,-2.0f,2.0f), expected=signal;
    const float eps=1.0e-6f;
    for(int l=0;l<L;l++){
        double ss=0;for(int i=0;i<D;i++)ss+=double(proj[l*D+i])*proj[l*D+i];
        float rms=1.0f/std::sqrt(float(ss/D)+eps);
        for(int i=0;i<D;i++)expected[l*D+i]=(proj[l*D+i]*rms*norm[i]+signal[l*D+i])*0.7071067811865476f;
    }
    auto bp=makeBuffer(gpu,"ple_proj",proj.data(),N),bn=makeBuffer(gpu,"ple_norm",norm.data(),D);
    auto bs=makeBuffer(gpu,"ple_signal",signal.data(),N);
    auto pp=makeParams(gpu,"ple_params",{D,L,f32AsU32(eps),0u});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bp},{1,bn},{2,bs},{3,pp}},L,1,1,bs,N*4,4);
    return assertClose((const float*)result.data(),expected.data(),N,2e-5f,2e-5f);
}

TEST(ple_q4_asymmetric_gather_gemma_shape) {
    const char* wgsl = R"(
@group(0) @binding(0) var<storage, read> B: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<u32>;
@group(0) @binding(2) var<storage, read> Z: array<u32>;
@group(0) @binding(3) var<storage, read> Token: array<i32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x;let K=P[0];if(i>=K){return;}
    let row=u32(max(Token[0],0));let V=P[2];let D=P[3];
    let layer=i/D;let col=i%D;let packedRow=layer*V+row;
    let byteIndex=packedRow*(D/2u)+col/2u;let word=B[byteIndex/4u];
    let qv=i32((word>>((byteIndex&3u)*8u+(col&1u)*4u))&15u);
    let block=packedRow*(D/32u)+col/32u;let zpByteIndex=block/2u;
    let zpWord=Z[zpByteIndex/4u];let zpByte=(zpWord>>((zpByteIndex&3u)*8u))&255u;
    let zp=i32(select(zpByte&15u,zpByte>>4u,(block&1u)==1u));
    let sp=unpack2x16float(S[block/2u]);let scale=select(sp.x,sp.y,(block&1u)!=0u);
    Y[i]=f32(qv-zp)*scale*bitcast<f32>(P[1]);
}
)";
    const int D=256,L=3,V=5,K=D*L,blocks=D/32,rows=L*V;
    std::vector<uint8_t> wb(rows*D/2),zb(rows*blocks/2);
    std::vector<uint16_t> sh(rows*blocks);std::vector<float> expected(K);
    for(size_t i=0;i<wb.size();i++)wb[i]=uint8_t((i*29u+7u)&255u);
    for(int b=0;b<rows*blocks;b++){
        int zp=(b*3+1)&15;zb[b/2]|=uint8_t(zp<<(4*(b&1)));
        sh[b]=f32ToF16(.01f+.001f*float(b%11));
    }
    std::vector<uint32_t> W((wb.size()+3)/4),Z((zb.size()+3)/4),S((sh.size()+1)/2);
    memcpy(W.data(),wb.data(),wb.size());memcpy(Z.data(),zb.data(),zb.size());
    for(size_t i=0;i<sh.size();i++)S[i/2]|=uint32_t(sh[i])<<(16*(i&1));
    int32_t token=3;float scale=16.0f;
    for(int i=0;i<K;i++){
        int row=(i/D)*V+token,col=i%D,b= row*blocks+col/32;
        int q=(wb[row*D/2+col/2]>>(4*(col&1)))&15;
        int zp=(zb[b/2]>>(4*(b&1)))&15;
        expected[i]=float(q-zp)*f16ToF32(sh[b])*scale;
    }
    auto bw=makeBufferU32(gpu,"ple_w",W.data(),W.size()),bs=makeBufferU32(gpu,"ple_s",S.data(),S.size());
    auto bz=makeBufferU32(gpu,"ple_z",Z.data(),Z.size()),bt=makeBufferI32(gpu,"ple_t",&token,1);
    auto by=makeBuffer(gpu,"ple_y",nullptr,K);auto pp=makeParams(gpu,"ple_gp",{K,f32AsU32(scale),V,D});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bw},{1,bs},{2,bz},{3,bt},{4,by},{5,pp}},ceilDiv(K,256),1,1,by,K*4,6);
    return assertClose((const float*)result.data(),expected.data(),K,1e-6f,0);
}

TEST(q8_gather_batched) {
    auto wgsl=loadWgsl("quant_q8","q8_gather_batched");if(wgsl.empty())return{false,"cannot load kernel"};
    const int T=3,D=64,V=4,B=D/32;std::vector<uint32_t>W(V*D/4);for(size_t i=0;i<W.size();i++)W[i]=uint32_t(i*2654435761u);
    std::vector<uint16_t>sh(V*B);std::vector<uint32_t>S((sh.size()+1)/2);for(size_t i=0;i<sh.size();i++){sh[i]=f32ToF16(.01f+.002f*i);S[i/2]|=uint32_t(sh[i])<<(16*(i&1));}
    int32_t tok[T]={2,-1,3};auto bw=makeBufferU32(gpu,"W",W.data(),W.size()),bs=makeBufferU32(gpu,"S",S.data(),S.size());auto bt=makeBufferI32(gpu,"T",tok,T),bo=makeBuffer(gpu,"O",nullptr,T*D);auto p=makeParams(gpu,"P",{T,D,V});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bw},{1,bs},{2,bt},{3,bo},{4,p}},ceilDiv(T*D,256),1,1,bo,T*D*4,5);std::vector<float>e(T*D);
    for(int t=0;t<T;t++){int id=tok[t]>=0?tok[t]:0;for(int d=0;d<D;d++){int q=int8_t((W[id*(D/4)+d/4]>>(8*(d&3)))&255);e[t*D+d]=q*f16ToF32(sh[id*B+d/32]);}}
    return assertClose((const float*)result.data(),e.data(),T*D,1e-6f,0);
}

TEST(q8_matmul_batched_dp4a) {
    auto wgsl=loadWgsl("quant_q8","q8_matmul_batched_dp4a");if(wgsl.empty())return{false,"cannot load kernel"};
    const int M=5,N=9,K=256,B=K/32;Rng rng(777);auto x=rng.randnVec(M*K),bias=rng.randnVec(N);std::vector<uint32_t>w(N*K/4);for(size_t i=0;i<w.size();i++)w[i]=uint32_t(i*2246822519u);
    std::vector<uint16_t>sh(N*B);std::vector<uint32_t>s((sh.size()+1)/2);for(size_t i=0;i<sh.size();i++){sh[i]=f32ToF16(.003f+.0002f*(i%13));s[i/2]|=uint32_t(sh[i])<<(16*(i&1));}
    auto bx=makeBuffer(gpu,"X",x.data(),x.size()),bw=makeBufferU32(gpu,"W",w.data(),w.size()),bs=makeBufferU32(gpu,"S",s.data(),s.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N);auto p=makeParams(gpu,"P",{K,N,M});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bs},{3,bb},{4,by},{5,p}},ceilDiv(M,4),ceilDiv(N,32),1,by,M*N*4,6);std::vector<float>e(M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float sum=bias[n];for(int b=0;b<B;b++){float am=0;for(int j=0;j<32;j++)am=std::max(am,std::abs(x[m*K+b*32+j]));float as=am/127;for(int j=0;j<32;j++){int aq=as?std::max(-127,std::min(127,int(std::round(x[m*K+b*32+j]/as)))):0;int d=b*32+j;int wq=int8_t((w[n*(K/4)+d/4]>>(8*(d&3)))&255);sum+=aq*wq*as*f16ToF32(sh[n*B+b]);}}e[m*N+n]=sum;}
    return assertClose((const float*)r.data(),e.data(),M*N,3e-4f,3e-4f);
}

TEST(q8_matmul_batched_dp4a_real_smoke) {
    auto wgsl=loadWgsl("quant_q8","q8_matmul_batched_dp4a");if(wgsl.empty())return{false,"cannot load kernel"};
    const int M=1,N=6144,K=2048;std::vector<float>x(M*K,0.25f),bias(N,0.125f);std::vector<uint32_t>w((size_t)N*K/4,0),s((size_t)N*(K/32)/2,0);
    auto bx=makeBuffer(gpu,"X",x.data(),x.size()),bw=makeBufferU32(gpu,"W",w.data(),w.size()),bs=makeBufferU32(gpu,"S",s.data(),s.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bs},{3,bb},{4,by},{5,p}},ceilDiv(M,4),ceilDiv(N,32),1,by,M*N*4,6);
    return assertClose((const float*)r.data(),bias.data(),N,1e-6f,0);
}

static CheckResult testQ8DecodeProjection(GPUContext& gpu, bool fusedNorm) {
    auto wgsl = loadWgsl("quant_q8", fusedNorm ? "q8_matmul_norm" : "q8_matmul");
    if (wgsl.empty()) return {false, "cannot load kernel"};
    const int K = 1536, N = 13, B = K / 32;
    Rng rng(fusedNorm ? 0x4141u : 0x3131u);
    auto x = rng.randnVec(K), bias = rng.uniformVec(N, -0.1f, 0.1f);
    auto norm = rng.uniformVec(K, 0.25f, 2.0f);
    std::vector<uint32_t> w((size_t)N * K / 4);
    for (size_t i = 0; i < w.size(); ++i) w[i] = rng.next();
    std::vector<uint16_t> sh((size_t)N * B);
    std::vector<uint32_t> scales((sh.size() + 1) / 2, 0);
    for (size_t i = 0; i < sh.size(); ++i) {
        sh[i] = f32ToF16(0.001f + 0.0001f * float(i % 19));
        scales[i / 2] |= uint32_t(sh[i]) << (16 * (i & 1));
    }
    float rstd = 1.0f;
    if (fusedNorm) {
        double ss = 0.0; for (float v : x) ss += double(v) * v;
        rstd = 1.0f / std::sqrt(float(ss / K) + 1.0e-6f);
    }
    std::vector<float> expected(N);
    for (int n = 0; n < N; ++n) {
        float sum = bias[n];
        for (int k = 0; k < K; ++k) {
            int q = int8_t((w[(size_t)n * K / 4 + k / 4] >> (8 * (k & 3))) & 255);
            float xv = x[k] * (fusedNorm ? rstd * norm[k] : 1.0f);
            sum += xv * q * f16ToF32(sh[(size_t)n * B + k / 32]);
        }
        expected[n] = sum;
    }
    auto bx=makeBuffer(gpu,"X",x.data(),K), bw=makeBufferU32(gpu,"W",w.data(),w.size());
    auto bs=makeBufferU32(gpu,"S",scales.data(),scales.size()), bb=makeBuffer(gpu,"B",bias.data(),N);
    auto by=makeBuffer(gpu,"Y",nullptr,N), bn=makeBuffer(gpu,"Norm",norm.data(),K);
    uint32_t epsBits=f32AsU32(1.0e-6f);
    auto p=makeParams(gpu,"P",{K,N,0u,epsBits});
    std::vector<std::pair<uint32_t,GPUBuffer>> bindings={{0,bx},{1,bw},{2,bs},{3,bb},{4,by},{5,p}};
    if (fusedNorm) bindings.push_back({6,bn});
    auto result=dispatchAndReadback(gpu,wgsl,bindings,1,ceilDiv(N,8),1,by,N*4,bindings.size());
    return assertClose((const float*)result.data(),expected.data(),N,3e-4f,3e-4f);
}

TEST(q8_matmul_decode_gemma_shape) { return testQ8DecodeProjection(gpu, false); }
TEST(q8_matmul_norm_gemma_shape) { return testQ8DecodeProjection(gpu, true); }

TEST(fused_qknorm_rope_gemma_hd256) {
    auto wgsl=loadWgsl("norm","fused_qknorm_rope");
    if(wgsl.empty())return{false,"cannot load kernel"};
    const std::string from="const HD: u32 = 128u;",to="const HD: u32 = 256u;";
    auto at=wgsl.find(from);if(at==std::string::npos)return{false,"HD marker missing"};
    wgsl.replace(at,from.size(),to);
    const int HD=256,NH=2,NKV=1,Q=NH*HD,KV=NKV*HD;
    Rng rng(0x5252u);auto qkv=rng.randnVec(Q+2*KV),qw=rng.uniformVec(HD,.5f,1.5f),kw=rng.uniformVec(HD,.5f,1.5f);
    std::vector<float>cs(HD/2,1.0f),sn(HD/2,0.0f),expectedQ(Q),expectedK(HD),expectedV(HD);
    for(int h=0;h<NH;h++){double ss=0;for(int d=0;d<HD;d++)ss+=double(qkv[h*HD+d])*qkv[h*HD+d];float r=1/std::sqrt(float(ss/HD)+1e-6f);for(int d=0;d<HD;d++)expectedQ[h*HD+d]=qkv[h*HD+d]*r*qw[d];}
    double ks=0,vs=0;for(int d=0;d<HD;d++){ks+=double(qkv[Q+d])*qkv[Q+d];vs+=double(qkv[Q+KV+d])*qkv[Q+KV+d];}
    float kr=1/std::sqrt(float(ks/HD)+1e-6f),vr=1/std::sqrt(float(vs/HD)+1e-6f);
    for(int d=0;d<HD;d++){expectedK[d]=qkv[Q+d]*kr*kw[d];expectedV[d]=qkv[Q+KV+d]*vr;}
    auto bi=makeBuffer(gpu,"qkv",qkv.data(),qkv.size()),bq=makeBuffer(gpu,"q",nullptr,Q);
    auto bk=makeBufferU32(gpu,"k",nullptr,HD/2),bv=makeBufferU32(gpu,"v",nullptr,HD/2);
    auto bc=makeBuffer(gpu,"c",cs.data(),cs.size()),bs=makeBuffer(gpu,"s",sn.data(),sn.size());
    auto bqw=makeBuffer(gpu,"qw",qw.data(),HD),bkw=makeBuffer(gpu,"kw",kw.data(),HD);
    auto p=makeParams(gpu,"p",{NH,Q,KV,0u,HD/2,0u,f32AsU32(1e-6f),1u});
    auto qr=dispatchAndReadback(gpu,wgsl,{{0,bi},{1,bq},{2,bk},{3,bv},{4,bc},{5,bs},{6,bqw},{7,bkw},{8,p}},NH+NKV,1,1,bq,Q*4,9);
    auto qc=assertClose((const float*)qr.data(),expectedQ.data(),Q,2e-5f,2e-5f);if(!qc.ok)return qc;
    auto kh=gpu.readBuffer(bk,HD*2),vh=gpu.readBuffer(bv,HD*2);std::vector<float>ka(HD),va(HD);
    for(int d=0;d<HD;d++){ka[d]=f16ToF32(reinterpret_cast<const uint16_t*>(kh.data())[d]);va[d]=f16ToF32(reinterpret_cast<const uint16_t*>(vh.data())[d]);}
    auto kc=assertClose(ka.data(),expectedK.data(),HD,1e-3f,1e-3f);if(!kc.ok)return kc;
    return assertClose(va.data(),expectedV.data(),HD,1e-3f,1e-3f);
}

TEST(gqa_chunked_gemma_hd256) {
    auto p1=loadWgsl("attention","gqa_chunked_pass1"),p2=loadWgsl("attention","gqa_chunked_pass2");
    if(p1.empty()||p2.empty())return{false,"cannot load kernels"};
    const std::string hd128="const HD: u32 = 128u;",hd256="const HD: u32 = 256u;";
    const std::string ept4="const HD_PER_THREAD: u32 = 4u;",ept8="const HD_PER_THREAD: u32 = 8u;";
    for(auto* s:{&p1,&p2}){auto a=s->find(hd128);if(a==std::string::npos)return{false,"HD marker missing"};s->replace(a,hd128.size(),hd256);a=s->find(ept4);if(a==std::string::npos)return{false,"EPT marker missing"};s->replace(a,ept4.size(),ept8);}
    const int HD=256,NH=2,T=5,MC=4,PS=HD+2;Rng rng(0x6262u);
    auto q=rng.uniformVec(NH*HD,-.2f,.2f),kf=rng.uniformVec(T*HD,-.2f,.2f),vf=rng.uniformVec(T*HD,-.2f,.2f);
    auto kp=packF16(kf),vp=packF16(vf);std::vector<float>expected(NH*HD);
    for(int h=0;h<NH;h++){float mx=-1e30f;std::vector<float>score(T);for(int t=0;t<T;t++){float z=0;for(int d=0;d<HD;d++)z+=q[h*HD+d]*f16ToF32(uint16_t(kp[(t*HD+d)/2]>>(16*((t*HD+d)&1))));score[t]=z;mx=std::max(mx,z);}float den=0;for(float&z:score){z=std::exp(z-mx);den+=z;}for(int d=0;d<HD;d++)for(int t=0;t<T;t++)expected[h*HD+d]+=score[t]/den*f16ToF32(uint16_t(vp[(t*HD+d)/2]>>(16*((t*HD+d)&1))));}
    auto bq=makeBuffer(gpu,"q",q.data(),q.size()),bk=makeBufferU32(gpu,"k",kp.data(),kp.size()),bv=makeBufferU32(gpu,"v",vp.data(),vp.size());
    auto bp=makeBuffer(gpu,"partial",nullptr,NH*MC*PS),bo=makeBuffer(gpu,"out",nullptr,NH*HD);
    auto prm=makeParams(gpu,"p",{HD,2u,T,0u,1u,f32AsU32(1.0f),f32AsU32(-1e9f),MC});
    dispatchAndReadback(gpu,p1,{{0,bq},{1,bk},{2,bv},{3,bp},{4,prm}},NH,MC,1,bp,NH*MC*PS*4,5);
    auto out=dispatchAndReadback(gpu,p2,{{0,bp},{1,bo},{2,prm}},NH,1,1,bo,NH*HD*4,3);
    return assertClose((const float*)out.data(),expected.data(),NH*HD,1e-4f,1e-4f);
}

TEST(q5k_matmul_reference) {
    auto wgsl=loadWgsl("quant_kq","q5k_matmul");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=8,K=256;Rng rng(4242);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*176);
    for(int n=0;n<N;n++){uint8_t*b=raw.data()+n*176;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0005f*n);memcpy(b,&d,2);memcpy(b+2,&dm,2);for(int i=4;i<176;i++)b[i]=uint8_t((n*37+i*13+7)&255);}
    auto packed=pack_q5k(raw.data(),N,K);std::vector<float>deq(N*K),expected(N);dequant_kquant(raw.data(),deq.data(),N,K,GGUF_TYPE_Q5_K);for(int n=0;n<N;n++)for(int k=0;k<K;k++)expected[n]+=x[k]*deq[n*K+k];
    std::vector<float>bias(N,0);auto bx=makeBuffer(gpu,"X",x.data(),K),bw=makeBufferU32(gpu,"W",packed.data.data(),(int)packed.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,packed.nBlocks,packed.rowStrideWords,0});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},1,1,1,by,N*4,5);return assertClose((const float*)r.data(),expected.data(),N,2e-4f,2e-4f);
}

TEST(q5k_matmul_prequant_batched_dp4a_reference) {
    std::string q=WGSL_Q8_QUANTIZE_BATCHED_DP4A,mm=WGSL_Q5K_MATMUL_PREQUANT_BATCHED_DP4A;
    const int M=9,N=17,K=512,NB=K/256;Rng rng(0x55B8);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*NB*176);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*176;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<176;j++)p[j]=uint8_t(n*37+b*11+j*13+7);}
    auto pk=pack_q5k(raw.data(),N,K);std::vector<float>dq(N*K),xq(M*K),bias(N),exp(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q5_K);
    for(int m=0;m<M;m++)for(int b=0;b<K/32;b++){float amax=0;for(int j=0;j<32;j++)amax=std::max(amax,std::abs(x[m*K+b*32+j]));float s=amax/127.0f;for(int j=0;j<32;j++){int v=s==0?0:std::max(-127,std::min(127,(int)std::round(x[m*K+b*32+j]/s)));xq[m*K+b*32+j]=v*s;}}
    for(int n=0;n<N;n++)bias[n]=.01f*n;for(int m=0;m<M;m++)for(int n=0;n<N;n++){exp[m*N+n]=bias[n];for(int k=0;k<K;k++)exp[m*N+n]+=xq[m*K+k]*dq[n*K+k];}
    auto bx=makeBuffer(gpu,"X",x.data(),M*K),bq=makeBufferU32(gpu,"XQ",nullptr,M*K/4),bs=makeBuffer(gpu,"XS",nullptr,M*K/32),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M,pk.nBlocks,pk.rowStrideWords});
    dispatchAndReadback(gpu,q,{{0,bx},{1,bq},{2,bs},{3,p}},K/256,M,1,bq,M*K,4);auto r=dispatchAndReadback(gpu,mm,{{0,bq},{1,bs},{2,bw},{3,bb},{4,by},{5,p}},ceilDiv(M,8),ceilDiv(N,8),1,by,M*N*4,6);return assertClose((const float*)r.data(),exp.data(),M*N,3e-3f,3e-3f);
}

TEST(q4k_matmul_reference) {
    auto wgsl=loadWgsl("quant_kq","q4k_matmul");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=8,K=256;Rng rng(4343);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*144);
    for(int n=0;n<N;n++){uint8_t*b=raw.data()+n*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0005f*n);memcpy(b,&d,2);memcpy(b+2,&dm,2);for(int i=4;i<144;i++)b[i]=uint8_t((n*31+i*17+3)&255);}
    auto packed=pack_q4k(raw.data(),N,K);std::vector<float>deq(N*K),expected(N);dequant_kquant(raw.data(),deq.data(),N,K,GGUF_TYPE_Q4_K);for(int n=0;n<N;n++)for(int k=0;k<K;k++)expected[n]+=x[k]*deq[n*K+k];
    std::vector<float>bias(N,0);auto bx=makeBuffer(gpu,"X",x.data(),K),bw=makeBufferU32(gpu,"W",packed.data.data(),(int)packed.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,packed.nBlocks,packed.rowStrideWords,0});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},1,1,1,by,N*4,5);return assertClose((const float*)r.data(),expected.data(),N,2e-4f,2e-4f);
}

TEST(q4k_ort_dense_layout_reference) {
    // Validate the exact Q4_K -> ORT dense-nibble mapping independently of
    // the large 64x64 matmul tile. This catches byte-order/layout mistakes
    // without risking a long-running end-to-end shader.
    const char* wgsl=R"WGSL(
@group(0) @binding(0)var<storage,read>W:array<u32>;
@group(0) @binding(1)var<storage,read_write>Y:array<f32>;
@group(0) @binding(2)var<storage,read>P:array<u32>;
fn pair(a:u32,b:u32,high:bool)->u32{let mask=0x0F0F0F0Fu;let x=select(a&mask,(a>>4u)&mask,high);let z=select(b&mask,(b>>4u)&mask,high);let px=(x&15u)|((x>>4u)&240u)|((x>>8u)&0xF00u)|((x>>12u)&0xF000u);let pz=(z&15u)|((z>>4u)&240u)|((z>>8u)&0xF00u)|((z>>12u)&0xF000u);return px|(pz<<16u);}
fn nibble(v:u32,i:u32)->u32{return(v>>((i&7u)*4u))&15u;}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let flat=gid.x;let N=P[0];let K=P[1];let rs=P[2];if(flat>=N*K){return;}let n=flat/K;let k=flat%K;let block=k/256u;let sb=(k%256u)/32u;let in32=k&31u;let half=in32/16u;let j=in32&15u;let base=n*rs+block*36u;let qbase=base+4u+(sb/2u)*8u+half*4u;let d0=pair(W[qbase],W[qbase+1u],(sb&1u)!=0u);let d1=pair(W[qbase+2u],W[qbase+3u],(sb&1u)!=0u);let q=select(nibble(d0,j),nibble(d1,j-8u),j>=8u);let dm=unpack2x16float(W[base]);let shift=(sb&3u)*8u;let dv=(W[base+1u]>>shift)&255u;let mv=(W[base+2u]>>shift)&255u;var sc:u32;var mn:u32;if(sb<4u){sc=dv&63u;mn=mv&63u;}else{let md=(W[base+3u]>>shift)&255u;sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}Y[flat]=dm.x*f32(sc)*f32(q)-dm.y*f32(mn);}
)WGSL";
    const int N=67,K=512,NB=K/256;std::vector<uint8_t>raw(N*NB*144);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+b*13+j*17);}
    auto pk=pack_q4k(raw.data(),N,K);std::vector<float>expected(N*K);dequant_kquant(raw.data(),expected.data(),N,K,GGUF_TYPE_Q4_K);
    auto bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),by=makeBuffer(gpu,"Y",nullptr,N*K),p=makeParams(gpu,"P",{N,K,pk.rowStrideWords});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bw},{1,by},{2,p}},ceilDiv(N*K,256),1,1,by,N*K*4,3);
    return assertClose((const float*)r.data(),expected.data(),N*K,1e-6f,1e-6f);
}

TEST(q4k_gpu_repack_ort_dense_reference) {
    auto wgsl=loadWgsl("quant_kq","q4k_repack_ort_dense");if(wgsl.empty())return{false,"cannot load repack kernel"};
    const int N=67,K=512,NB=K/256;std::vector<uint8_t>raw(N*NB*144);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+b*13+j*17);}
    auto rawPacked=pack_q4k(raw.data(),N,K);auto expected=repack_q4k_dense(raw.data(),N,K);
    auto br=makeBufferU32(gpu,"Raw",rawPacked.data.data(),(int)rawPacked.data.size()),bd=makeBufferU32(gpu,"Dense",nullptr,(int)expected.weights.size()),bs=makeBuffer(gpu,"ScaleMin",nullptr,(int)expected.scalesMins.size()),p=makeParams(gpu,"P",{K,N,rawPacked.nBlocks,rawPacked.rowStrideWords});
    auto bytes=dispatchAndReadback(gpu,wgsl,{{0,br},{1,bd},{2,bs},{3,p}},ceilDiv(N*(K/32),256),1,1,bd,expected.weights.size()*4,4);
    if(memcmp(bytes.data(),expected.weights.data(),bytes.size())!=0)return{false,"dense nibble layout mismatch"};
    auto scaleBytes=dispatchAndReadback(gpu,wgsl,{{0,br},{1,bd},{2,bs},{3,p}},ceilDiv(N*(K/32),256),1,1,bs,expected.scalesMins.size()*4,4);
    return assertClose((const float*)scaleBytes.data(),expected.scalesMins.data(),(int)expected.scalesMins.size(),1e-6f,1e-6f);
}

TEST(q4k_ort_tile64_reference) {
    std::string mm=q4kOrtRepackedTileTestSource();if(mm.empty())return{false,"cannot adapt ORT tile"};
    const int M=64,N=64,K=256,NB=1;Rng rng(0x4A64);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*144);
    for(int n=0;n<N;n++){uint8_t*p=raw.data()+n*144;uint16_t d=f32ToF16(.02f+.0002f*n),dm=f32ToF16(.01f+.0001f*n);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+j*17);}
    auto pk=pack_q4k(raw.data(),N,K);auto dense=repack_q4k_dense(raw.data(),N,K);std::vector<float>dq(N*K),xq(M*K),expected(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);
    for(int m=0;m<M;m++)for(int b=0;b<K/32;b++){float amax=0;for(int j=0;j<32;j++)amax=std::max(amax,std::abs(x[m*K+b*32+j]));float sc=amax/127.0f;for(int j=0;j<32;j++){int q=sc==0?0:std::max(-127,std::min(127,(int)std::round(x[m*K+b*32+j]/sc)));xq[m*K+b*32+j]=q*sc;}}
    for(int m=0;m<M;m++)for(int n=0;n<N;n++)for(int k=0;k<K;k++)expected[m*N+n]+=xq[m*K+k]*dq[n*K+k];
    auto bx=makeBuffer(gpu,"X",x.data(),M*K),bq=makeBufferU32(gpu,"XQ",nullptr,M*K/4),bs=makeBuffer(gpu,"XS",nullptr,M*K/32),bw=makeBufferU32(gpu,"W",dense.weights.data(),(int)dense.weights.size()),bsm=makeBuffer(gpu,"SM",dense.scalesMins.data(),(int)dense.scalesMins.size()),by=makeBuffer(gpu,"Y",nullptr,M*N),qp=makeParams(gpu,"QP",{K,N,M,NB,pk.rowStrideWords});
    dispatchAndReadback(gpu,WGSL_Q8_QUANTIZE_BATCHED_DP4A,{{0,bx},{1,bq},{2,bs},{3,qp}},1,M,1,bq,M*K,4);
    auto p=makeParams(gpu,"P",{1,M,N,K,K/8,K/16,1,1,pk.rowStrideWords,0});
    auto r=dispatchAndReadback(gpu,mm,{{0,bq},{1,bs},{2,bw},{3,bsm},{4,by},{5,p}},1,1,1,by,M*N*4,6);
    return assertClose((const float*)r.data(),expected.data(),M*N,5e-3f,5e-3f);
}

TEST(ort_dp4a_tile64_upstream_reference) {
    const int M=64,N=64,K=256;Rng rng(0x0A64);auto x=rng.randnVec(M*K);
    std::vector<uint32_t>aq(M*K/4),wq(N*K/8),as16((M*K/128+1)/2),ws16((N*K/32+1)/2);
    std::vector<float>adeq(M*K),wdeq(N*K),expected(M*N);
    for(int m=0;m<M;m++)for(int g=0;g<K/128;g++){float mx=0;for(int j=0;j<128;j++)mx=std::max(mx,std::abs(x[m*K+g*128+j]));float sc=mx/127.0f;uint16_t h=f32ToF16(sc);int si=m*(K/128)+g;as16[si/2]|=uint32_t(h)<<(16*(si&1));for(int j=0;j<128;j++){int k=g*128+j;int q=sc==0?0:std::max(-127,std::min(127,(int)std::round(x[m*K+k]/sc)));adeq[m*K+k]=q*sc;aq[(m*K+k)/4]|=uint32_t(q&255)<<(8*(k&3));}}
    for(int n=0;n<N;n++)for(int g=0;g<K/32;g++){float sc=.01f+.0001f*(n+g);int si=n*(K/32)+g;ws16[si/2]|=uint32_t(f32ToF16(sc))<<(16*(si&1));for(int j=0;j<32;j++){int k=g*32+j;int q=(n*13+k*7+3)&15;wdeq[n*K+k]=(q-8)*f16ToF32(f32ToF16(sc));int nib=n*K+k;wq[nib/8]|=uint32_t(q)<<(4*(nib&7));}}
    for(int m=0;m<M;m++)for(int n=0;n<N;n++)for(int k=0;k<K;k++)expected[m*N+n]+=adeq[m*K+k]*wdeq[n*K+k];
    auto ba=makeBufferU32(gpu,"A",aq.data(),(int)aq.size()),bas=makeBufferU32(gpu,"AS",as16.data(),(int)as16.size()),bw=makeBufferU32(gpu,"W",wq.data(),(int)wq.size()),bws=makeBufferU32(gpu,"WS",ws16.data(),(int)ws16.size()),by=makeBufferU32(gpu,"Y",nullptr,M*N/2),p=makeParams(gpu,"P",{1,M,N,K,K/8,K/16,1,1,0,0});
    auto r=dispatchAndReadback(gpu,WGSL_ORT_DP4A_MATMUL_EXACT,{{0,ba},{1,bas},{2,bw},{3,bws},{4,by},{5,p}},1,1,1,by,M*N*2,6);
    std::vector<float>actual(M*N);const uint16_t*hp=(const uint16_t*)r.data();for(int i=0;i<M*N;i++)actual[i]=f16ToF32(hp[i]);
    return assertClose(actual.data(),expected.data(),M*N,.08f,.02f);
}

TEST(q4k_matmul_dp4a_reference) {
    auto wgsl=loadWgsl("quant_kq","q4k_matmul_dp4a");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=8,K=512,NB=K/256;Rng rng(0x44D4);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*NB*144);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+b*13+j*17);}
    auto pk=pack_q4k(raw.data(),N,K);std::vector<float>dq(N*K),xq(K),bias(N),exp(N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);
    for(int b=0;b<K/32;b++){float amax=0;for(int j=0;j<32;j++)amax=std::max(amax,std::abs(x[b*32+j]));float s=amax/127.0f;for(int j=0;j<32;j++){int q=s==0?0:std::max(-127,std::min(127,(int)std::round(x[b*32+j]/s)));xq[b*32+j]=q*s;}}
    for(int n=0;n<N;n++){bias[n]=.01f*n;exp[n]=bias[n];for(int k=0;k<K;k++)exp[n]+=xq[k]*dq[n*K+k];}
    auto bx=makeBuffer(gpu,"X",x.data(),K),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,pk.nBlocks,pk.rowStrideWords,0});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},1,1,1,by,N*4,5);return assertClose((const float*)r.data(),exp.data(),N,2e-3f,2e-3f);
}

TEST(q4k_matmul_prequant_dp4a_reference) {
    auto q=loadWgsl("quant_kq","q8_quantize_dp4a"),mm=loadWgsl("quant_kq","q4k_matmul_prequant_dp4a");if(q.empty()||mm.empty())return{false,"cannot load kernels"};
    const int N=35,K=512,NB=K/256;Rng rng(0x44D6);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*NB*144);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+b*13+j*17);}
    auto pk=pack_q4k(raw.data(),N,K);std::vector<float>dq(N*K),xq(K),bias(N),exp(N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);
    for(int b=0;b<K/32;b++){float amax=0;for(int j=0;j<32;j++)amax=std::max(amax,std::abs(x[b*32+j]));float s=amax/127.0f;for(int j=0;j<32;j++){int v=s==0?0:std::max(-127,std::min(127,(int)std::round(x[b*32+j]/s)));xq[b*32+j]=v*s;}}
    for(int n=0;n<N;n++){bias[n]=.01f*n;exp[n]=bias[n];for(int k=0;k<K;k++)exp[n]+=xq[k]*dq[n*K+k];}
    auto bx=makeBuffer(gpu,"X",x.data(),K),bq=makeBufferU32(gpu,"XQ",nullptr,K/4),bs=makeBuffer(gpu,"XS",nullptr,K/32),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,pk.nBlocks,pk.rowStrideWords,0});
    dispatchAndReadback(gpu,q,{{0,bx},{1,bq},{2,bs},{3,p}},K/256,1,1,bq,K,4);
    auto r=dispatchAndReadback(gpu,mm,{{0,bq},{1,bs},{2,bw},{3,bb},{4,by},{5,p}},1,ceilDiv(N,8),1,by,N*4,6);return assertClose((const float*)r.data(),exp.data(),N,2e-3f,2e-3f);
}

TEST(q4k_matmul_prequant_batched_dp4a_reference) {
    std::string q=WGSL_Q8_QUANTIZE_BATCHED_DP4A,mm=WGSL_Q4K_MATMUL_PREQUANT_BATCHED_DP4A;
    const int M=9,N=17,K=512,NB=K/256;Rng rng(0x44B8);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*NB*144);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+b*13+j*17);}
    auto pk=pack_q4k(raw.data(),N,K);std::vector<float>dq(N*K),xq(M*K),bias(N),exp(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);
    for(int m=0;m<M;m++)for(int b=0;b<K/32;b++){float amax=0;for(int j=0;j<32;j++)amax=std::max(amax,std::abs(x[m*K+b*32+j]));float s=amax/127.0f;for(int j=0;j<32;j++){int v=s==0?0:std::max(-127,std::min(127,(int)std::round(x[m*K+b*32+j]/s)));xq[m*K+b*32+j]=v*s;}}
    for(int n=0;n<N;n++)bias[n]=.01f*n;for(int m=0;m<M;m++)for(int n=0;n<N;n++){exp[m*N+n]=bias[n];for(int k=0;k<K;k++)exp[m*N+n]+=xq[m*K+k]*dq[n*K+k];}
    auto bx=makeBuffer(gpu,"X",x.data(),M*K),bq=makeBufferU32(gpu,"XQ",nullptr,M*K/4),bs=makeBuffer(gpu,"XS",nullptr,M*K/32),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M,pk.nBlocks,pk.rowStrideWords});
    dispatchAndReadback(gpu,q,{{0,bx},{1,bq},{2,bs},{3,p}},K/256,M,1,bq,M*K,4);auto r=dispatchAndReadback(gpu,mm,{{0,bq},{1,bs},{2,bw},{3,bb},{4,by},{5,p}},ceilDiv(M,8),ceilDiv(N,8),1,by,M*N*4,6);return assertClose((const float*)r.data(),exp.data(),M*N,2e-3f,2e-3f);
}

TEST(q4k_matmul_128_reference) {
 auto wgsl=loadWgsl("quant_kq","q4k_matmul_128");if(wgsl.empty())return{false,"cannot load kernel"};const int N=7,K=512,NB=2;Rng rng(5555);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*NB*144);for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int j=4;j<144;j++)p[j]=uint8_t(n*31+b*13+j*17);}auto pk=pack_q4k(raw.data(),N,K);std::vector<float>dq(N*K),bias(N),exp(N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);for(int n=0;n<N;n++){bias[n]=.01f*n;exp[n]=bias[n];for(int k=0;k<K;k++)exp[n]+=x[k]*dq[n*K+k];}auto bx=makeBuffer(gpu,"X",x.data(),K),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,pk.nBlocks,pk.rowStrideWords,0});auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},1,ceilDiv(N,4),1,by,N*4,5);return assertClose((const float*)r.data(),exp.data(),N,2e-4f,2e-4f);
}

TEST(q6k_matmul_reference) {
    auto wgsl=loadWgsl("quant_kq","q6k_matmul");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=8,K=768,NB=K/256;Rng rng(4646);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*NB*210);
    for(int n=0;n<N;n++)for(int q=0;q<NB;q++){uint8_t*b=raw.data()+(n*NB+q)*210;for(int i=0;i<208;i++)b[i]=uint8_t((n*29+q*23+i*19+5)&255);uint16_t d=f32ToF16(.02f+.001f*n+.0003f*q);memcpy(b+208,&d,2);}
    auto packed=pack_q6k(raw.data(),N,K);std::vector<float>deq(N*K),expected(N);dequant_kquant(raw.data(),deq.data(),N,K,GGUF_TYPE_Q6_K);for(int n=0;n<N;n++)for(int k=0;k<K;k++)expected[n]+=x[k]*deq[n*K+k];
    std::vector<float>bias(N,0);auto bx=makeBuffer(gpu,"X",x.data(),K),bw=makeBufferU32(gpu,"W",packed.data.data(),(int)packed.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,packed.nBlocks,packed.rowStrideWords,0});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},1,1,1,by,N*4,5);return assertClose((const float*)r.data(),expected.data(),N,2e-4f,2e-4f);
}

TEST(q6k_matmul_prequant_dp4a_reference) {
    auto q=loadWgsl("quant_kq","q8_quantize_dp4a");
    std::string mm=WGSL_Q6K_MATMUL_PREQUANT_DP4A;
    if(q.empty()||mm.empty())return{false,"cannot load kernels"};
    const int N=35,K=768,NB=K/256;Rng rng(0x66D4);auto x=rng.randnVec(K);
    std::vector<uint8_t>raw(N*NB*210);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){
        uint8_t*p=raw.data()+(n*NB+b)*210;
        for(int i=0;i<208;i++)p[i]=uint8_t(n*29+b*23+i*19+5);
        uint16_t d=f32ToF16(.02f+.001f*n+.0003f*b);memcpy(p+208,&d,2);
    }
    auto pk=pack_q6k(raw.data(),N,K);std::vector<float>dq(N*K),xq(K),bias(N),exp(N);
    dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q6_K);
    for(int b=0;b<K/32;b++){
        float amax=0;for(int j=0;j<32;j++)amax=std::max(amax,std::abs(x[b*32+j]));
        float s=amax/127.0f;for(int j=0;j<32;j++){
            int v=s==0?0:std::max(-127,std::min(127,(int)std::round(x[b*32+j]/s)));
            xq[b*32+j]=v*s;
        }
    }
    for(int n=0;n<N;n++){bias[n]=.01f*n;exp[n]=bias[n];
        for(int k=0;k<K;k++)exp[n]+=xq[k]*dq[n*K+k];}
    auto bx=makeBuffer(gpu,"X",x.data(),K),bq=makeBufferU32(gpu,"XQ",nullptr,K/4),
         bs=makeBuffer(gpu,"XS",nullptr,K/32),
         bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),
         bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),
         p=makeParams(gpu,"P",{K,N,pk.nBlocks,pk.rowStrideWords,0});
    dispatchAndReadback(gpu,q,{{0,bx},{1,bq},{2,bs},{3,p}},K/256,1,1,bq,K,4);
    auto r=dispatchAndReadback(gpu,mm,{{0,bq},{1,bs},{2,bw},{3,bb},{4,by},{5,p}},
        1,ceilDiv(N,8),1,by,N*4,6);
    return assertClose((const float*)r.data(),exp.data(),N,3e-3f,3e-3f);
}

TEST(q6k_gather_reference) {
    auto wgsl=loadWgsl("quant_kq","q6k_gather");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=3,K=768,NB=K/256,TOK=2;std::vector<uint8_t>raw(N*NB*210);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*210;for(int i=0;i<208;i++)p[i]=uint8_t((n*41+b*23+i*11+9)&255);uint16_t d=f32ToF16(.015f+.001f*n+.0002f*b);memcpy(p+208,&d,2);}
    auto packed=pack_q6k(raw.data(),N,K);std::vector<float>deq(N*K);dequant_kquant(raw.data(),deq.data(),N,K,GGUF_TYPE_Q6_K);int32_t tok=TOK;
    auto bw=makeBufferU32(gpu,"W",packed.data.data(),(int)packed.data.size()),bt=makeBufferU32(gpu,"T",reinterpret_cast<uint32_t*>(&tok),1),bx=makeBuffer(gpu,"X",nullptr,K),p=makeParams(gpu,"P",{K,packed.rowStrideWords});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bw},{1,bt},{2,bx},{3,p}},ceilDiv(K,256),1,1,bx,K*4,4);return assertClose((const float*)r.data(),deq.data()+TOK*K,K,1e-6f,1e-6f);
}

TEST(q6k_matmul_wide_reference) {
    auto wgsl=loadWgsl("quant_kq","q6k_matmul_wide");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=19,K=768,NB=K/256;Rng rng(4747);auto x=rng.randnVec(K);std::vector<uint8_t>raw(N*NB*210);
    for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*210;for(int i=0;i<208;i++)p[i]=uint8_t((n*37+b*17+i*13+11)&255);uint16_t d=f32ToF16(.018f+.0007f*n+.0002f*b);memcpy(p+208,&d,2);}
    auto packed=pack_q6k(raw.data(),N,K);std::vector<float>deq(N*K),expected(N),bias(N);dequant_kquant(raw.data(),deq.data(),N,K,GGUF_TYPE_Q6_K);for(int n=0;n<N;n++){bias[n]=.01f*n;expected[n]=bias[n];for(int k=0;k<K;k++)expected[n]+=x[k]*deq[n*K+k];}
    auto bx=makeBuffer(gpu,"X",x.data(),K),bw=makeBufferU32(gpu,"W",packed.data.data(),(int)packed.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,N),p=makeParams(gpu,"P",{K,N,packed.nBlocks,packed.rowStrideWords,0});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},1,ceilDiv(N,16),1,by,N*4,5);return assertClose((const float*)r.data(),expected.data(),N,2e-4f,2e-4f);
}

TEST(q6k_gather_batched_reference) {
    auto wgsl=loadWgsl("quant_kq","q6k_gather_batched");if(wgsl.empty())return{false,"cannot load kernel"};
    const int N=4,K=768,NB=3,M=3;std::vector<uint8_t>raw(N*NB*210);for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*210;for(int i=0;i<208;i++)p[i]=uint8_t(n*43+b*19+i*7+3);uint16_t d=f32ToF16(.014f+.001f*n);memcpy(p+208,&d,2);}
    auto pk=pack_q6k(raw.data(),N,K);std::vector<float>deq(N*K),exp(M*K);dequant_kquant(raw.data(),deq.data(),N,K,GGUF_TYPE_Q6_K);uint32_t ts[M]={3,0,2};for(int m=0;m<M;m++)memcpy(exp.data()+m*K,deq.data()+ts[m]*K,K*4);
    auto bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bt=makeBufferU32(gpu,"T",ts,M),bx=makeBuffer(gpu,"X",nullptr,M*K),p=makeParams(gpu,"P",{M,K,pk.rowStrideWords});
    auto r=dispatchAndReadback(gpu,wgsl,{{0,bw},{1,bt},{2,bx},{3,p}},ceilDiv(M*K,256),1,1,bx,M*K*4,4);return assertClose((const float*)r.data(),exp.data(),M*K,1e-6f,1e-6f);
}

TEST(q4k_matmul_batched4_reference) {
 auto wgsl=loadWgsl("quant_kq","q4k_matmul_batched4");if(wgsl.empty())return{false,"cannot load kernel"};const int M=5,N=9,K=512,NB=2;Rng rng(4848);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*NB*144);for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int i=4;i<144;i++)p[i]=uint8_t(n*31+b*13+i*17);}
 auto pk=pack_q4k(raw.data(),N,K);std::vector<float>dq(N*K),bias(N),exp(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);for(int n=0;n<N;n++)bias[n]=.01f*n;for(int m=0;m<M;m++)for(int n=0;n<N;n++){float a=bias[n];for(int k=0;k<K;k++)a+=x[m*K+k]*dq[n*K+k];exp[m*N+n]=a;}
 auto bx=makeBuffer(gpu,"X",x.data(),M*K),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M,pk.nBlocks,pk.rowStrideWords});auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},ceilDiv(M,4),ceilDiv(N,8),1,by,M*N*4,5);return assertClose((const float*)r.data(),exp.data(),M*N,2e-4f,2e-4f);
}

TEST(q5k_matmul_batched4_reference) {
 auto wgsl=loadWgsl("quant_kq","q5k_matmul_batched4");if(wgsl.empty())return{false,"cannot load kernel"};const int M=5,N=9,K=512,NB=2;Rng rng(4949);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*NB*176);for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*176;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int i=4;i<176;i++)p[i]=uint8_t(n*29+b*11+i*19);}
 auto pk=pack_q5k(raw.data(),N,K);std::vector<float>dq(N*K),bias(N),exp(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q5_K);for(int n=0;n<N;n++)bias[n]=.01f*n;for(int m=0;m<M;m++)for(int n=0;n<N;n++){float a=bias[n];for(int k=0;k<K;k++)a+=x[m*K+k]*dq[n*K+k];exp[m*N+n]=a;}
 auto bx=makeBuffer(gpu,"X",x.data(),M*K),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M,pk.nBlocks,pk.rowStrideWords});auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},ceilDiv(M,4),ceilDiv(N,8),1,by,M*N*4,5);return assertClose((const float*)r.data(),exp.data(),M*N,2e-4f,2e-4f);
}

TEST(q6k_matmul_batched4_reference) {
 auto wgsl=loadWgsl("quant_kq","q6k_matmul_batched4");if(wgsl.empty())return{false,"cannot load kernel"};const int M=5,N=9,K=768,NB=3;Rng rng(5050);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*NB*210);for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*210;for(int i=0;i<208;i++)p[i]=uint8_t(n*23+b*17+i*13);uint16_t d=f32ToF16(.018f+.0007f*n+.0002f*b);memcpy(p+208,&d,2);}
 auto pk=pack_q6k(raw.data(),N,K);std::vector<float>dq(N*K),bias(N),exp(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q6_K);for(int n=0;n<N;n++)bias[n]=.01f*n;for(int m=0;m<M;m++)for(int n=0;n<N;n++){float a=bias[n];for(int k=0;k<K;k++)a+=x[m*K+k]*dq[n*K+k];exp[m*N+n]=a;}
 auto bx=makeBuffer(gpu,"X",x.data(),M*K),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M,pk.nBlocks,pk.rowStrideWords});auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},ceilDiv(M,4),ceilDiv(N,8),1,by,M*N*4,5);return assertClose((const float*)r.data(),exp.data(),M*N,2e-4f,2e-4f);
}

TEST(q4k_matmul_batched8_reference) {
 auto wgsl=loadWgsl("quant_kq","q4k_matmul_batched8");if(wgsl.empty())return{false,"cannot load kernel"};const int M=9,N=9,K=512,NB=2;Rng rng(5151);auto x=rng.randnVec(M*K);std::vector<uint8_t>raw(N*NB*144);for(int n=0;n<N;n++)for(int b=0;b<NB;b++){uint8_t*p=raw.data()+(n*NB+b)*144;uint16_t d=f32ToF16(.02f+.001f*n),dm=f32ToF16(.01f+.0002f*b);memcpy(p,&d,2);memcpy(p+2,&dm,2);for(int i=4;i<144;i++)p[i]=uint8_t(n*31+b*13+i*17);}
 auto pk=pack_q4k(raw.data(),N,K);std::vector<float>dq(N*K),bias(N),exp(M*N);dequant_kquant(raw.data(),dq.data(),N,K,GGUF_TYPE_Q4_K);for(int n=0;n<N;n++)bias[n]=.01f*n;for(int m=0;m<M;m++)for(int n=0;n<N;n++){float a=bias[n];for(int k=0;k<K;k++)a+=x[m*K+k]*dq[n*K+k];exp[m*N+n]=a;}
 auto bx=makeBuffer(gpu,"X",x.data(),M*K),bw=makeBufferU32(gpu,"W",pk.data.data(),(int)pk.data.size()),bb=makeBuffer(gpu,"B",bias.data(),N),by=makeBuffer(gpu,"Y",nullptr,M*N),p=makeParams(gpu,"P",{K,N,M,pk.nBlocks,pk.rowStrideWords});auto r=dispatchAndReadback(gpu,wgsl,{{0,bx},{1,bw},{2,bb},{3,by},{4,p}},ceilDiv(M,8),ceilDiv(N,8),1,by,M*N*4,5);return assertClose((const float*)r.data(),exp.data(),M*N,2e-4f,2e-4f);
}


TEST(gemma_rope_batched_qonly) {
    auto wgsl=loadWgsl("attention","gemma_rope_batched");
    if(wgsl.empty()) return {false,"cannot load kernel"};
    const int HD=128;
    Rng rng(789); auto q=rng.randnVec(HD); std::vector<float> ones(HD,1.0f);
    std::vector<float> cs(HD/2,1.0f), sn(HD/2,0.0f);
    auto bq=makeBuffer(gpu,"QKV",q.data(),HD), bo=makeBuffer(gpu,"O",nullptr,HD);
    auto bk=gpu.createBuffer("K",HD*2), bv=gpu.createBuffer("V",HD*2);
    auto bc=makeBuffer(gpu,"C",cs.data(),HD/2), bs=makeBuffer(gpu,"S",sn.data(),HD/2);
    auto bw=makeBuffer(gpu,"W",ones.data(),HD);
    // Exercise a partial rotary prefix: the non-rotary tail must still be
    // normalized and copied to the output.
    auto pp=makeParams(gpu,"P",{1,HD,HD,0,HD/8,0,1,1});
    auto result=dispatchAndReadback(gpu,wgsl,{{0,bq},{1,bo},{2,bk},{3,bv},{4,bc},{5,bs},{6,bw},{7,bw},{8,pp}},
        1,1,1,bo,HD*4,9);
    double ss=0;for(float v:q)ss+=double(v)*v;float r=1.0f/std::sqrt(float(ss/HD)+1e-6f);
    std::vector<float> expected(HD);for(int i=0;i<HD;i++)expected[i]=q[i]*r;
    return assertClose((const float*)result.data(),expected.data(),HD,2e-5f,2e-5f);
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
        else if (std::string(argv[i]) == "--backend" && i + 1 < argc) {
            std::string backend = argv[++i];
            if (backend == "d3d12") g_backend = WGPUBackendType_D3D12;
            else if (backend == "vulkan") g_backend = WGPUBackendType_Vulkan;
            else {
                fprintf(stderr, "Unknown backend: %s (expected d3d12 or vulkan)\n",
                        backend.c_str());
                return 2;
            }
        }
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
        // Out-of-tree builds live under gitignore/runtime/build.
        g_kernelDir = (fs::path(__FILE__).parent_path().parent_path() / "kernels").string();
    }
    if (!fs::exists(g_kernelDir)) {
        fprintf(stderr, "Cannot find kernel directory. Tried:\n  %s\n",
                g_kernelDir.c_str());
        return 1;
    }
    fprintf(stderr, "Kernel dir: %s\n", g_kernelDir.c_str());

    // Init GPU
    GPUContext gpu;
    if (!gpu.init(g_backend)) {
        fprintf(stderr, "Failed to init requested GPU backend\n");
        return 1;
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
