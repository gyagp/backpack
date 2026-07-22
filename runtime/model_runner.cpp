#include "model_runner.h"
#include "onnx_loader.h"
#include "mapped_file.h"
#include "wgsl_shaders.h"
#include "clock_calibration.h"
#include "profile_html.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <execution>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace {

std::string deltaNetValueMajorSource(const char* source) {
    std::string result(source);
    auto replace = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = result.find(from, pos)) != std::string::npos) {
            result.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace("ki * dv + vi", "vi * dk + ki");
    replace("ki*dv+vi", "vi*dk+ki");
    return result;
}

const WGPULimits& effectiveLimits(const GPUContext& gpu) {
    return gpu.deviceLimits.maxComputeInvocationsPerWorkgroup != 0
        ? gpu.deviceLimits
        : gpu.adapterLimits;
}

bool canCompileEmbeddedKernel(GPUContext& gpu, const char* kernelName) {
    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(kernelName);
    if (it == kernels.end()) return false;

    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = {it->second.source, strlen(it->second.source)};
    WGPUShaderModuleDescriptor smD{};
    smD.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&src);
    auto sm = wgpuDeviceCreateShaderModule(gpu.device, &smD);
    if (!sm) return false;
    wgpuShaderModuleRelease(sm);
    return true;
}

void dumpFloatBufferStats(GPUContext& gpu, const char* name, GPUBuffer buf, uint32_t elems) {
    if (!buf.handle || elems == 0) return;
    auto bytes = gpu.readBuffer(buf, (uint64_t)elems * 4ull);
    if (bytes.size() < (size_t)elems * 4ull) return;

    const float* data = reinterpret_cast<const float*>(bytes.data());
    float mn = data[0], mx = data[0];
    double sum = 0.0;
    uint32_t nonzero = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < elems; i++) {
        float v = data[i];
        if (!std::isfinite(v)) continue;
        finite++;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        if (v != 0.0f) nonzero++;
    }
    fprintf(stderr,
            "[debug] %s stats: elems=%u finite=%u nonzero=%u min=% .6e max=% .6e mean=% .6e first4=[% .6e % .6e % .6e % .6e]\n",
            name, elems, finite, nonzero, mn, mx, sum / std::max<uint32_t>(1, finite),
            elems > 0 ? data[0] : 0.0f,
            elems > 1 ? data[1] : 0.0f,
            elems > 2 ? data[2] : 0.0f,
            elems > 3 ? data[3] : 0.0f);
}

void dumpFloatArrayStats(const char* name, const float* data, uint32_t elems) {
    if (!data || elems == 0) return;
    float mn = data[0], mx = data[0];
    double sum = 0.0;
    uint32_t nonzero = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < elems; i++) {
        float v = data[i];
        if (!std::isfinite(v)) continue;
        finite++;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        if (v != 0.0f) nonzero++;
    }
    fprintf(stderr,
            "[debug] %s stats: elems=%u finite=%u nonzero=%u min=% .6e max=% .6e mean=% .6e first4=[% .6e % .6e % .6e % .6e]\n",
            name, elems, finite, nonzero, mn, mx, sum / std::max<uint32_t>(1, finite),
            elems > 0 ? data[0] : 0.0f,
            elems > 1 ? data[1] : 0.0f,
            elems > 2 ? data[2] : 0.0f,
            elems > 3 ? data[3] : 0.0f);
}

int chooseDecodePoolDepth(const GPUContext& gpu) {
    const auto& limits = effectiveLimits(gpu);
    int depth = 2;
    if (limits.maxComputeInvocationsPerWorkgroup >= 256) depth++;
    if (limits.maxComputeWorkgroupStorageSize >= 24u * 1024u) depth++;
    if (gpu.backendType != WGPUBackendType_D3D12 && gpu.supportsSubgroups) depth++;
    return gpu.backendType == WGPUBackendType_D3D12
        ? std::clamp(depth, 3, 8)
        : std::clamp(depth, 2, 8);
}

int chooseDecodeCbPoolBatch(const GPUContext& gpu, const ModelConfig& cfg) {
    const auto& limits = effectiveLimits(gpu);
    int batch = 64;
    if (limits.maxComputeInvocationsPerWorkgroup >= 256) batch += 32;
    if (limits.maxComputeWorkgroupStorageSize >= 24u * 1024u) batch += 32;
    if (gpu.backendType != WGPUBackendType_D3D12 && gpu.supportsSubgroupMatrix) batch += 32;
    if (cfg.nLayer >= 32) batch += 32;
    if (limits.maxStorageBufferBindingSize >= (512ull << 20)) batch += 32;
    return gpu.backendType == WGPUBackendType_D3D12
        ? std::clamp(batch, 96, 160)
        : std::clamp(batch, 64, 256);
}

std::string backendName(WGPUBackendType backendType) {
    switch (backendType) {
        case WGPUBackendType_D3D12: return "d3d12";
        case WGPUBackendType_Vulkan: return "vulkan";
        case WGPUBackendType_Metal: return "metal";
        default: return "other";
    }
}

}  // namespace

// ─── Helpers ─────────────────────────────────────────────────────────────────

WGPUBindGroup ModelRunner::makeBG(
        const CompiledPipeline& pl,
        const std::vector<std::pair<uint32_t, GPUBuffer>>& bindings) {
    WGPUBindGroupEntry entries[16];
    for (size_t i = 0; i < bindings.size() && i < 16; i++) {
        memset(&entries[i], 0, sizeof(WGPUBindGroupEntry));
        entries[i].binding = bindings[i].first;
        entries[i].buffer  = bindings[i].second.handle;
        entries[i].offset  = bindings[i].second.offset;
        entries[i].size    = bindings[i].second.size;
    }
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = pl.bgLayout;
    d.entryCount = (uint32_t)bindings.size();
    d.entries = entries;
    return wgpuDeviceCreateBindGroup(gpu->device, &d);
}

const CompiledPipeline& ModelRunner::getKernel(const std::string& name) {
    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(name);
    if (it == kernels.end()) {
        fprintf(stderr, "Kernel not found: %s\n", name.c_str());
        exit(1);
    }
    // The K-quant kernels partition a workgroup into logical 32-lane columns.
    // Subgroup width is adapter/driver dependent. The optimized replacement
    // below hard-codes 32 logical lanes, which is validated on NVIDIA but not
    // on Intel Arc or AMD. Keep the portable workgroup-memory reduction there.
    std::string adapter = gpu->adapterName;
    std::transform(adapter.begin(), adapter.end(), adapter.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (adapter.find("nvidia") != std::string::npos) {
        std::string source = it->second.source;
        bool patched = false;
        auto replaceRange = [&](const std::string& begin, const std::string& end,
                                const std::string& replacement) {
            size_t first = source.find(begin);
            if (first == std::string::npos) return;
            size_t last = source.find(end, first);
            if (last == std::string::npos) return;
            last += end.size();
            source.replace(first, last - first, replacement);
            patched = true;
        };
        if (name == "q4k_matmul" || name == "q5k_matmul" || name == "q6k_matmul") {
            replaceRange("smem_x[tid] = acc;", "let sum = smem_x[warp_id * 32u];",
                         "let sum = subgroupAdd(acc);");
        } else if (name == "q4k_matmul_128") {
            replaceRange("sx[tid]=acc;", "let s=sx[warp*32u];",
                         "let s=subgroupAdd(acc);");
        } else if (name == "q8_matmul" || name == "qwen35_beta_alpha_gate_q8") {
            const std::string result = name == "q8_matmul" ? "warp_sum" : "sum";
            replaceRange("reduce_scratch[tid] = acc;",
                         "let " + result + " = reduce_scratch[warp_id * 32u];",
                         "let " + result + " = subgroupAdd(acc);");
        }
        if (patched) {
            source.insert(0, "enable subgroups;\n");
            return gpu->getOrCreatePipeline(name + "_subgroup32", source,
                                             it->second.numBindings);
        }
    }
    // AMD exposes wave64 here, while these kernels map one output column to
    // each logical 32-lane half. subgroupAdd therefore mixes adjacent logits.
    // XOR masks 16..1 reduce each half independently and retain the packed-dot
    // arithmetic used by the NVIDIA/Intel path.
    if (gpu->supportsSubgroups &&
        (adapter.find("amd") != std::string::npos ||
         adapter.find("radeon") != std::string::npos) &&
        (name == "q4k_matmul_dp4a" ||
         name == "q4k_matmul_prequant_dp4a")) {
        std::string source = it->second.source;
        if (name == "q4k_matmul_prequant_dp4a") {
            // llama.cpp's Vulkan quantized matvec keeps the prequantized
            // activation fragment in subgroup-local registers.  On AMD this
            // trades a few cache-friendly duplicate reads for eliminating
            // two workgroup-wide barriers per Q4_K block.
            auto replaceOnce = [&](const std::string& from,
                                   const std::string& to) {
                const size_t at = source.find(from);
                if (at != std::string::npos) source.replace(at, from.size(), to);
            };
            replaceOnce(
                "        if (tid < 64u) { xq[tid] = XQ[b * 64u + tid]; }\n"
                "        if (tid < 8u) { xs[tid] = XS[b * 8u + tid]; }\n"
                "        workgroupBarrier();\n", "");
            replaceOnce("        let aq0 = xq[lane * 2u];\n"
                        "        let aq1 = xq[lane * 2u + 1u];",
                        "        let aq0 = XQ[b * 64u + lane * 2u];\n"
                        "        let aq1 = XQ[b * 64u + lane * 2u + 1u];\n"
                        "        let xscale = XS[b * 8u + sb];");
            replaceOnce("acc[c] += xs[sb] *", "acc[c] += xscale *");
            replaceOnce("        workgroupBarrier();\n    }", "    }");
        }
        const std::string from = name == "q4k_matmul_dp4a"
            ? "let total = subgroupAdd(acc);"
            : "let total = subgroupAdd(acc[c]);";
        const std::string initial = name == "q4k_matmul_dp4a"
            ? "var total = acc;" : "var total = acc[c];";
        const std::string to = initial +
            "\n        total += subgroupShuffleXor(total, 16u);"
            "\n        total += subgroupShuffleXor(total, 8u);"
            "\n        total += subgroupShuffleXor(total, 4u);"
            "\n        total += subgroupShuffleXor(total, 2u);"
            "\n        total += subgroupShuffleXor(total, 1u);";
        const size_t pos = source.find(from);
        if (pos != std::string::npos) {
            source.replace(pos, from.size(), to);
            return gpu->getOrCreatePipeline(name + "_amd_shuffle32", source,
                                             it->second.numBindings);
        }
    }
    return gpu->getOrCreatePipeline(name, it->second.source, it->second.numBindings);
}

// ─── HD-patched kernel loading ───────────────────────────────────────────────

std::string ModelRunner::patchShaderHD(const char* source) const {
    return patchShaderHD(source, cfg.headDim);
}

std::string ModelRunner::patchShaderHD(const char* source, uint32_t hd) const {
    std::string s(source);
    if (hd == 128) return s;  // no patching needed

    // Replace "const HD: u32 = 128u;" with actual value
    {
        const char* pat = "const HD: u32 = 128u;";
        std::string rep = "const HD: u32 = " + std::to_string(hd) + "u;";
        auto pos = s.find(pat);
        if (pos != std::string::npos)
            s.replace(pos, strlen(pat), rep);
    }

    // HD_PER_THREAD: for 32-thread kernels, ceil(HD/32). Kernels that use this
    // constant must bounds-check each element so head_dim=80 covers all 80
    // elements without reading/writing the padded 16 lanes.
    {
        uint32_t hpt = (hd + 31u) / 32u;
        const char* pat = "const HD_PER_THREAD: u32 = 4u;";
        std::string rep = "const HD_PER_THREAD: u32 = " + std::to_string(hpt) + "u;";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, strlen(pat), rep);
            pos += rep.size();
        }
    }

    // HD_TILES = HD / 16 (flash_attn_vulkan)
    {
        const char* pat = "const HD_TILES: u32 = 8u;";
        if (hd % 16 == 0) {
            uint32_t hdt = hd / 16;
            std::string rep = "const HD_TILES: u32 = " + std::to_string(hdt) + "u;";
            auto pos = s.find(pat);
            if (pos != std::string::npos)
                s.replace(pos, strlen(pat), rep);
        }
    }

    // HALF = HD / 2 (rope_batched_simple)
    {
        const char* pat = "const HALF: u32 = 64u;";
        uint32_t half = hd / 2;
        std::string rep = "const HALF: u32 = " + std::to_string(half) + "u;";
        auto pos = s.find(pat);
        if (pos != std::string::npos)
            s.replace(pos, strlen(pat), rep);
    }

    // Local variable arrays sized to HD
    {
        std::string pat = "array<f32, 128>";
        std::string rep = "array<f32, " + std::to_string(hd) + ">";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }

    // Shared memory: out_acc for flash_attn (BQ=16 × HD)
    {
        const char* pat = "array<f32, 2048>";
        std::string rep = "array<f32, " + std::to_string(16 * hd) + ">";
        auto pos = s.find(pat);
        if (pos != std::string::npos)
            s.replace(pos, strlen(pat), rep);
    }

    // Generated WGSL kernel: literal "* 128;" and "f32(128.0)"
    {
        std::string pat1 = "* 128;";
        std::string rep1 = "* " + std::to_string(hd) + ";";
        size_t pos = 0;
        while ((pos = s.find(pat1, pos)) != std::string::npos) {
            s.replace(pos, pat1.size(), rep1);
            pos += rep1.size();
        }
    }
    {
        std::string pat2 = "f32(128.0)";
        std::string rep2 = "f32(" + std::to_string(hd) + ".0)";
        size_t pos = 0;
        while ((pos = s.find(pat2, pos)) != std::string::npos) {
            s.replace(pos, pat2.size(), rep2);
            pos += rep2.size();
        }
    }

    return s;
}

const CompiledPipeline& ModelRunner::getKernelHD(const std::string& name) {
    return getKernelHD(name, cfg.headDim);
}

const CompiledPipeline& ModelRunner::getKernelHD(const std::string& name, uint32_t headDim) {
    const bool subgroupChunkedP1 = name == "gqa_chunked_pass1_subgroup";
    const std::string sourceName = subgroupChunkedP1 ? "gqa_chunked_pass1" : name;
    if (!subgroupChunkedP1 && headDim == 128 &&
        (name != "gqa_chunked_pass1" || gqaChunkSize == 64))
        return getKernel(name);

    // Patched pipeline: keyed by name + "_HD" + headDim
    std::string patchedName = name + "_HD" + std::to_string(headDim);
    if (sourceName == "gqa_chunked_pass1")
        patchedName += "_C" + std::to_string(gqaChunkSize);

    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(sourceName);
    if (it == kernels.end()) {
        fprintf(stderr, "Kernel not found: %s\n", name.c_str());
        exit(1);
    }

    std::string patchedSource = patchShaderHD(it->second.source, headDim);
    if (sourceName == "gqa_chunked_pass1" && gqaChunkSize != 64) {
        const std::string from = "const CHUNK: u32 = 64u;";
        const std::string to = "const CHUNK: u32 = " + std::to_string(gqaChunkSize) + "u;";
        size_t pos = patchedSource.find(from);
        if (pos != std::string::npos)
            patchedSource.replace(pos, from.size(), to);
    }
    if (subgroupChunkedP1) {
        const std::string enable = "enable f16;";
        auto enablePos = patchedSource.find(enable);
        if (enablePos != std::string::npos)
            patchedSource.insert(enablePos + enable.size(), "\nenable subgroups;");
        const std::string signature =
            "fn main(@builtin(local_invocation_id) lid: vec3<u32>,\n"
            "        @builtin(workgroup_id) wid: vec3<u32>) {";
        const std::string subgroupSignature =
            "fn main(@builtin(local_invocation_id) lid: vec3<u32>,\n"
            "        @builtin(workgroup_id) wid: vec3<u32>,\n"
            "        @builtin(subgroup_invocation_id) sg_lane: u32,\n"
            "        @builtin(subgroup_size) sg_size: u32) {";
        auto signaturePos = patchedSource.find(signature);
        if (signaturePos != std::string::npos)
            patchedSource.replace(signaturePos, signature.size(), subgroupSignature);
        const std::string reduction =
            "        dot_scratch[lane] = dot_partial;\n"
            "        workgroupBarrier();\n"
            "        for (var stride = 16u; stride > 0u; stride >>= 1u) {\n"
            "            if (lane < stride) {\n"
            "                dot_scratch[lane] += dot_scratch[lane + stride];\n"
            "            }\n"
            "            workgroupBarrier();\n"
            "        }\n"
            "        let dot = dot_scratch[0];";
        const std::string subgroupReduction =
            "        let subgroup_sum = subgroupAdd(dot_partial);\n"
            "        let subgroup_id = lane / sg_size;\n"
            "        if (sg_lane == 0u) { dot_scratch[subgroup_id] = subgroup_sum; }\n"
            "        workgroupBarrier();\n"
            "        if (lane == 0u) {\n"
            "            var total = 0.0;\n"
            "            let subgroup_count = (32u + sg_size - 1u) / sg_size;\n"
            "            for (var s = 0u; s < subgroup_count; s++) { total += dot_scratch[s]; }\n"
            "            dot_scratch[0] = total;\n"
            "        }\n"
            "        workgroupBarrier();\n"
            "        let dot = dot_scratch[0];";
        auto reductionPos = patchedSource.find(reduction);
        if (reductionPos != std::string::npos)
            patchedSource.replace(reductionPos, reduction.size(), subgroupReduction);
    }
    return gpu->getOrCreatePipeline(patchedName, patchedSource,
                                     it->second.numBindings);
}

// ─── Patch SiLU→GELU in kernel source ───────────────────────────────────────

static std::string patchSiluToGelu(const std::string& source) {
    std::string s = source;

    // Replace inline SiLU: gate / (1.0 + exp(-gate)) → GELU approximation
    // The GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // For fused down+act+add kernels, the pattern is:
    //   let silu_gate = gate / (1.0 + exp(-gate));
    // or inline:
    //   gate / (1.0 + exp(-gate)) * up

    // Helper function approach: inject a GELU function and replace SiLU calls
    // First, replace the "silu_gate" named pattern
    {
        std::string pat = "gate / (1.0 + exp(-gate))";
        std::string rep = "(gate * 0.5 * (1.0 + tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate))))";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }
    // Also handle the f16 variant
    {
        std::string pat = "f16(gate / (1.0 + exp(-gate)) * up)";
        std::string rep = "f16(gate * 0.5 * (1.0 + tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate))) * up)";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }

    // Replace kernel description comments
    {
        std::string pat = "SiLU";
        std::string rep = "GELU";
        size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), rep);
            pos += rep.size();
        }
    }

    return s;
}

const CompiledPipeline& ModelRunner::getKernelGelu(const std::string& siluName) {
    std::string geluName = siluName;
    // Replace "silu" with "gelu" in the kernel name
    size_t pos = geluName.find("silu");
    if (pos != std::string::npos)
        geluName.replace(pos, 4, "gelu");
    else
        geluName += "_gelu";

    auto& kernels = getEmbeddedKernels();
    auto it = kernels.find(siluName);
    if (it == kernels.end()) {
        fprintf(stderr, "Kernel not found for GELU patching: %s\n", siluName.c_str());
        exit(1);
    }

    std::string patchedSource = patchSiluToGelu(it->second.source);
    return gpu->getOrCreatePipeline(geluName, patchedSource,
                                     it->second.numBindings);
}

// ─── fp16 conversion helpers ─────────────────────────────────────────────────

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

static uint16_t f32_to_fp16(float v) {
    uint32_t fb;
    memcpy(&fb, &v, 4);
    uint32_t s = (fb >> 16) & 0x8000;
    int32_t  e = ((fb >> 23) & 0xFF) - 112;
    uint32_t m = (fb >> 13) & 0x3FF;
    if (e <= 0)  return (uint16_t)s;
    if (e > 30)  return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | m);
}

// ─── Upload Q8 weight pair ──────────────────────────────────────────────────

static void uploadQ8Weight(GPUContext& gpu, const std::string& name,
                           const Q8Repacked& rep,
                           GPUBuffer& wBuf, GPUBuffer& sBuf) {
    uint64_t wSize = rep.weights.size() * 4;
    uint64_t sSize = rep.scales.size() * 4;
    wBuf = gpu.createBuffer(name + ".w", wSize);
    sBuf = gpu.createBuffer(name + ".s", sSize);
    constexpr uint64_t CHUNK = 64ull * 1024 * 1024;
    for (uint64_t off = 0; off < wSize; off += CHUNK)
        gpu.writeBuffer(wBuf, reinterpret_cast<const uint8_t*>(rep.weights.data()) + off,
                        std::min(CHUNK, wSize - off), off);
    for (uint64_t off = 0; off < sSize; off += CHUNK)
        gpu.writeBuffer(sBuf, reinterpret_cast<const uint8_t*>(rep.scales.data()) + off,
                        std::min(CHUNK, sSize - off), off);
}

static Q8Repacked repackQ4_0Native(const void* rawData, uint32_t N, uint32_t K) {
    Q8Repacked out;
    out.N = N; out.K = K;
    const uint8_t* src = static_cast<const uint8_t*>(rawData);
    const uint32_t blocksPerRow = K / 32;
    const size_t totalBlocks = (size_t)N * blocksPerRow;
    out.weights.resize((size_t)N * (K / 8));
    out.scales.assign((totalBlocks + 1) / 2, 0);
    for (size_t b = 0; b < totalBlocks; b++) {
        const uint8_t* block = src + b * 18;
        uint16_t scale; memcpy(&scale, block, 2);
        out.scales[b / 2] |= (uint32_t)scale << ((b & 1) * 16);
        const uint8_t* qs = block + 2;
        for (uint32_t group = 0; group < 4; group++) {
            uint32_t packed = 0;
            for (uint32_t j = 0; j < 8; j++) {
                uint32_t e = group * 8 + j;
                uint32_t q = e < 16 ? (qs[e] & 0xFu) : (qs[e - 16] >> 4);
                packed |= q << (j * 4);
            }
            out.weights[b * 4 + group] = packed;
        }
    }
    return out;
}

static Q8Repacked concatRepacked(const Q8Repacked& a, const Q8Repacked& b) {
    Q8Repacked out;
    out.N = a.N + b.N; out.K = a.K;
    out.weights.reserve(a.weights.size() + b.weights.size());
    out.weights.insert(out.weights.end(), a.weights.begin(), a.weights.end());
    out.weights.insert(out.weights.end(), b.weights.begin(), b.weights.end());
    // All supported decoder dimensions contain an even number of Q4 blocks
    // per tensor, so packed fp16 scale pairs concatenate without re-alignment.
    out.scales.reserve(a.scales.size() + b.scales.size());
    out.scales.insert(out.scales.end(), a.scales.begin(), a.scales.end());
    out.scales.insert(out.scales.end(), b.scales.begin(), b.scales.end());
    return out;
}

// ─── Load model ──────────────────────────────────────────────────────────────

bool ModelRunner::load(GPUContext& ctx, const std::string& path) {
    gpu = &ctx;
    ggufPath = path;
    modelFormat = "gguf";
    fprintf(stderr, "  [runner.load] opening GGUF: %s\n", path.c_str()); fflush(stderr);

    // Parse GGUF (metadata + tensor index)
    if (!gguf.open(ggufPath)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", ggufPath.c_str());
        return false;
    }
    fprintf(stderr, "  [runner.load] GGUF opened, extracting config...\n"); fflush(stderr);

    // Extract model config from GGUF metadata
    cfg = extractModelConfig(gguf);
    rotaryDim = gguf.getU32(cfg.arch + ".rope.dimension_count", 0);
    if (rotaryDim == cfg.headDim) rotaryDim = 0;
    swaRotaryDim = gguf.getU32(cfg.arch + ".rope.dimension_count_swa", 0);
    fprintf(stderr, "  [runner.load] config extracted: arch=%s nLayer=%u\n", cfg.arch.c_str(), cfg.nLayer); fflush(stderr);
    if (cfg.numExperts > 0) {
        fprintf(stderr, "  [runner.load] MoE detected: %u experts, top-%u active, expert FFN dim=%u\n",
                cfg.numExperts, cfg.numExpertsPerTok, cfg.moeIntermediateSize);
    }

    // ─── Early support gate ────────────────────────────────────────────────
    // Reject combinations that backpack cannot decode at all. MoE archs
    // proceed through loading (small weights are uploaded); the forward path
    // will refuse cleanly if expert dispatch isn't wired up yet.
    {
        // Unsupported GGUF quant formats (have no dequantizer yet).
        auto quantName = [](uint32_t t) -> const char* {
            switch (t) {
                case 16: return "IQ2_XXS";  // not yet ported
                case 19: return "IQ1_S";    // not yet ported
                case 29: return "IQ1_M";    // not yet ported
                case 34: return "TQ1_0";    // not yet ported
                case 35: return "TQ2_0";    // not yet ported
                default: return nullptr;
            }
        };
        const char* probeNames[] = {
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_qkv.weight",
        };
        for (auto* n : probeNames) {
            auto it = gguf.tensor_index.find(n);
            if (it == gguf.tensor_index.end()) continue;
            uint32_t t = (uint32_t)gguf.tensors[it->second].type;
            if (auto* qn = quantName(t)) {
                fprintf(stderr,
                    "\nERROR: GGUF quant format '%s' (type=%u) is not yet supported by backpack.\n"
                    "       Probed tensor: %s\n",
                    qn, t, n);
                return false;
            }
        }
    }

    fprintf(stderr, "  [runner.load] printing model info...\n"); fflush(stderr);

    fprintf(stderr, "Model: %s (%u layers, E=%u, HD=%u, V=%u, KV=%u)\n",
           cfg.arch.c_str(), cfg.nLayer, cfg.nEmbd, cfg.headDim,
           cfg.nVocab, cfg.nKvHeads);
    fprintf(stderr, "  RoPE theta=%.0f, RMSNorm eps=%.1e, QK-norm=%s\n",
           cfg.ropeTheta, cfg.rmsNormEps,
           cfg.hasQkNorm ? "yes" : "no");
    if (rotaryDim > 0)
        fprintf(stderr, "  Partial RoPE: rotary_dim=%u (head_dim=%u)\n", rotaryDim, cfg.headDim);
    if (cfg.numExperts > 0) {
        fprintf(stderr, "  MoE: %u experts, top-%u active per token, expert FFN dim=%u, shared FFN dim=%u\n",
                cfg.numExperts, cfg.numExpertsPerTok, cfg.moeIntermediateSize, cfg.moeSharedIntermediateSize);
    }
    if (cfg.ssmInnerSize > 0) {
        fprintf(stderr, "  SSM/linear attention: d_inner=%u, d_state=%u, conv_k=%u, groups=%u, dt_rank=%u%s\n",
                cfg.ssmInnerSize, cfg.ssmStateSize, cfg.ssmConvKernel,
                cfg.ssmGroupCount, cfg.ssmTimeStepRank,
                cfg.fullAttentionInterval > 0
                    ? (std::string(" (full-attn every ") + std::to_string(cfg.fullAttentionInterval) + " layers)").c_str()
                    : "");
    }
    // Single compute pass for all backends by default; BP_PASS_PER_DISPATCH=1
    // is a debug/validation override for command-buffer hazard isolation.
    passPerDispatch = std::getenv("BP_PASS_PER_DISPATCH") != nullptr;
    fprintf(stderr, "  Backend: %s, %s dispatch\n",
           gpu->backendType == WGPUBackendType_D3D12 ? "D3D12" : "Vulkan",
           passPerDispatch ? "pass-per-dispatch" : "single-pass");

    // Memory-map GGUF file for tensor data
    MappedFile ggufMap;
    if (!ggufMap.open(ggufPath)) {
        fprintf(stderr, "Failed to mmap GGUF file: %s\n", ggufPath.c_str());
        return false;
    }

    // Some GGUFs encode proportional/partial RoPE with a frequency-factor
    // tensor rather than a smaller rope.dimension_count. Gemma 4 global
    // attention, for example, declares a 512-wide rotary domain but stores a
    // contiguous active prefix followed by 1e30 sentinels for unrotated pairs.
    // Our RoPE kernels represent that layout directly as a smaller rotary
    // prefix, so derive its width from the tensor instead of rotating the
    // sentinel dimensions with ordinary frequencies.
    auto ropeFreqIt = gguf.tensor_index.find("rope_freqs.weight");
    if (ropeFreqIt != gguf.tensor_index.end()) {
        const auto& ti = gguf.tensors[ropeFreqIt->second];
        uint64_t count = 1;
        for (uint64_t d : ti.shape) count *= d;
        if (count > 0 && count <= cfg.headDim / 2 &&
            (ti.type == GGUF_TYPE_F32 || ti.type == GGUF_TYPE_F16)) {
            const uint8_t* src = ggufMap.data + gguf.data_offset + ti.offset;
            uint64_t active = 0;
            for (; active < count; active++) {
                float factor = ti.type == GGUF_TYPE_F32
                    ? reinterpret_cast<const float*>(src)[active]
                    : fp16_to_f32(reinterpret_cast<const uint16_t*>(src)[active]);
                if (!std::isfinite(factor) || std::fabs(factor) >= 1.0e20f)
                    break;
            }
            if (active > 0 && active < count) {
                ropeFreqRotaryDim = static_cast<uint32_t>(active * 2);
                fprintf(stderr,
                        "  RoPE frequency factors: %llu/%llu active pairs "
                        "(global rotary_dim=%u)\n",
                        (unsigned long long)active, (unsigned long long)count,
                        ropeFreqRotaryDim);
            }
        }
    }

    // Load weights
    loadWeights(gguf, ggufMap.data);
    fprintf(stderr, "  [runner.load] weights loaded\n"); fflush(stderr);

    // Upload IQ codebook buffers (small — 2 KB + 8 KB — once per session).
    // Only needed when iq3s_matmul / iq2s_matmul kernels will be dispatched,
    // i.e. MoE archs whose routed-expert weights are IQ-quantized. Skipping
    // for non-MoE / dense models saves ~10 KB GPU memory.
    if (cfg.numExperts > 0) {
        uint32_t cb3_n = 0; const uint32_t* cb3 = getIq3sGrid(&cb3_n);
        uint32_t cb2_n = 0; const uint32_t* cb2 = getIq2sGridU32(&cb2_n);
        iq3sCodebookBuf = gpu->createBuffer("iq3s_codebook", cb3_n * 4);
        iq2sCodebookBuf = gpu->createBuffer("iq2s_codebook", cb2_n * 4);
        gpu->writeBuffer(iq3sCodebookBuf, cb3, cb3_n * 4);
        gpu->writeBuffer(iq2sCodebookBuf, cb2, cb2_n * 4);
        fprintf(stderr, "  IQ codebooks uploaded: iq3s=%u u32, iq2s=%u u32\n", cb3_n, cb2_n);
    }

    // For hybrid MoE + SSM archs: MoE FFN dispatch IS wired (Phase 3d).
    // SSM layers emit NO dispatches (passthrough — output incorrect).
    // Allow run so a first tg/s number is measurable even though output is wrong.
    if (cfg.numExperts > 0) {
        fprintf(stderr,
            "  NOTE: hybrid arch — MoE FFN wired; SSM layers passthrough.\n"
            "        Output will be INCORRECT but forward pass completes.\n"
            "        Reported tg/s is a first-attempt timing — do NOT compare to\n"
            "        llama.cpp until SSM forward + attn_gate are also wired.\n");
    }

    // RoPE tables
    computeRopeTables();
    fprintf(stderr, "  [runner.load] RoPE tables computed\n"); fflush(stderr);

    // Build decode pipeline
    buildDecodePipeline();
    fprintf(stderr, "  [runner.load] decode pipeline built\n"); fflush(stderr);

    return true;
}

// ─── Load ONNX model ─────────────────────────────────────────────────────────

bool ModelRunner::loadOnnx(GPUContext& ctx, const std::string& onnxDir) {
    gpu = &ctx;
    ggufPath = onnxDir;
    modelFormat = "onnx";

    // 1. Load ONNX model (parse protobuf, extract & repack weights)
    OnnxLoadResult onnx;
    if (!loadOnnxModel(onnxDir, onnx)) {
        fprintf(stderr, "Failed to load ONNX model from: %s\n", onnxDir.c_str());
        return false;
    }

    cfg = onnx.cfg;
    rotaryDim = onnx.rotaryDim;
    hasPrecomputedRope = onnx.hasPrecomputedRope;

    // The GGUF path infers this table while loading tensor metadata.  ONNX
    // previously left it empty and loadOnnx indexed it unconditionally.
    cfg.perLayer.resize(cfg.nLayer);
    cfg.layerAttnTypes.resize(cfg.nLayer, AttnLayerType::Global);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& pl = cfg.perLayer[i];
        pl.headDim = cfg.headDim;
        pl.qDim = cfg.nHead * cfg.headDim;
        pl.kvDim = cfg.nKvHeads * cfg.headDim;
        pl.intermediateSize = cfg.intermediateSize;
        pl.kvSourceLayer = -1;

        // Fused QKV has N = heads*HD + 2*kv_heads*HD.  This also recovers
        // Gemma 4's alternating 256/512 head dimensions from the exported
        // projection shapes instead of assuming the decoder-level head_size.
        const uint32_t attentionN = i < onnx.layers.size()
            ? (onnx.layers[i].qkv.N > 0
                ? onnx.layers[i].qkv.N : onnx.layers[i].qOnly.N)
            : 0;
        if (i < onnx.layers.size() && onnx.layers[i].qkv.N > 0) {
            const uint32_t denom = cfg.nHead + 2 * cfg.nKvHeads;
            if (denom && onnx.layers[i].qkv.N % denom == 0) {
                pl.headDim = onnx.layers[i].qkv.N / denom;
                pl.qDim = cfg.nHead * pl.headDim;
                pl.kvDim = cfg.nKvHeads * pl.headDim;
            }
        } else if (attentionN > 0 && attentionN % cfg.nHead == 0) {
            pl.headDim = attentionN / cfg.nHead;
            pl.qDim = attentionN;
            pl.kvDim = cfg.nKvHeads * pl.headDim;
        }
        if (i < onnx.layers.size() && onnx.layers[i].gateup.N > 0)
            pl.intermediateSize = onnx.layers[i].gateup.N / 2;
        if (i > 0 && (pl.headDim != cfg.perLayer[0].headDim ||
                      pl.intermediateSize != cfg.perLayer[0].intermediateSize))
            cfg.hasPerLayerDims = true;
    }
    if (cfg.arch == "gemma4") {
        uint32_t maxHd = 0;
        for (const auto& pl : cfg.perLayer) maxHd = std::max(maxHd, pl.headDim);
        for (uint32_t i = 0; i < cfg.nLayer; i++)
            cfg.layerAttnTypes[i] = cfg.perLayer[i].headDim == maxHd
                ? AttnLayerType::Global : AttnLayerType::SlidingWindow;

        // Gemma 4 exports K/V projections only for the first cache-owning
        // layers.  Later Q-only layers reuse the final local/global cache.
        int lastSliding = -1, lastGlobal = -1;
        cfg.sharedKvLayers = 0;
        for (uint32_t i = 0; i < cfg.nLayer; i++) {
            const bool qOnly = i < onnx.layers.size() &&
                onnx.layers[i].qkv.N == 0 && onnx.layers[i].qOnly.N > 0;
            const bool sliding = cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            if (!qOnly) {
                if (sliding) lastSliding = (int)i; else lastGlobal = (int)i;
            } else {
                cfg.perLayer[i].kvSourceLayer = sliding ? lastSliding : lastGlobal;
                cfg.sharedKvLayers++;
            }
        }
    }

    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        if (i >= onnx.layers.size() ||
            (onnx.layers[i].qkv.N == 0 && onnx.layers[i].qOnly.N == 0) ||
            onnx.layers[i].o.N == 0 || onnx.layers[i].gateup.N == 0 ||
            onnx.layers[i].down.N == 0) {
            fprintf(stderr,
                "Unsupported/incomplete ONNX transformer layer %u: "
                "qkv=%u qOnly=%u o=%u gateup=%u down=%u\n", i,
                i < onnx.layers.size() ? onnx.layers[i].qkv.N : 0,
                i < onnx.layers.size() ? onnx.layers[i].qOnly.N : 0,
                i < onnx.layers.size() ? onnx.layers[i].o.N : 0,
                i < onnx.layers.size() ? onnx.layers[i].gateup.N : 0,
                i < onnx.layers.size() ? onnx.layers[i].down.N : 0);
            return false;
        }
    }
    if (onnx.embeddingCPU.size() < (size_t)cfg.nVocab * cfg.nEmbd) {
        fprintf(stderr,
            "Unsupported/incomplete ONNX embedding: got %zu values, need %zu\n",
            onnx.embeddingCPU.size(), (size_t)cfg.nVocab * cfg.nEmbd);
        return false;
    }

    fprintf(stderr, "Model: %s (%u layers, E=%u, HD=%u, V=%u, KV=%u) [ONNX]\n",
           cfg.arch.c_str(), cfg.nLayer, cfg.nEmbd, cfg.headDim,
           cfg.nVocab, cfg.nKvHeads);
    fprintf(stderr, "  RoPE theta=%.0f, RMSNorm eps=%.1e, QK-norm=%s\n",
           cfg.ropeTheta, cfg.rmsNormEps,
           cfg.hasQkNorm ? "yes" : "no");
    if (rotaryDim > 0 && rotaryDim != cfg.headDim)
        fprintf(stderr, "  Partial RoPE: rotary_dim=%u (head_dim=%u)\n", rotaryDim, cfg.headDim);
    if (cfg.headDim != 128)
        fprintf(stderr, "  Note: head_dim=%u (attention kernels will be HD-patched)\n", cfg.headDim);
    if (cfg.headDim % 2 != 0) {
        fprintf(stderr, "Error: odd head_dim=%u is unsupported by RoPE\n", cfg.headDim);
        return false;
    }

    passPerDispatch = false;
    fprintf(stderr, "  Backend: %s, single-pass dispatch\n",
           gpu->backendType == WGPUBackendType_D3D12 ? "D3D12" : "Vulkan");

    // 2. Upload weights to GPU
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "  Uploading weights to GPU...\n");

    uint32_t qDimL = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;

    // Compute max per-layer dimensions for buffer allocation
    uint32_t maxQkvOut = qkvOutL;
    uint32_t maxIntermediateSize = cfg.intermediateSize;
    // For MoE archs (qwen35moe) without dense ffn dim, fall back to moe dim.
    if (maxIntermediateSize == 0 && cfg.moeIntermediateSize > 0) {
        maxIntermediateSize = std::max(cfg.moeIntermediateSize, cfg.moeSharedIntermediateSize);
        // For qwen35moe attention dispatch needs qDim_actual sized buffers too;
        // use a generous bound to cover all temp buffer uses.
        maxIntermediateSize = std::max(maxIntermediateSize, 4u * cfg.nEmbd);
    }
    uint32_t maxQDim = qDim;
    for (auto& pl : cfg.perLayer) {
        uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
        if (plQkvOut > maxQkvOut) maxQkvOut = plQkvOut;
        if (pl.intermediateSize > maxIntermediateSize) maxIntermediateSize = pl.intermediateSize;
        if (pl.qDim > maxQDim) maxQDim = pl.qDim;
    }

    // Zero bias buffers (sized to max across layers)
    uint32_t maxBias = std::max({cfg.nEmbd, maxQkvOut,
                                 2 * maxIntermediateSize, cfg.nVocab});
    fprintf(stderr, "  zero-bias sizing: nEmbd=%u maxQkvOut=%u maxIntermediateSize=%u maxBias=%u\n",
            cfg.nEmbd, maxQkvOut, maxIntermediateSize, maxBias);
    std::vector<float> zeros(maxBias, 0.0f);
    zeroBiasE   = gpu->createBuffer("zero_bias_E", cfg.nEmbd * 4);
    zeroBiasQKV = gpu->createBuffer("zero_bias_QKV", maxQkvOut * 4);
    zeroBiasGU  = gpu->createBuffer("zero_bias_GU", 2 * maxIntermediateSize * 4);
    zeroBiasV   = gpu->createBuffer("zero_bias_V", cfg.nVocab * 4);
    gpu->writeBuffer(zeroBiasE,   zeros.data(), cfg.nEmbd * 4);
    gpu->writeBuffer(zeroBiasQKV, zeros.data(), maxQkvOut * 4);
    gpu->writeBuffer(zeroBiasGU,  zeros.data(), 2 * maxIntermediateSize * 4);
    gpu->writeBuffer(zeroBiasV,   zeros.data(), cfg.nVocab * 4);

    // KV cache
    kvCache.resize(cfg.nLayer);
    uint64_t totalKvBytes = 0;
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& pl = cfg.perLayer[i];
        if (pl.kvSourceLayer >= 0) {
            // Shared KV: reuse source layer's buffers
            kvCache[i].K = kvCache[pl.kvSourceLayer].K;
            kvCache[i].V = kvCache[pl.kvSourceLayer].V;
            kvCache[i].len = 0;
        } else {
            uint32_t layerKvDim = pl.kvDim > 0 ? pl.kvDim : cfg.nKvHeads * cfg.headDim;
            uint64_t kvSize = (uint64_t)maxSeqLen * layerKvDim * 2;  // fp16
            kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
            kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
            kvCache[i].len = 0;
            totalKvBytes += kvSize * 2;
        }
    }
    fprintf(stderr, "  KV cache: %.0f MB (fp16, %u shared)\n",
            totalKvBytes / 1048576.0, cfg.sharedKvLayers);

    // Per-layer weights
    layerWeights.resize(cfg.nLayer);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        auto& ld = onnx.layers[i];

        if (ld.qkv.N > 0) {
            uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qkv",
                           ld.qkv, lw.qkvW, lw.qkvS);
        } else if (ld.qOnly.N > 0) {
            uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".q_only",
                           ld.qOnly, lw.qOnlyW, lw.qOnlyS);
            lw.qOnly = true;
        }
        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".o", ld.o, lw.oW, lw.oS);
        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".gu", ld.gateup, lw.guW, lw.guS);
        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".dn", ld.down, lw.dnW, lw.dnS);
        if (ld.pleInputGate.N > 0)
            uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".ple_gate",
                           ld.pleInputGate, lw.pleInpGateW, lw.pleInpGateS);
        if (ld.pleProjection.N > 0)
            uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".ple_proj",
                           ld.pleProjection, lw.pleProjW, lw.pleProjS);

        // Norm weights
        if (!ld.inputNorm.empty()) {
            lw.inputNorm = gpu->createBuffer("L" + std::to_string(i) + ".inorm",
                                              ld.inputNorm.size() * 4);
            gpu->writeBuffer(lw.inputNorm, ld.inputNorm.data(), ld.inputNorm.size() * 4);
        }
        if (!ld.postAttnNorm.empty()) {
            GPUBuffer& dst = cfg.arch == "gemma4" ? lw.postNorm : lw.postAttnNorm;
            dst = gpu->createBuffer("L" + std::to_string(i) + ".panorm",
                                                 ld.postAttnNorm.size() * 4);
            gpu->writeBuffer(dst, ld.postAttnNorm.data(), ld.postAttnNorm.size() * 4);
        }
        // QK norm (optional)
        if (!ld.qNorm.empty()) {
            lw.qNorm = gpu->createBuffer("L" + std::to_string(i) + ".qnorm",
                                          ld.qNorm.size() * 4);
            gpu->writeBuffer(lw.qNorm, ld.qNorm.data(), ld.qNorm.size() * 4);
        }
        if (!ld.kNorm.empty()) {
            lw.kNorm = gpu->createBuffer("L" + std::to_string(i) + ".knorm",
                                          ld.kNorm.size() * 4);
            gpu->writeBuffer(lw.kNorm, ld.kNorm.data(), ld.kNorm.size() * 4);
        }
        auto uploadNorm = [&](const std::string& name, const std::vector<float>& src,
                              GPUBuffer& dst) {
            if (src.empty()) return;
            dst = gpu->createBuffer(name, src.size() * 4);
            gpu->writeBuffer(dst, src.data(), src.size() * 4);
        };
        uploadNorm("L" + std::to_string(i) + ".ffn_norm", ld.preFfnNorm, lw.ffnNorm);
        uploadNorm("L" + std::to_string(i) + ".post_ffn_norm", ld.postFfnNorm, lw.postFfwNorm);
        uploadNorm("L" + std::to_string(i) + ".ple_post_norm", ld.plePostNorm, lw.plePostNorm);
        uploadNorm("L" + std::to_string(i) + ".output_scale", ld.outputScale, lw.outScale);

        if (i % 7 == 6 || i == cfg.nLayer - 1)
            fprintf(stderr, "  uploaded layer %u/%u\n", i + 1, cfg.nLayer);
    }

    // Final norm
    if (!onnx.finalNorm.empty()) {
        finalNormW = gpu->createBuffer("final_norm", onnx.finalNorm.size() * 4);
        gpu->writeBuffer(finalNormW, onnx.finalNorm.data(), onnx.finalNorm.size() * 4);
    }

    // Embedding
    embeddingCPU = std::move(onnx.embeddingCPU);
    fprintf(stderr, "  Embedding: %u × %u (fp32)\n", cfg.nVocab, cfg.nEmbd);

    if (cfg.pleSize > 0 && onnx.pleEmbedding.layers == cfg.nLayer) {
        auto uploadRaw = [&](const std::string& name, const std::vector<uint8_t>& src) {
            GPUBuffer b = gpu->createBuffer(name, src.size());
            constexpr uint64_t CHUNK = 64ull * 1024 * 1024;
            for (uint64_t off = 0; off < src.size(); off += CHUNK)
                gpu->writeBuffer(b, src.data() + off,
                    std::min<uint64_t>(CHUNK, src.size() - off), off);
            return b;
        };
        pleTokenEmbW = uploadRaw("ple_token_q4", onnx.pleEmbedding.weights);
        pleTokenEmbS = uploadRaw("ple_token_scales", onnx.pleEmbedding.scales);
        pleTokenEmbZ = uploadRaw("ple_token_zero_points", onnx.pleEmbedding.zeroPoints);
        pleTokenEmbAsymmetric = true;
        if (onnx.pleModelProjection.N > 0)
            uploadQ8Weight(*gpu, "ple_model_projection", onnx.pleModelProjection,
                           pleModelProjW, pleModelProjS);
        if (!onnx.pleProjectionNorm.empty()) {
            pleProjNormW = gpu->createBuffer("ple_projection_norm",
                                             onnx.pleProjectionNorm.size() * 4);
            gpu->writeBuffer(pleProjNormW, onnx.pleProjectionNorm.data(),
                             onnx.pleProjectionNorm.size() * 4);
        }
        pleGpuPreprocess = pleTokenEmbW.handle && pleTokenEmbS.handle &&
                           pleTokenEmbZ.handle && pleModelProjW.handle &&
                           pleProjNormW.handle;
        fprintf(stderr, "  PLE: packed asymmetric Q4, %u × %u × %u\n",
                onnx.pleEmbedding.layers, onnx.pleEmbedding.vocab,
                onnx.pleEmbedding.dim);
    }

    // LM head
    if (onnx.hasLmHeadQ8) {
        uploadQ8Weight(*gpu, "lm_head_q8", onnx.lmHeadQ8, lmHeadQ8W, lmHeadQ8S);
        lmHeadIsQ8 = true;
        fprintf(stderr, "  LM head: separate (Q8)\n");
    } else if (cfg.tieWordEmbeddings) {
        // Build Q8 LM head from embedding table
        uint32_t nBlocksPerRow = (cfg.nEmbd + 31) / 32;
        size_t totalBlocks = (size_t)cfg.nVocab * nBlocksPerRow;
        struct Q8B { uint16_t d; int8_t qs[32]; };
        std::vector<Q8B> blocks(totalBlocks);
        for (uint32_t row = 0; row < cfg.nVocab; row++) {
            for (uint32_t blk = 0; blk < nBlocksPerRow; blk++) {
                auto& b = blocks[row * nBlocksPerRow + blk];
                float maxAbs = 0.0f;
                for (int q = 0; q < 32; q++) {
                    uint32_t col = blk * 32 + q;
                    float v = (col < cfg.nEmbd) ? embeddingCPU[row * cfg.nEmbd + col] : 0.0f;
                    maxAbs = std::max(maxAbs, std::abs(v));
                }
                float scale = maxAbs / 127.0f;
                float inv = (scale > 0) ? 1.0f / scale : 0.0f;
                uint32_t fb; memcpy(&fb, &scale, 4);
                uint32_t s16 = (fb >> 16) & 0x8000;
                int32_t e = ((fb >> 23) & 0xFF) - 112;
                uint32_t m = (fb >> 13) & 0x3FF;
                if (e <= 0) b.d = (uint16_t)s16;
                else if (e > 30) b.d = (uint16_t)(s16 | 0x7C00);
                else b.d = (uint16_t)(s16 | (e << 10) | m);
                for (int q = 0; q < 32; q++) {
                    uint32_t col = blk * 32 + q;
                    float v = (col < cfg.nEmbd) ? embeddingCPU[row * cfg.nEmbd + col] : 0.0f;
                    int iv = (int)roundf(v * inv);
                    iv = std::max(-128, std::min(127, iv));
                    b.qs[q] = (int8_t)iv;
                }
            }
        }
        auto rep = repack_q8_0(blocks.data(), cfg.nVocab, nBlocksPerRow * 32);
        uploadQ8Weight(*gpu, "lm_head_q8", rep, lmHeadQ8W, lmHeadQ8S);
        lmHeadIsQ8 = true;
        fprintf(stderr, "  LM head: tied embeddings (Q8)\n");
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "  Weights uploaded in %lldms\n", (long long)ms);

    // 3. RoPE tables
    if (hasPrecomputedRope && !onnx.ropeCos.empty()) {
        // Upload pre-computed ONNX cos/sin tables
        uint32_t ropeHalf = onnx.ropeHalfDim;
        uint32_t maxPos = std::min(onnx.ropeMaxPositions, maxSeqLen);
        uint64_t ropeBytes = (uint64_t)maxPos * ropeHalf * 4;
        ropeCosBuf = gpu->createBuffer("rope_cos", ropeBytes);
        ropeSinBuf = gpu->createBuffer("rope_sin", ropeBytes);
        // Only upload up to maxSeqLen positions
        if (onnx.ropeMaxPositions > maxSeqLen) {
            // Truncate
            gpu->writeBuffer(ropeCosBuf, onnx.ropeCos.data(), ropeBytes);
            gpu->writeBuffer(ropeSinBuf, onnx.ropeSin.data(), ropeBytes);
        } else {
            gpu->writeBuffer(ropeCosBuf, onnx.ropeCos.data(), ropeBytes);
            gpu->writeBuffer(ropeSinBuf, onnx.ropeSin.data(), ropeBytes);
        }
        fprintf(stderr, "  Using ONNX RoPE cache: %u positions × %u half-dim\n",
               maxPos, ropeHalf);
    }
    computeRopeTables();

    // 4. Build decode pipeline (identical to GGUF path from here)
    buildDecodePipeline();

    fprintf(stderr, "[onnx] loadOnnx complete\n"); fflush(stderr);
    return true;
}

// ─── Load weights ────────────────────────────────────────────────────────────

void ModelRunner::loadWeights(const GGUFFile& gguf,
                               const uint8_t* fileData) {
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "  Loading %llu tensors...\n", (unsigned long long)gguf.n_tensors);

    if (std::getenv("BP_DUMP_GGUF_WEIGHT_TYPES")) {
        auto typeName = [](uint32_t type) {
            switch ((GGUFType)type) {
                case GGUF_TYPE_Q4_K: return "Q4_K";
                case GGUF_TYPE_Q5_K: return "Q5_K";
                case GGUF_TYPE_Q6_K: return "Q6_K";
                case GGUF_TYPE_Q8_0: return "Q8_0";
                case GGUF_TYPE_F16: return "F16";
                case GGUF_TYPE_F32: return "F32";
                default: return "other";
            }
        };
        for (const auto& tensor : gguf.tensors) {
            if (tensor.name.find(".weight") != std::string::npos)
                fprintf(stderr, "  [weight-type] %-48s %s (%u)\n",
                        tensor.name.c_str(), typeName(tensor.type), tensor.type);
        }
    }

    uint32_t qDim  = cfg.nHead * cfg.headDim;
    uint32_t kvDim = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOut = qDim + 2 * kvDim;
    // For MoE archs (qwen35moe) without dense FFN dim, use moe dim + generous
    // bound to cover qwen35moe attention temp buffers (qDim_actual up to 4*nEmbd).
    uint32_t effIntermediate = cfg.intermediateSize;
    if (effIntermediate == 0 && cfg.moeIntermediateSize > 0) {
        effIntermediate = std::max(cfg.moeIntermediateSize, cfg.moeSharedIntermediateSize);
        effIntermediate = std::max(effIntermediate, 4u * cfg.nEmbd);
    }
    uint32_t effQkvOut = qkvOut;
    // For qwen35moe attention dispatch: needs qOutDim=4*nEmbd for joint Q+gate.
    if ((cfg.arch == "qwen35" || cfg.numExperts > 0) && cfg.fullAttentionInterval > 0) {
        effQkvOut = std::max(effQkvOut, 4u * cfg.nEmbd);
    }

    // Zero bias buffers
    uint32_t maxBias = std::max({cfg.nEmbd, effQkvOut,
                                 2 * effIntermediate, cfg.nVocab});
    fprintf(stderr, "  zero-bias sizing (GGUF): nEmbd=%u effQkvOut=%u effIntermediate=%u maxBias=%u\n",
            cfg.nEmbd, effQkvOut, effIntermediate, maxBias);
    std::vector<float> zeros(maxBias, 0.0f);
    zeroBiasE   = gpu->createBuffer("zero_bias_E", cfg.nEmbd * 4);
    zeroBiasQKV = gpu->createBuffer("zero_bias_QKV", effQkvOut * 4);
    zeroBiasGU  = gpu->createBuffer("zero_bias_GU", 2 * effIntermediate * 4);
    zeroBiasV   = gpu->createBuffer("zero_bias_V", cfg.nVocab * 4);
    gpu->writeBuffer(zeroBiasE,   zeros.data(), cfg.nEmbd * 4);
    gpu->writeBuffer(zeroBiasQKV, zeros.data(), effQkvOut * 4);
    gpu->writeBuffer(zeroBiasGU,  zeros.data(), 2 * effIntermediate * 4);
    gpu->writeBuffer(zeroBiasV,   zeros.data(), cfg.nVocab * 4);

    // KV cache (fp16 — halves attention bandwidth)
    kvCache.resize(cfg.nLayer);
    uint64_t kvSize = (uint64_t)maxSeqLen * cfg.nKvHeads * cfg.headDim * 2;  // 2 bytes per f16
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        kvCache[i].K = gpu->createBuffer("kv_K_" + std::to_string(i), kvSize);
        kvCache[i].V = gpu->createBuffer("kv_V_" + std::to_string(i), kvSize);
        kvCache[i].len = 0;
    }
    fprintf(stderr, "  KV cache: %.0f MB (fp16)\n", cfg.nLayer * 2.0 * kvSize / 1048576.0);

    // Helper: load fp32/fp16 norm weight from GGUF tensor. Gemma stores its
    // RMSNorm weights as (real_weight - 1) so the kernel needs (1 + w) — bake
    // NOTE: Gemma's RMSNorm uses (1 + weight), but llama.cpp's GGUF conversion
    // already bakes the +1 into the stored norm weights (they sit around ~1-6,
    // not ~0). So we must NOT add 1 again here — the raw GGUF weight is used
    // directly, exactly like llama.cpp. (Adding +1 double-counts and blows up
    // the residual stream.)
    const bool gemmaNormBias = false;
    auto loadNorm = [&](const std::string& ggufName, GPUBuffer& buf) {
        auto it = gguf.tensor_index.find(ggufName);
        if (it == gguf.tensor_index.end()) return;
        auto& ti = gguf.tensors[it->second];
        const uint8_t* data = fileData + gguf.data_offset + ti.offset;
        uint32_t nel = 1;
        for (auto d : ti.shape) nel *= (uint32_t)d;
        std::vector<float> fp32(nel);
        if (ti.type == GGUF_TYPE_F16) {
            const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
            for (uint32_t j = 0; j < nel; j++)
                fp32[j] = fp16_to_f32(fp16[j]);
        } else {
            memcpy(fp32.data(), data, nel * 4);
        }
        if (gemmaNormBias) {
            for (uint32_t j = 0; j < nel; j++) fp32[j] += 1.0f;
        }
        buf = gpu->createBuffer(ggufName, nel * 4);
        gpu->writeBuffer(buf, fp32.data(), nel * 4);
    };

    // Helper: upload K-quant weight buffer
    auto uploadKQWeight = [&](const std::string& name, const KQuantPacked& kq,
                              GPUBuffer& buf) {
        uint64_t bytes = (uint64_t)kq.data.size() * 4;
        buf = gpu->createBuffer(name, bytes);
        gpu->writeBuffer(buf, kq.data.data(), bytes);
    };

    // Helper: repack any quantized tensor to Q8_0 format for GPU
    // For Q8_0: direct repack. For other types: dequant→fp32→quantize→Q8.
    auto repackToQ8 = [&](const uint8_t* data, uint32_t N, uint32_t K, GGUFType type) -> Q8Repacked {
        if (type == GGUF_TYPE_Q8_0) {
            return repack_q8_0(data, N, K);
        }
        // Dequantize to fp32 then quantize to Q8_0
        std::vector<float> fp32((size_t)N * K);
        dequant_tensor(data, fp32.data(), N, K, type);
        // Quantize fp32 → Q8_0 blocks then repack
        uint32_t nBlocksPerRow = K / 32;
        size_t totalBlocks = (size_t)N * nBlocksPerRow;
        std::vector<Q8_0Block> blocks(totalBlocks);
        for (uint32_t r = 0; r < N; r++) {
            for (uint32_t b = 0; b < nBlocksPerRow; b++) {
                auto& blk = blocks[r * nBlocksPerRow + b];
                float amax = 0;
                for (int j = 0; j < 32; j++) {
                    float v = fp32[r * K + b * 32 + j];
                    if (fabsf(v) > amax) amax = fabsf(v);
                }
                float d = amax / 127.0f;
                float id = (d != 0.0f) ? 1.0f / d : 0.0f;
                blk.d = f32_to_fp16(d);
                for (int j = 0; j < 32; j++) {
                    float v = fp32[r * K + b * 32 + j];
                    int q = (int)roundf(v * id);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    blk.qs[j] = (int8_t)q;
                }
            }
        }
        return repack_q8_0(blocks.data(), N, K);
    };

    // Helper: fuse multiple K-quant tensors vertically (concat rows)
    auto fuseKQ = [](const KQuantPacked& a, const KQuantPacked& b) -> KQuantPacked {
        KQuantPacked fused;
        fused.N = a.N + b.N;
        fused.K = a.K;
        fused.rowStrideWords = a.rowStrideWords;
        fused.nBlocks = a.nBlocks;
        fused.data.reserve(a.data.size() + b.data.size());
        fused.data.insert(fused.data.end(), a.data.begin(), a.data.end());
        fused.data.insert(fused.data.end(), b.data.begin(), b.data.end());
        return fused;
    };

    // Detect weight quantization type from first weight tensor
    // For mixed-quant models (Q4_K_M), use Q8 pipeline (dequant all to Q8)
    {
        auto it = gguf.tensor_index.find("blk.0.attn_q.weight");
        if (it == gguf.tensor_index.end())
            it = gguf.tensor_index.find("blk.0.attn_qkv.weight");
        if (it == gguf.tensor_index.end())
            it = gguf.tensor_index.find("blk.0.ffn_gate.weight");
        if (it == gguf.tensor_index.end())
            it = gguf.tensor_index.find("blk.0.ffn_down.weight");
        if (it != gguf.tensor_index.end()) {
            weightQuantType = (GGUFType)gguf.tensors[it->second].type;
        }
    }
    // Only use K-quant pipeline if ALL primary weight tensors are K-quant
    // Check multiple tensor types to detect mixed quantization
    bool allSameKQ = true;
    {
        const char* checkNames[] = {"blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"};
        for (auto name : checkNames) {
            auto it = gguf.tensor_index.find(name);
            if (it != gguf.tensor_index.end()) {
                auto t = (GGUFType)gguf.tensors[it->second].type;
                if ((t != GGUF_TYPE_Q4_K && t != GGUF_TYPE_Q5_K && t != GGUF_TYPE_Q6_K) ||
                    t != weightQuantType) {
                    allSameKQ = false;
                    break;
                }
            }
        }
    }
    bool isKQuant = allSameKQ &&
                    (weightQuantType == GGUF_TYPE_Q4_K ||
                     weightQuantType == GGUF_TYPE_Q5_K ||
                     weightQuantType == GGUF_TYPE_Q6_K);
    weightsUseNativeKQ = isKQuant;
    // The decode kernels below implement logical 32-lane warps with XOR
    // shuffles.  NVIDIA and Intel D3D12 adapters have been validated with
    // that mapping, while AMD exposes wave64 here and produces incorrect
    // Gemma logits.  Keep AMD on the conformant Q8-expanded path until a
    // wave64-native kernel is available.
    const bool nativeQ4AdapterValidated =
        gpu->adapterName.find("AMD") == std::string::npos;
    weightsAreNativeQ4 = cfg.arch == "gemma4" &&
                         weightQuantType == GGUF_TYPE_Q4_0 &&
                         nativeQ4AdapterValidated &&
                         !std::getenv("BP_DISABLE_NATIVE_Q4");
    auto packKQ = [&](const void* data, uint32_t N, uint32_t K, GGUFType type) -> KQuantPacked {
        switch (type) {
            case GGUF_TYPE_Q4_K: return pack_q4k(data, N, K);
            case GGUF_TYPE_Q5_K: return pack_q5k(data, N, K);
            case GGUF_TYPE_Q6_K: return pack_q6k(data, N, K);
            default: return {};
        }
    };
    const char* kqName = !isKQuant && cfg.arch == "qwen35" ? "mixed K-quant (per-tensor native)" :
                         !isKQuant ? "mixed K-quant → Q8" :
                         (weightQuantType == GGUF_TYPE_Q4_K) ? "Q4_K" :
                         (weightQuantType == GGUF_TYPE_Q5_K) ? "Q5_K" :
                         (weightQuantType == GGUF_TYPE_Q6_K) ? "Q6_K" : "Q8_0";
    fprintf(stderr, "  Weight format: %s\n",
            weightsAreNativeQ4 ? "Q4_0 native" : kqName);
    auto repackPrimary = [&](const uint8_t* data, uint32_t N, uint32_t K,
                             GGUFType type) -> Q8Repacked {
        if (weightsAreNativeQ4 && type == GGUF_TYPE_Q4_0)
            return repackQ4_0Native(data, N, K);
        return repackToQ8(data, N, K, type);
    };

    // Infer per-layer dimensions from tensor shapes (Gemma 4 has variable dims)
    cfg.perLayer.resize(cfg.nLayer);
    cfg.hasPerLayerDims = false;
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto pfx = "blk." + std::to_string(i) + ".";
        auto& pl = cfg.perLayer[i];
        pl.headDim = cfg.headDim;
        pl.qDim = cfg.nHead * cfg.headDim;
        pl.kvDim = cfg.nKvHeads * cfg.headDim;
        pl.intermediateSize = cfg.intermediateSize;
        pl.kvSourceLayer = -1;

        // Infer from Q tensor shape: Q=[E, qDim] or fused QKV=[E, qDim+2*kvDim]
        auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
        if (qi != gguf.tensor_index.end()) {
            auto& qt = gguf.tensors[qi->second];
            if (qt.shape.size() >= 2) {
                pl.qDim = (uint32_t)qt.shape[1];
                if (cfg.arch == "qwen35" && cfg.fullAttentionInterval > 0 &&
                    cfg.isAttentionLayer(i)) {
                    pl.qDim /= 2u; // attn_q.weight is joint Q+gate.
                }
                pl.headDim = pl.qDim / cfg.nHead;
            }
        } else {
            // Try fused QKV: qkv=[E, qDim + 2*kvDim]
            auto qkvi = gguf.tensor_index.find(pfx + "attn_qkv.weight");
            if (qkvi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qkvi->second];
                if (qt.shape.size() >= 2) {
                    // qkvOut = qDim + 2*kvDim; keep defaults since we can't decompose
                }
            }
        }
        auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
        if (ki != gguf.tensor_index.end()) {
            auto& kt = gguf.tensors[ki->second];
            if (kt.shape.size() >= 2)
                pl.kvDim = (uint32_t)kt.shape[1];
        }
        auto gi = gguf.tensor_index.find(pfx + "ffn_gate.weight");
        if (gi != gguf.tensor_index.end()) {
            auto& gt = gguf.tensors[gi->second];
            if (gt.shape.size() >= 2)
                pl.intermediateSize = (uint32_t)gt.shape[1];
        }

        if (i > 0 && (pl.qDim != cfg.perLayer[0].qDim ||
                      pl.kvDim != cfg.perLayer[0].kvDim ||
                      pl.intermediateSize != cfg.perLayer[0].intermediateSize))
            cfg.hasPerLayerDims = true;
    }

    // Gemma 4: derive sliding-window vs global attention from the per-layer
    // head dim (sliding layers are key_length_swa=256 wide, global layers
    // key_length=512). This is more reliable than a hardcoded period and
    // matches the actual tensor shapes; the GGUF pattern heuristic was wrong.
    if (cfg.arch == "gemma4" && cfg.hasPerLayerDims && !cfg.layerAttnTypes.empty()) {
        uint32_t maxHd = 0;
        for (uint32_t i = 0; i < cfg.nLayer; i++)
            maxHd = std::max(maxHd, cfg.perLayer[i].headDim);
        for (uint32_t i = 0; i < cfg.nLayer && i < cfg.layerAttnTypes.size(); i++)
            cfg.layerAttnTypes[i] = (cfg.perLayer[i].headDim == maxHd)
                ? AttnLayerType::Global : AttnLayerType::SlidingWindow;
    }

    // Shared KV mapping
    if (cfg.sharedKvLayers > 0) {
        uint32_t kvStart = cfg.nLayer - cfg.sharedKvLayers;
        // Find the last non-shared sliding and global layers
        int lastSliding = -1, lastGlobal = -1;
        for (uint32_t i = 0; i < kvStart; i++) {
            if (!cfg.layerAttnTypes.empty() && i < cfg.layerAttnTypes.size() &&
                cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow)
                lastSliding = (int)i;
            else
                lastGlobal = (int)i;
        }
        for (uint32_t i = kvStart; i < cfg.nLayer; i++) {
            bool isSWA = !cfg.layerAttnTypes.empty() && i < cfg.layerAttnTypes.size() &&
                         cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            cfg.perLayer[i].kvSourceLayer = isSWA ? lastSliding : lastGlobal;
        }
        fprintf(stderr, "  Shared KV: %u layers (from layer %u), sliding→L%d, global→L%d\n",
                cfg.sharedKvLayers, kvStart, lastSliding, lastGlobal);
    }

    if (cfg.hasPerLayerDims)
        fprintf(stderr, "  Variable per-layer dims detected\n");
    if (cfg.hasSandwichNorm)
        fprintf(stderr, "  Sandwich norms: 4 norms per layer\n");
    if (cfg.pleSize > 0)
        fprintf(stderr, "  PLE: dim=%u\n", cfg.pleSize);

    // Per-layer weights
    layerWeights.resize(cfg.nLayer);
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto pfx = "blk." + std::to_string(i) + ".";
        auto& lw = layerWeights[i];
        auto& pl = cfg.perLayer[i];
        uint32_t layerQDim = pl.qDim;
        uint32_t layerKvDim = pl.kvDim;
        uint32_t layerQkvOut = layerQDim + 2 * layerKvDim;
        uint32_t layerIM = pl.intermediateSize;

        // qwen35moe attention layer: load Q/K/V SEPARATELY (no fuse) so Q can be
        // joint Q+gate (2x normal Q dim). Skip the fuse path below.
        bool isQ35AttnLayer = ((cfg.arch == "qwen35" || cfg.numExperts > 0) && cfg.fullAttentionInterval > 0 &&
                               cfg.isAttentionLayer(i));
        if (isQ35AttnLayer) {
            auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
            auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
            auto vi = gguf.tensor_index.find(pfx + "attn_v.weight");
            if (qi != gguf.tensor_index.end() && ki != gguf.tensor_index.end() && vi != gguf.tensor_index.end()) {
                auto& qt = gguf.tensors[qi->second];
                auto& kt = gguf.tensors[ki->second];
                auto& vt = gguf.tensors[vi->second];
                // Native output dims from tensor shapes
                uint32_t qOutDim = (uint32_t)qt.shape[1];   // 2*qDim (joint Q+gate)
                uint32_t kOutDim = (uint32_t)kt.shape[1];   // kvDim (decoupled key_length)
                uint32_t vOutDim = (uint32_t)vt.shape[1];   // value_length
                auto loadExact = [&](const GGUFTensorInfo& t, uint32_t N,
                                     const std::string& suffix, GPUBuffer& kqBuf,
                                     GGUFType& type, uint32_t& nb, uint32_t& rs,
                                     GPUBuffer& q8w, GPUBuffer& q8s) {
                    type = (GGUFType)t.type;
                    const uint8_t* src = fileData + gguf.data_offset + t.offset;
                    if (type == GGUF_TYPE_Q4_K || type == GGUF_TYPE_Q5_K ||
                        type == GGUF_TYPE_Q6_K) {
                        auto packed = packKQ(src, N, cfg.nEmbd, type);
                        nb = packed.nBlocks; rs = packed.rowStrideWords;
                        uploadKQWeight("L" + std::to_string(i) + suffix, packed, kqBuf);
                    } else {
                        auto rep = repackToQ8(src, N, cfg.nEmbd, type);
                        uploadQ8Weight(*gpu, "L" + std::to_string(i) + suffix, rep, q8w, q8s);
                    }
                };
                loadExact(qt,qOutDim,".qj_kq",lw.qjKQ,lw.qjKQType,lw.qjKQNBlocks,lw.qjKQRowStride,lw.qjW,lw.qjS);
                loadExact(kt,kOutDim,".k_kq",lw.kSepKQ,lw.kSepKQType,lw.kSepKQNBlocks,lw.kSepKQRowStride,lw.kSepW,lw.kSepS);
                loadExact(vt,vOutDim,".v_kq",lw.vSepKQ,lw.vSepKQType,lw.vSepKQNBlocks,lw.vSepKQRowStride,lw.vSepW,lw.vSepS);
                if (i == 3) {
                    fprintf(stderr, "  qwen35 attn layer %u: Q(2x)=%u K=%u V=%u (E=%u)\n",
                            i, qOutDim, kOutDim, vOutDim, cfg.nEmbd);
                }
            }
        }

        // Fuse Q/K/V into single QKV (or load pre-fused attn_qkv.weight for archs that ship it)
        if (!isQ35AttnLayer) {
        {
            auto qi = gguf.tensor_index.find(pfx + "attn_q.weight");
            auto ki = gguf.tensor_index.find(pfx + "attn_k.weight");
            auto vi = gguf.tensor_index.find(pfx + "attn_v.weight");
            auto qkvi_pre = gguf.tensor_index.find(pfx + "attn_qkv.weight");
            if (qi == gguf.tensor_index.end() && qkvi_pre != gguf.tensor_index.end()) {
                // Pre-fused QKV (qwen35moe / some MoE archs)
                auto& qkvt = gguf.tensors[qkvi_pre->second];
                uint32_t qkvN = (uint32_t)qkvt.shape[1];  // out-dim = qDim + 2*kvDim
                bool tensorIsKQ =
                    (qkvt.type == GGUF_TYPE_Q4_K || qkvt.type == GGUF_TYPE_Q5_K || qkvt.type == GGUF_TYPE_Q6_K);
                const uint8_t* src = fileData + gguf.data_offset + qkvt.offset;
                if (tensorIsKQ) {
                    KQuantPacked kq;
                    switch ((GGUFType)qkvt.type) {
                        case GGUF_TYPE_Q4_K: kq = pack_q4k(src, qkvN, cfg.nEmbd); break;
                        case GGUF_TYPE_Q5_K: kq = pack_q5k(src, qkvN, cfg.nEmbd); break;
                        case GGUF_TYPE_Q6_K: kq = pack_q6k(src, qkvN, cfg.nEmbd); break;
                        default: break;
                    }
                    lw.qkvKQType = (GGUFType)qkvt.type;
                    lw.qkvKQNBlocks = kq.nBlocks;
                    lw.qkvKQRowStride = kq.rowStrideWords;
                    if (i == 0) { kqQkvNBlocks = kq.nBlocks; kqQkvRowStride = kq.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".qkv_kq", kq, lw.qkvKQ);
                } else {
                    auto rep = repackToQ8(src, qkvN, cfg.nEmbd, (GGUFType)qkvt.type);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qkv", rep, lw.qkvW, lw.qkvS);
                }
                if (i == 0) {
                    fprintf(stderr, "  Pre-fused attn_qkv loaded: %u x %u, type=%u, %s path\n",
                            qkvN, cfg.nEmbd, (unsigned)qkvt.type, tensorIsKQ ? "K-quant" : "Q8");
                }
            } else if (qi != gguf.tensor_index.end()) {
                if (ki == gguf.tensor_index.end() || vi == gguf.tensor_index.end()) {
                    // Shared-KV layers (Gemma 4 layers 15+) ship attn_q but not
                    // attn_k/attn_v — the layer reuses K/V from a source layer.
                    // Load Q-only so a dedicated Q-projection + shared-KV
                    // attention path can run at decode time.
                    auto& qt = gguf.tensors[qi->second];
                    uint32_t qN = (uint32_t)qt.shape[1];
                    auto qr = repackPrimary(fileData + gguf.data_offset + qt.offset,
                                         qN, cfg.nEmbd, (GGUFType)qt.type);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qonly",
                                   qr, lw.qOnlyW, lw.qOnlyS);
                    lw.qOnly = true;
                } else {
                    auto& qt = gguf.tensors[qi->second];
                    auto& kt = gguf.tensors[ki->second];
                    auto& vt = gguf.tensors[vi->second];
                    bool allKQ = isKQuant &&
                        (qt.type == GGUF_TYPE_Q4_K || qt.type == GGUF_TYPE_Q5_K || qt.type == GGUF_TYPE_Q6_K) &&
                        (kt.type == GGUF_TYPE_Q4_K || kt.type == GGUF_TYPE_Q5_K || kt.type == GGUF_TYPE_Q6_K) &&
                        (vt.type == GGUF_TYPE_Q4_K || vt.type == GGUF_TYPE_Q5_K || vt.type == GGUF_TYPE_Q6_K);
                    if (allKQ) {
                        auto qp = packKQ(fileData + gguf.data_offset + qt.offset, layerQDim, cfg.nEmbd, (GGUFType)qt.type);
                        auto kp = packKQ(fileData + gguf.data_offset + kt.offset, layerKvDim, cfg.nEmbd, (GGUFType)kt.type);
                        auto vp = packKQ(fileData + gguf.data_offset + vt.offset, layerKvDim, cfg.nEmbd, (GGUFType)vt.type);
                        auto fused = fuseKQ(fuseKQ(qp, kp), vp);
                        if (i == 0) { kqQkvNBlocks = fused.nBlocks; kqQkvRowStride = fused.rowStrideWords; }
                        uploadKQWeight("L" + std::to_string(i) + ".qkv_kq", fused, lw.qkvKQ);
                    } else {
                        auto qr = repackPrimary(fileData + gguf.data_offset + qt.offset, layerQDim, cfg.nEmbd, (GGUFType)qt.type);
                        auto kr = repackPrimary(fileData + gguf.data_offset + kt.offset, layerKvDim, cfg.nEmbd, (GGUFType)kt.type);
                        auto vr = repackPrimary(fileData + gguf.data_offset + vt.offset, layerKvDim, cfg.nEmbd, (GGUFType)vt.type);
                        auto fused = concatRepacked(concatRepacked(qr, kr), vr);
                        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".qkv", fused, lw.qkvW, lw.qkvS);
                    }
                }
            }
        }
        }  // close if (!isQ35AttnLayer)

        // O projection
        {
            auto it = gguf.tensor_index.find(pfx + "attn_output.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                    bool tensorIsKQ = (isKQuant || cfg.arch == "qwen35") &&
                        (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K || ti.type == GGUF_TYPE_Q6_K);
                    if (tensorIsKQ) {
                        auto kq = packKQ(fileData + gguf.data_offset + ti.offset, cfg.nEmbd, layerQDim, (GGUFType)ti.type);
                        lw.oKQType = (GGUFType)ti.type;
                        lw.oKQNBlocks = kq.nBlocks;
                        lw.oKQRowStride = kq.rowStrideWords;
                        if (i == 0) { kqONBlocks = kq.nBlocks; kqORowStride = kq.rowStrideWords; }
                        uploadKQWeight("L" + std::to_string(i) + ".o_kq", kq, lw.oKQ);
                    } else {
                        auto rep = repackPrimary(fileData + gguf.data_offset + ti.offset,
                                                cfg.nEmbd, layerQDim, (GGUFType)ti.type);
                        uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".o", rep, lw.oW, lw.oS);
                    }
            }
        }

        // Fuse gate + up
        {
            auto gi = gguf.tensor_index.find(pfx + "ffn_gate.weight");
            auto ui = gguf.tensor_index.find(pfx + "ffn_up.weight");
            if (gi != gguf.tensor_index.end() && ui != gguf.tensor_index.end()) {
                auto& gt = gguf.tensors[gi->second];
                auto& ut = gguf.tensors[ui->second];
                bool bothKQ = (isKQuant || cfg.arch == "qwen35") && gt.type == ut.type &&
                    (gt.type == GGUF_TYPE_Q4_K || gt.type == GGUF_TYPE_Q5_K || gt.type == GGUF_TYPE_Q6_K) &&
                    (ut.type == GGUF_TYPE_Q4_K || ut.type == GGUF_TYPE_Q5_K || ut.type == GGUF_TYPE_Q6_K);
                if (bothKQ) {
                    auto gp = packKQ(fileData + gguf.data_offset + gt.offset, layerIM, cfg.nEmbd, (GGUFType)gt.type);
                    auto up = packKQ(fileData + gguf.data_offset + ut.offset, layerIM, cfg.nEmbd, (GGUFType)ut.type);
                    auto fused = fuseKQ(gp, up);
                    lw.guKQType = (GGUFType)gt.type;
                    lw.guKQNBlocks = fused.nBlocks;
                    lw.guKQRowStride = fused.rowStrideWords;
                    if (i == 0) { kqGuNBlocks = fused.nBlocks; kqGuRowStride = fused.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".gu_kq", fused, lw.guKQ);
                } else {
                    auto gr = repackPrimary(fileData + gguf.data_offset + gt.offset,
                                           layerIM, cfg.nEmbd, (GGUFType)gt.type);
                    auto ur = repackPrimary(fileData + gguf.data_offset + ut.offset,
                                           layerIM, cfg.nEmbd, (GGUFType)ut.type);
                    auto fused = concatRepacked(gr, ur);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".gu", fused, lw.guW, lw.guS);
                }
            }
        }

        // Down projection
        {
            auto it = gguf.tensor_index.find(pfx + "ffn_down.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                bool tensorIsKQ = (isKQuant || cfg.arch == "qwen35") &&
                    (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K || ti.type == GGUF_TYPE_Q6_K);
                if (tensorIsKQ) {
                    auto kq = packKQ(fileData + gguf.data_offset + ti.offset, cfg.nEmbd, layerIM, (GGUFType)ti.type);
                    lw.dnKQType = (GGUFType)ti.type;
                    lw.dnKQNBlocks = kq.nBlocks;
                    lw.dnKQRowStride = kq.rowStrideWords;
                    if (i == 0) { kqDnNBlocks = kq.nBlocks; kqDnRowStride = kq.rowStrideWords; }
                    uploadKQWeight("L" + std::to_string(i) + ".dn_kq", kq, lw.dnKQ);
                } else {
                    auto rep = repackPrimary(fileData + gguf.data_offset + ti.offset,
                                            cfg.nEmbd, layerIM, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, "L" + std::to_string(i) + ".dn", rep, lw.dnW, lw.dnS);
                }
            }
        }

        // Norm weights
        loadNorm(pfx + "attn_norm.weight", lw.inputNorm);
        if (cfg.hasSandwichNorm) {
            // Gemma sandwich (4 norms): pre-attn, post-attn, pre-FFN, post-FFN.
            // The post-attention norm is post_attention_norm.weight on all Gemma
            // versions. Gemma 4 additionally ships a separate post_norm.weight
            // (a per-layer output norm) — load it into postLayerNorm, not the
            // post-attention slot.
            loadNorm(pfx + "ffn_norm.weight", lw.ffnNorm);  // pre-FFN norm
            loadNorm(pfx + "post_attention_norm.weight", lw.postNorm);
            if (!lw.postNorm.handle)
                loadNorm(pfx + "attn_post_norm.weight", lw.postNorm);
            if (!lw.postNorm.handle)
                loadNorm(pfx + "post_norm.weight", lw.postNorm);
            loadNorm(pfx + "post_ffw_norm.weight", lw.postFfwNorm); // post-FFN sandwich
            if (!lw.postFfwNorm.handle)
                loadNorm(pfx + "ffn_post_norm.weight", lw.postFfwNorm);
        } else {
            // Standard: 2-norm (pre-attn + pre-FFN fused with residual add)
            loadNorm(pfx + "ffn_norm.weight", lw.postAttnNorm);
            // Fallback for archs (qwen35moe / hybrid Mamba-MoE) that use
            // post_attention_norm.weight instead of ffn_norm.weight.
            if (!lw.postAttnNorm.handle)
                loadNorm(pfx + "post_attention_norm.weight", lw.postAttnNorm);
        }
        loadNorm(pfx + "attn_q_norm.weight", lw.qNorm);
        loadNorm(pfx + "attn_k_norm.weight", lw.kNorm);

        // PLE weights (Gemma 4)
        if (cfg.pleSize > 0) {
            auto loadWeight = [&](const std::string& name, uint32_t N, uint32_t K,
                                   GPUBuffer& wBuf, GPUBuffer& sBuf) {
                auto it = gguf.tensor_index.find(name);
                if (it != gguf.tensor_index.end()) {
                    auto& ti = gguf.tensors[it->second];
                    if (weightsAreNativeQ4 && ti.type == GGUF_TYPE_Q4_0 &&
                        !std::getenv("BP_Q8_PLE")) {
                        auto rep = repackQ4_0Native(
                            fileData + gguf.data_offset + ti.offset, N, K);
                        uploadQ8Weight(*gpu, name + ".q4", rep, wBuf, sBuf);
                        pleWeightsUseFp16 = true;
                    } else {
                        auto rep = repackPrimary(fileData + gguf.data_offset + ti.offset,
                                               N, K, (GGUFType)ti.type);
                        uploadQ8Weight(*gpu, name, rep, wBuf, sBuf);
                    }
                }
            };
            loadWeight(pfx + "inp_gate.weight", cfg.pleSize, cfg.nEmbd,
                        lw.pleInpGateW, lw.pleInpGateS);
            loadWeight(pfx + "proj.weight", cfg.nEmbd, cfg.pleSize,
                        lw.pleProjW, lw.pleProjS);
            loadNorm(pfx + "per_layer_post_norm.weight", lw.plePostNorm);
            if (!lw.plePostNorm.handle)
                loadNorm(pfx + "post_norm_ple.weight", lw.plePostNorm);
            // Gemma 4: the per-layer PLE-injection norm (post_per_layer_input_norm)
            // is stored as blk.N.post_norm.weight.
            if (!lw.plePostNorm.handle)
                loadNorm(pfx + "post_norm.weight", lw.plePostNorm);
        }

        // Per-layer output scale
        {
            auto it = gguf.tensor_index.find(pfx + "layer_output_scale.weight");
            if (it != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it->second];
                const uint8_t* data = fileData + gguf.data_offset + ti.offset;
                float scale;
                if (ti.type == GGUF_TYPE_F32) {
                    memcpy(&scale, data, 4);
                } else if (ti.type == GGUF_TYPE_F16) {
                    uint16_t h; memcpy(&h, data, 2);
                    uint32_t sign=(h>>15)&1, exp=(h>>10)&0x1F, mant=h&0x3FF, f;
                    if(exp==0)f=(sign<<31)|(mant<<13);
                    else if(exp==31)f=(sign<<31)|0x7F800000|(mant<<13);
                    else f=(sign<<31)|((exp+112)<<23)|(mant<<13);
                    memcpy(&scale, &f, 4);
                } else {
                    scale = 1.0f;
                }
                // Gemma 4 stores per-layer output scales as (real - 1): the raw
                // values sit near 0 (e.g. 0.02, 0.24, 0.79) and are applied as a
                // multiplier on the whole residual (xBuf *= scale). Loading them
                // as-is multiplies layer 0's residual by ~0.02 and annihilates the
                // signal, so the +1 must always be baked in here — unlike the
                // RMSNorm weights, which llama.cpp already stores with the +1 baked
                // (they sit ~1-56, see gemmaNormBias=false). Different conventions.
                // Gemma 4's per-layer output scale is applied as-is (raw, small
                // values ~0.02-0.79 that damp each layer's residual). Empirically
                // baking +1 (giving ~1.02-1.79) makes the residual explode over
                // depth and produces garbage, and disabling it also breaks output;
                // the raw value is correct. NOT the (real-1) convention despite the
                // small magnitude.
                lw.outScale = gpu->createBuffer("L" + std::to_string(i) + ".out_scale", 4);
                gpu->writeBuffer(lw.outScale, &scale, 4);
            }
        }

        // ── MoE weights (qwen35moe and similar) ─────────────────────────────
        // Loads the small, always-resident MoE weights (router, shared expert,
        // attention gate). The large 3D routed-expert tensors are intentionally
        // NOT uploaded here — they need a native IQ GPU decode path (Phase 4)
        // because dequantizing them to Q8 would expand the model from 13 GiB
        // to ~31 GiB which won't fit on a 16 GiB GPU. A separate loader pass
        // (TODO) will keep them in their native IQ format as raw GPU bytes.
        if (cfg.numExperts > 0) {
            auto loadMoeSmall = [&](const std::string& name, uint32_t N, uint32_t K,
                                     GPUBuffer& wBuf, GPUBuffer& sBuf) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                       N, K, (GGUFType)ti.type);
                uploadQ8Weight(*gpu, name, rep, wBuf, sBuf);
                return true;
            };
            // Router: [nExperts, E]
            loadMoeSmall(pfx + "ffn_gate_inp.weight", cfg.numExperts, cfg.nEmbd,
                          lw.routerW, lw.routerS);
            // Shared-expert router (gating scalar) — shape varies; try common forms
            loadMoeSmall(pfx + "ffn_gate_inp_shexp.weight", 1, cfg.nEmbd,
                          lw.shexpRouterW, lw.shexpRouterS);
            // Shared expert (always active). IM_s = moeIntermediateSize for now;
            // the actual tensor shape will override if different (TODO: read shape).
            uint32_t IMs = cfg.moeSharedIntermediateSize;
            loadMoeSmall(pfx + "ffn_gate_shexp.weight", IMs, cfg.nEmbd,
                          lw.shexpGateW, lw.shexpGateS);
            loadMoeSmall(pfx + "ffn_up_shexp.weight",   IMs, cfg.nEmbd,
                          lw.shexpUpW,   lw.shexpUpS);
            loadMoeSmall(pfx + "ffn_down_shexp.weight", cfg.nEmbd, IMs,
                          lw.shexpDownW, lw.shexpDownS);
            // Attention gate (Qwen3.6 gated attention output)
            loadMoeSmall(pfx + "attn_gate.weight", cfg.nEmbd, cfg.nEmbd,
                          lw.attnGateW, lw.attnGateS);
            if (i == 0) {
                fprintf(stderr, "  MoE small weights loaded for layer 0 "
                        "(router%s, shexp%s, attn_gate%s)\n",
                        lw.routerW.handle ? "=ok" : "=MISSING",
                        lw.shexpGateW.handle ? "=ok" : "=MISSING",
                        lw.attnGateW.handle ? "=ok" : "=MISSING");
            }

            // ── Routed-expert weights (3D tensors, native IQ format on GPU) ─
            // Keep raw IQ bytes packed per-block (no dequant) so 13 GiB model
            // fits in 16 GiB VRAM. Shapes per llama.cpp / GGUF convention:
            //   ffn_gate_exps.weight: [nExperts, IM_e, E]
            //   ffn_up_exps.weight:   [nExperts, IM_e, E]
            //   ffn_down_exps.weight: [nExperts, E, IM_e]
            auto loadExpertsRaw = [&](const std::string& name,
                                       uint32_t rows_per_expert, uint32_t cols_per_expert,
                                       GPUBuffer& wBuf, uint32_t* outType) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                const uint8_t* src = fileData + gguf.data_offset + ti.offset;
                if (outType) *outType = (uint32_t)ti.type;
                uint32_t total_rows = cfg.numExperts * rows_per_expert;
                KQuantPacked packed;
                switch ((GGUFType)ti.type) {
                    case GGUF_TYPE_IQ3_S: packed = pack_iq3s(src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_IQ2_S: packed = pack_iq2s(src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_IQ4_XS:packed = pack_iq4xs(src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q2_K:  packed = pack_q2k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q3_K:  packed = pack_q3k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q4_K:  packed = pack_q4k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q5_K:  packed = pack_q5k (src, total_rows, cols_per_expert); break;
                    case GGUF_TYPE_Q6_K:  packed = pack_q6k (src, total_rows, cols_per_expert); break;
                    default:
                        fprintf(stderr, "  WARN: routed-expert quant type %u not handled for %s\n",
                                (unsigned)ti.type, name.c_str());
                        return false;
                }
                size_t bytes = (size_t)packed.data.size() * 4;
                wBuf = gpu->createBuffer(name, bytes);
                gpu->writeBuffer(wBuf, packed.data.data(), bytes);
                if (i == 0) {
                    fprintf(stderr, "    %s: %u rows × %u cols, type=%u, %.1f MB on GPU\n",
                            name.c_str(), total_rows, cols_per_expert,
                            (unsigned)ti.type, bytes / 1048576.0);
                }
                return true;
            };
            uint32_t IMe = cfg.moeIntermediateSize;
            if (i == 0) {
                moeExpertsGateType.resize(cfg.nLayer, 0);
                moeExpertsUpType.resize(cfg.nLayer, 0);
                moeExpertsDownType.resize(cfg.nLayer, 0);
            }
            loadExpertsRaw(pfx + "ffn_gate_exps.weight", IMe, cfg.nEmbd, lw.expertsGateW, &moeExpertsGateType[i]);
            loadExpertsRaw(pfx + "ffn_up_exps.weight",   IMe, cfg.nEmbd, lw.expertsUpW,   &moeExpertsUpType[i]);
            loadExpertsRaw(pfx + "ffn_down_exps.weight", cfg.nEmbd, IMe, lw.expertsDownW, &moeExpertsDownType[i]);
        }

        // ── SSM (Mamba) weights for hybrid archs ────────────────────────────
        // Loaded as raw fp16/Q8 buffers. Forward dispatch (selective scan + conv1d +
        // dt/alpha/beta projections) is a separate multi-week sub-project.
        // For hybrid archs: only SSM-only layers actually have these tensors —
        // attention layers (every Nth per full_attention_interval) skip.
        if (cfg.ssmInnerSize > 0 && !cfg.isAttentionLayer(i)) {
            auto loadRaw = [&](const std::string& name, GPUBuffer& buf) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                uint32_t nel = 1;
                for (auto d : ti.shape) nel *= (uint32_t)d;
                // Dequant to fp32 (small tensors; SSM weights aren't huge)
                std::vector<float> fp32((size_t)nel);
                dequant_tensor(fileData + gguf.data_offset + ti.offset,
                               fp32.data(), 1, nel, (GGUFType)ti.type);
                size_t bytes = (size_t)nel * 4;
                buf = gpu->createBuffer(name, bytes);
                gpu->writeBuffer(buf, fp32.data(), bytes);
                return true;
            };
            auto loadProjQ8 = [&](const std::string& name, uint32_t N, uint32_t K,
                                  GPUBuffer& wBuf, GPUBuffer& sBuf) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                auto rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                      N, K, (GGUFType)ti.type);
                uploadQ8Weight(*gpu, name, rep, wBuf, sBuf);
                return true;
            };
            auto loadProjKQ = [&](const std::string& name, uint32_t N, uint32_t K,
                                  GPUBuffer& buf, GGUFType& type,
                                  uint32_t& nBlocks, uint32_t& rowStride) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                type = (GGUFType)ti.type;
                if (type != GGUF_TYPE_Q4_K && type != GGUF_TYPE_Q5_K &&
                    type != GGUF_TYPE_Q6_K) return false;
                auto packed = packKQ(fileData + gguf.data_offset + ti.offset, N, K, type);
                nBlocks = packed.nBlocks;
                rowStride = packed.rowStrideWords;
                uploadKQWeight(name + ".kq", packed, buf);
                return true;
            };
            auto repackProjQ8 = [&](const std::string& name, uint32_t N, uint32_t K,
                                    Q8Repacked& rep) -> bool {
                auto it = gguf.tensor_index.find(name);
                if (it == gguf.tensor_index.end()) return false;
                auto& ti = gguf.tensors[it->second];
                rep = repackToQ8(fileData + gguf.data_offset + ti.offset,
                                 N, K, (GGUFType)ti.type);
                return true;
            };
            int loaded = 0;
            if (loadRaw(pfx + "ssm_conv1d.weight", lw.ssmConv1dW)) loaded++;
            if (loadRaw(pfx + "ssm_dt.bias",       lw.ssmDtBias))  loaded++;
            if (loadRaw(pfx + "ssm_a",             lw.ssmA))       loaded++;
            if (loadRaw(pfx + "ssm_norm.weight",   lw.ssmNorm))    loaded++;
            Q8Repacked betaRep, alphaRep;
            bool haveBeta = repackProjQ8(pfx + "ssm_beta.weight", cfg.ssmTimeStepRank,
                                         cfg.nEmbd, betaRep);
            bool haveAlpha = repackProjQ8(pfx + "ssm_alpha.weight", cfg.ssmTimeStepRank,
                                          cfg.nEmbd, alphaRep);
            if (haveBeta && haveAlpha && betaRep.K == alphaRep.K) {
                Q8Repacked fused;
                fused.N = betaRep.N + alphaRep.N;
                fused.K = betaRep.K;
                fused.weights.reserve(betaRep.weights.size() + alphaRep.weights.size());
                fused.weights.insert(fused.weights.end(), betaRep.weights.begin(), betaRep.weights.end());
                fused.weights.insert(fused.weights.end(), alphaRep.weights.begin(), alphaRep.weights.end());
                fused.scales.reserve(betaRep.scales.size() + alphaRep.scales.size());
                fused.scales.insert(fused.scales.end(), betaRep.scales.begin(), betaRep.scales.end());
                fused.scales.insert(fused.scales.end(), alphaRep.scales.begin(), alphaRep.scales.end());
                uploadQ8Weight(*gpu, pfx + "ssm_beta_alpha.weight", fused,
                               lw.ssmBetaAlphaW, lw.ssmBetaAlphaS);
                loaded += 2;
            } else {
                if (haveBeta) {
                    uploadQ8Weight(*gpu, pfx + "ssm_beta.weight", betaRep,
                                   lw.ssmBetaW, lw.ssmBetaS);
                    loaded++;
                }
                if (haveAlpha) {
                    uploadQ8Weight(*gpu, pfx + "ssm_alpha.weight", alphaRep,
                                   lw.ssmAlphaW, lw.ssmAlphaS);
                    loaded++;
                }
            }
            if (loadProjKQ(pfx + "ssm_out.weight", cfg.nEmbd, cfg.ssmInnerSize,
                           lw.ssmOutKQ, lw.ssmOutKQType,
                           lw.ssmOutKQNBlocks, lw.ssmOutKQRowStride) ||
                loadProjQ8(pfx + "ssm_out.weight", cfg.nEmbd, cfg.ssmInnerSize,
                           lw.ssmOutW, lw.ssmOutS)) loaded++;
            if (loadProjKQ(pfx + "attn_gate.weight", cfg.ssmInnerSize, cfg.nEmbd,
                           lw.attnGateKQ, lw.attnGateKQType,
                           lw.attnGateKQNBlocks, lw.attnGateKQRowStride) ||
                loadProjQ8(pfx + "attn_gate.weight", cfg.ssmInnerSize, cfg.nEmbd,
                           lw.attnGateW, lw.attnGateS)) loaded++;
            if (i == 0) {
                fprintf(stderr, "  SSM/DeltaNet weights loaded for layer 0: %d/8 tensors\n", loaded);
            }
        }

        if (i % 7 == 6 || i == cfg.nLayer - 1)
            fprintf(stderr, "  loaded layer %u/%u\n", i + 1, cfg.nLayer);
    }

    // Final norm
    loadNorm("output_norm.weight", finalNormW);

    // Embedding (dequantize to fp32 for CPU lookup)
    {
        auto it = gguf.tensor_index.find("token_embd.weight");
        if (it != gguf.tensor_index.end()) {
            auto& ti = gguf.tensors[it->second];
            uint32_t nel = 1;
            for (auto d : ti.shape) nel *= (uint32_t)d;
            const uint8_t* data = fileData + gguf.data_offset + ti.offset;
            const bool needCpuEmbedding = cfg.arch != "qwen35" ||
                std::getenv("BP_Q35_SYNC") || std::getenv("BP_SYNC_PREFILL") ||
                std::getenv("BP_DUMP_BUFFER_STATS");
            if (needCpuEmbedding) embeddingCPU.resize(nel);
            if (!needCpuEmbedding) {
                fprintf(stderr, "  Embedding: %u × %u (GPU-only)\n",
                        cfg.nVocab, cfg.nEmbd);
            } else if (ti.type == GGUF_TYPE_F16) {
                const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(data);
                for (uint32_t j = 0; j < nel; j++)
                    embeddingCPU[j] = fp16_to_f32(fp16[j]);
            } else if (ti.type == GGUF_TYPE_Q8_0) {
                uint32_t rows = (uint32_t)ti.shape[1];
                uint32_t cols = (uint32_t)ti.shape[0];
                uint32_t nBlocks = cols / 32;
                const Q8_0Block* blocks = reinterpret_cast<const Q8_0Block*>(data);
                for (uint32_t r = 0; r < rows; r++) {
                    for (uint32_t b = 0; b < nBlocks; b++) {
                        const auto& blk = blocks[r * nBlocks + b];
                        float scale = fp16_to_f32(blk.d);
                        for (int q = 0; q < 32; q++)
                            embeddingCPU[r * cols + b * 32 + q] = (float)blk.qs[q] * scale;
                    }
                }
            } else if (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K ||
                       ti.type == GGUF_TYPE_Q6_K) {
                uint32_t rows = (uint32_t)ti.shape[1];
                uint32_t cols = (uint32_t)ti.shape[0];
                dequant_kquant(data, embeddingCPU.data(), rows, cols, (GGUFType)ti.type);
            } else if (ti.type == GGUF_TYPE_F32) {
                memcpy(embeddingCPU.data(), data, nel * 4);
            } else {
                uint32_t rows = (uint32_t)ti.shape[1];
                uint32_t cols = (uint32_t)ti.shape[0];
                dequant_tensor(data, embeddingCPU.data(), rows, cols, (GGUFType)ti.type);
            }
            if (needCpuEmbedding) fprintf(stderr, "  Embedding: %u × %u (%s)\n", cfg.nVocab, cfg.nEmbd,
                   ti.type == GGUF_TYPE_Q8_0 ? "Q8_0→f32" :
                   ti.type == GGUF_TYPE_Q4_K ? "Q4_K→f32" :
                   ti.type == GGUF_TYPE_Q5_K ? "Q5_K→f32" :
                   ti.type == GGUF_TYPE_Q6_K ? "Q6_K→f32" :
                   ti.type == GGUF_TYPE_Q4_0 ? "Q4_0→f32" :
                   ti.type == GGUF_TYPE_Q4_1 ? "Q4_1→f32" :
                   ti.type == GGUF_TYPE_Q5_0 ? "Q5_0→f32" :
                   ti.type == GGUF_TYPE_Q5_1 ? "Q5_1→f32" :
                   ti.type == GGUF_TYPE_F16 ? "f16→f32" :
                   ti.type == GGUF_TYPE_BF16 ? "bf16→f32" : "f32");

            // LM head: use quantized format if embedding is quantized
            if (cfg.tieWordEmbeddings) {
                if (cfg.arch == "qwen35") {
                    lmHeadKQType = (GGUFType)ti.type;
                    auto kq = packKQ(data, cfg.nVocab, cfg.nEmbd, lmHeadKQType);
                    kqLmNBlocks = kq.nBlocks;
                    kqLmRowStride = kq.rowStrideWords;
                    uploadKQWeight("lm_head_kq", kq, lmHeadKQ);
                    lmHeadIsKQ = true;
                    fprintf(stderr, "  LM head: tied embeddings (native Q6_K, %llu MB)\n",
                           (unsigned long long)(kq.data.size()*4 / 1048576));
                } else if (isKQuant && (ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K ||
                                 ti.type == GGUF_TYPE_Q6_K)) {
                    // Upload as K-quant on GPU
                    auto kq = packKQ(data, cfg.nVocab, cfg.nEmbd, (GGUFType)ti.type);
                    kqLmNBlocks = kq.nBlocks;
                    kqLmRowStride = kq.rowStrideWords;
                    uploadKQWeight("lm_head_kq", kq, lmHeadKQ);
                    lmHeadIsKQ = true;
                    fprintf(stderr, "  LM head: tied embeddings (%s, %llu MB)\n", kqName,
                           (unsigned long long)(kq.data.size() * 4 / 1048576));
                } else if (ti.type == GGUF_TYPE_Q8_0) {
                    // Keep as Q8 on GPU — no dequant needed
                    auto rep = repack_q8_0(data, cfg.nVocab, cfg.nEmbd);
                    uploadQ8Weight(*gpu, "lm_head_q8", rep,
                                   lmHeadQ8W, lmHeadQ8S);
                    lmHeadIsQ8 = true;
                    uint64_t wBytes = (uint64_t)rep.weights.size() * 4;
                    uint64_t sBytes = (uint64_t)rep.scales.size() * 4;
                    fprintf(stderr, "  LM head: tied embeddings (Q8, %llu MB)\n",
                           (unsigned long long)((wBytes + sBytes) / 1048576));
                } else if (ti.type == GGUF_TYPE_Q4_0 || ti.type == GGUF_TYPE_Q4_1 ||
                           ti.type == GGUF_TYPE_Q5_0 || ti.type == GGUF_TYPE_Q5_1) {
                    if (weightsAreNativeQ4 && ti.type == GGUF_TYPE_Q4_0) {
                        auto rep = repackQ4_0Native(data, cfg.nVocab, cfg.nEmbd);
                        uploadQ8Weight(*gpu, "lm_head_q4", rep, lmHeadQ8W, lmHeadQ8S);
                        lmHeadIsQ4 = true;
                        fprintf(stderr, "  LM head: tied embeddings (native Q4_0, %llu MB)\n",
                            (unsigned long long)((rep.weights.size() * 4 + rep.scales.size() * 4) / 1048576));
                    } else {
                    // Legacy quants (e.g. Gemma 4 Q4_0): repack to Q8 so the LM
                    // head fits the standard Q8 decode dispatch path. The fp16
                    // fallback below would allocate a ~768 MB f16 buffer for a
                    // big-vocab model, which on D3D12+Dawn destabilizes the
                    // queue (small subsequent writeBuffer calls silently no-op).
                    auto rep = repackToQ8(data, cfg.nVocab, cfg.nEmbd, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, "lm_head_q8", rep, lmHeadQ8W, lmHeadQ8S);
                    lmHeadIsQ8 = true;
                    uint64_t wBytes = (uint64_t)rep.weights.size() * 4;
                    uint64_t sBytes = (uint64_t)rep.scales.size() * 4;
                    fprintf(stderr, "  LM head: tied embeddings (Q8 from %s, %llu MB)\n",
                           ti.type == GGUF_TYPE_Q4_0 ? "Q4_0" :
                           ti.type == GGUF_TYPE_Q4_1 ? "Q4_1" :
                           ti.type == GGUF_TYPE_Q5_0 ? "Q5_0" : "Q5_1",
                           (unsigned long long)((wBytes + sBytes) / 1048576));
                    }
                } else {
                    // fp16 fallback
                    std::vector<uint16_t> fp16(nel);
                    for (uint32_t j = 0; j < nel; j++)
                        fp16[j] = f32_to_fp16(embeddingCPU[j]);
                    uint64_t totalBytes = (uint64_t)nel * 2;
                    lmHeadW = gpu->createBuffer("lm_head_fp16", totalBytes);
                    const uint64_t CHUNK = 128 * 1024 * 1024;
                    for (uint64_t off = 0; off < totalBytes; off += CHUNK) {
                        uint64_t sz = std::min(CHUNK, totalBytes - off);
                        wgpuQueueWriteBuffer(gpu->queue, lmHeadW.handle, off,
                                             (const uint8_t*)fp16.data() + off, sz);
                    }
                    fprintf(stderr, "  LM head: tied embeddings (fp16, %llu MB)\n",
                           (unsigned long long)(totalBytes / 1048576));
                }
            }
        }
    }

    // ─── PLE global weight loading ──────────────────────────────────────
    if (cfg.pleSize > 0) {
        const bool requestGpuPle = weightsAreNativeQ4 &&
                                   std::getenv("BP_CPU_PLE") == nullptr;
        // per_layer_token_embd.weight — massive per-layer embedding table
        // Shape: [pleSize * nLayer, nVocab] — dequant to fp32 for CPU lookup
        auto it = gguf.tensor_index.find("per_layer_token_embd.weight");
        if (it != gguf.tensor_index.end()) {
            auto& ti = gguf.tensors[it->second];
            uint32_t rows = 1, cols = 1;
            if (ti.shape.size() >= 2) { cols = (uint32_t)ti.shape[0]; rows = (uint32_t)ti.shape[1]; }
            else if (ti.shape.size() == 1) { cols = (uint32_t)ti.shape[0]; }
            if (requestGpuPle && ti.type == GGUF_TYPE_Q4_0 && cols % 32 == 0) {
                auto rep = repackQ4_0Native(fileData + gguf.data_offset + ti.offset,
                                            rows, cols);
                uploadQ8Weight(*gpu, "ple_token_emb_q4", rep,
                               pleTokenEmbW, pleTokenEmbS);
                fprintf(stderr, "  PLE embedding: %u × %u (native Q4 GPU, %zu MB)\n",
                        rows, cols, (rep.weights.size() + rep.scales.size()) * 4 / 1048576);
            } else {
                pleEmbCPU.resize((size_t)rows * cols);
                dequant_tensor(fileData + gguf.data_offset + ti.offset,
                               pleEmbCPU.data(), rows, cols, (GGUFType)ti.type);
                fprintf(stderr, "  PLE embedding: %u × %u (%zu MB fp32)\n",
                        rows, cols, (size_t)rows * cols * 4 / 1048576);
            }
        }

        // per_layer_model_proj.weight — Q8 repack for future GPU use plus an
        // fp32 CPU reference used for numerically stable PLE preprocessing.
        {
            auto it2 = gguf.tensor_index.find("per_layer_model_proj.weight");
            if (it2 != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it2->second];
                uint32_t N = (ti.shape.size() >= 2) ? (uint32_t)ti.shape[1] : 0;
                uint32_t K = (ti.shape.size() >= 2) ? (uint32_t)ti.shape[0] : 0;
                if (N > 0 && K > 0) {
                    auto rep = requestGpuPle && ti.type == GGUF_TYPE_Q4_0
                        ? repackQ4_0Native(fileData + gguf.data_offset + ti.offset, N, K)
                        : repackToQ8(fileData + gguf.data_offset + ti.offset,
                                     N, K, (GGUFType)ti.type);
                    uploadQ8Weight(*gpu, "ple_model_proj", rep, pleModelProjW, pleModelProjS);
                    if (!requestGpuPle) {
                        pleModelProjCPU.resize((size_t)N * K);
                        dequant_tensor(fileData + gguf.data_offset + ti.offset,
                                       pleModelProjCPU.data(), N, K, (GGUFType)ti.type);
                    }
                }
            }
        }

        // per_layer_proj_norm.weight — GPU + fp32 CPU copy
        loadNorm("per_layer_proj_norm.weight", pleProjNormW);
        {
            auto it3 = gguf.tensor_index.find("per_layer_proj_norm.weight");
            if (it3 != gguf.tensor_index.end()) {
                auto& ti = gguf.tensors[it3->second];
                uint32_t nel = 1;
                for (auto d : ti.shape) nel *= (uint32_t)d;
                pleProjNormCPU.resize(nel);
                const uint8_t* data = fileData + gguf.data_offset + ti.offset;
                if (ti.type == GGUF_TYPE_F32) memcpy(pleProjNormCPU.data(), data, nel * 4);
                else dequant_tensor(data, pleProjNormCPU.data(), 1, nel, (GGUFType)ti.type);
            }
        }
        pleGpuPreprocess = requestGpuPle && pleTokenEmbW.handle &&
                           pleModelProjW.handle && pleProjNormW.handle;
        if (requestGpuPle && !pleGpuPreprocess)
            fprintf(stderr, "WARNING: GPU PLE requested but required tensors were unavailable; using fallback\n");
    }

    // ─── MTP weight loading ──────────────────────────────────────────────
    if (cfg.hasMtp) {
        auto hasT = [&](const std::string& name) {
            return gguf.tensor_index.count(name) > 0;
        };

        if (hasT("blk.0.nextn.eh_proj.weight")) {
            mtpCfg.type = MTPType::Gemma4;
            uint32_t nMtp = 0;
            while (hasT("blk." + std::to_string(nMtp) + ".nextn.eh_proj.weight"))
                nMtp++;
            mtpCfg.numLayers = nMtp;
            mtpCfg.numDraftTokens = 1;
            fprintf(stderr, "  MTP: Gemma4 style, %u layers\n", nMtp);
        } else if (hasT("blk.0.nextn.enorm.weight")) {
            mtpCfg.type = MTPType::Qwen36;
            uint32_t nMtp = 0;
            while (hasT("blk." + std::to_string(nMtp) + ".nextn.enorm.weight"))
                nMtp++;
            mtpCfg.numLayers = nMtp;
            mtpCfg.numDraftTokens = 1;
            fprintf(stderr, "  MTP: Qwen3.6 style, %u layers\n", nMtp);
        } else {
            fprintf(stderr, "  MTP: detected but tensor naming not recognized\n");
            cfg.hasMtp = false;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "  Weights loaded in %lldms\n", (long long)ms);
}

// ─── RoPE tables ─────────────────────────────────────────────────────────────

uint32_t ModelRunner::layerRotaryDim(uint32_t layerIdx) const {
    uint32_t headDim = cfg.headDim;
    if (layerIdx < cfg.perLayer.size() && cfg.perLayer[layerIdx].headDim > 0)
        headDim = cfg.perLayer[layerIdx].headDim;

    bool isSwa = layerIdx < cfg.layerAttnTypes.size() &&
                 cfg.layerAttnTypes[layerIdx] == AttnLayerType::SlidingWindow;
    uint32_t dim = headDim;
    if (isSwa && swaRotaryDim > 0)
        dim = swaRotaryDim;
    else if (!isSwa && ropeFreqRotaryDim > 0)
        dim = ropeFreqRotaryDim;
    else if (rotaryDim > 0)
        dim = rotaryDim;

    return std::min(dim, headDim) & ~1u;
}

void ModelRunner::computeRopeTables() {
    // For ONNX models with pre-computed RoPE tables, upload directly
    if (hasPrecomputedRope) return;  // already uploaded in loadOnnx()

    auto buildTables = [&](float theta, uint32_t ropeDim,
                           GPUBuffer& cosBuf, GPUBuffer& sinBuf,
                           const char* cosName, const char* sinName) {
        uint32_t ropeHalf = ropeDim / 2;
        std::vector<float> cosTable(maxSeqLen * ropeHalf), sinTable(maxSeqLen * ropeHalf);
        for (uint32_t pos = 0; pos < maxSeqLen; pos++) {
            for (uint32_t i = 0; i < ropeHalf; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / ropeDim);
                float angle = pos * freq;
                cosTable[pos * ropeHalf + i] = cosf(angle);
                sinTable[pos * ropeHalf + i] = sinf(angle);
            }
        }
        cosBuf = gpu->createBuffer(cosName, maxSeqLen * ropeHalf * 4);
        sinBuf = gpu->createBuffer(sinName, maxSeqLen * ropeHalf * 4);
        gpu->writeBuffer(cosBuf, cosTable.data(), maxSeqLen * ropeHalf * 4);
        gpu->writeBuffer(sinBuf, sinTable.data(), maxSeqLen * ropeHalf * 4);
    };

    uint32_t globalRopeDim = ropeFreqRotaryDim > 0
        ? ropeFreqRotaryDim : (rotaryDim > 0 ? rotaryDim : cfg.headDim);
    buildTables(cfg.ropeTheta, globalRopeDim,
                ropeCosBuf, ropeSinBuf, "rope_cos", "rope_sin");

    // Gemma 3/4: SWA layers use a separate, smaller theta (default 10000).
    // The GGUF only stores one rope.freq_base; the SWA base is conventionally
    // 10000 unless rope.freq_base_swa is present.
    bool needsSwa = (cfg.arch == "gemma3" || cfg.arch == "gemma4") &&
                    !cfg.layerAttnTypes.empty();
    if (needsSwa) {
        // No GGUF accessor here; just use the convention used by llama.cpp.
        const float swaTheta = 10000.0f;
        uint32_t swaRopeDim = swaRotaryDim;
        for (uint32_t i = 0; i < cfg.nLayer && i < cfg.perLayer.size(); i++) {
            if (i < cfg.layerAttnTypes.size() &&
                cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow) {
                swaRopeDim = layerRotaryDim(i);
                break;
            }
        }
        if (swaRopeDim == 0) swaRopeDim = globalRopeDim;
        buildTables(swaTheta, swaRopeDim, ropeCosBufSWA, ropeSinBufSWA,
                    "rope_cos_swa", "rope_sin_swa");
        fprintf(stderr, "  RoPE SWA tables: dim=%u theta=%.0f (global dim=%u theta=%.0f)\n",
                swaRopeDim, swaTheta, globalRopeDim, cfg.ropeTheta);
    }
}

// ─── Build decode pipeline ───────────────────────────────────────────────────

// MoE FFN dispatch helper (Phase 3d stub).
// Returns false; appended dispatches happen in this function once wired.
bool ModelRunner::appendMoeFfnDispatches(uint32_t layerIdx, GPUBuffer xIn, GPUBuffer xOut) {
    (void)layerIdx; (void)xIn; (void)xOut;
    // Pre-flight: required tensors must be loaded.
    if (layerIdx >= layerWeights.size()) return false;
    auto& lw = layerWeights[layerIdx];
    if (!lw.routerW.handle || !lw.expertsGateW.handle ||
        !lw.expertsUpW.handle || !lw.expertsDownW.handle) {
        return false;
    }
    // TODO(Phase 3d): emit the per-token MoE FFN dispatch sequence:
    //   1. router matmul: xIn @ routerW.T → moeRouterOutBuf
    //   2. moe_gate (top-k softmax) → moeIndicesBuf / moeWeightsBuf
    //   3. for k in 0..numExpertsPerTok:
    //        idx = moeIndices[k], w = moeWeights[k]
    //        gate = xIn @ expertsGateW[idx]   (indirect IQ matmul w/ expert_row_off)
    //        up   = xIn @ expertsUpW[idx]     (indirect IQ matmul)
    //        act  = silu(gate) * up
    //        down = act @ expertsDownW[idx]   (indirect IQ matmul)
    //        xOut += w * down
    //   4. shared expert: xOut += silu(xIn@shexpGate)*shexpUp @ shexpDown
    return false;
}

void ModelRunner::buildDecodePipeline() {
    qDim = cfg.nHead * cfg.headDim;
    kvDim = cfg.nKvHeads * cfg.headDim;
    qkvOut = qDim + 2 * kvDim;
    decodeUsesFusedRopeParams = false;
    const auto& limits = effectiveLimits(*gpu);
    std::string decodeAdapter = gpu->adapterName;
    std::transform(decodeAdapter.begin(), decodeAdapter.end(), decodeAdapter.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    const bool isNvidiaAdapter = decodeAdapter.find("nvidia") != std::string::npos;
    const bool isIntelAdapter = decodeAdapter.find("intel") != std::string::npos;
    const bool isAmdAdapter = decodeAdapter.find("amd") != std::string::npos ||
                              decodeAdapter.find("radeon") != std::string::npos;
    const bool canUse512ThreadKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 512u &&
        limits.maxComputeWorkgroupStorageSize >= 32u * 1024u;
    const bool canUse256ThreadSubgroupKernels =
        gpu->supportsSubgroups && isNvidiaAdapter &&
        limits.maxComputeInvocationsPerWorkgroup >= 256u &&
        limits.maxComputeWorkgroupStorageSize >= 16u * 1024u;
    const bool decodeFastQ8Eligible =
        canUse256ThreadSubgroupKernels &&
        (cfg.nEmbd % 512u == 0u) &&
        (qDim % 512u == 0u);
    const bool decodeWideFp16Eligible = canUse256ThreadSubgroupKernels;
    const bool qwen35VulkanDecode =
        (cfg.arch == "qwen35" && gpu->backendType != WGPUBackendType_D3D12 &&
         !std::getenv("BP_Q35_GENERIC_KERNELS"));
    const bool qwen35SubgroupSuite = cfg.arch == "qwen35" &&
        gpu->supportsSubgroups && isNvidiaAdapter &&
        !std::getenv("BP_Q35_GENERIC_KERNELS");
    uint32_t Q8_TILE = 8;
    // Gemma 4 has only a few very wide query heads. Smaller attention chunks
    // expose enough independent workgroups to occupy the GPU. Cross-device
    // sweeps select 8 on NVIDIA/AMD and 16 on Intel Arc; Intel loses occupancy
    // again at 8 due to the additional partial-reduction work.
    if (cfg.arch == "gemma4") {
        gqaChunkSize = gpu->adapterName.find("Intel") != std::string::npos
            ? 16u : 8u;
    } else if (cfg.arch == "qwen35" && isAmdAdapter) {
        // Qwen 3.5 has only six full-attention layers. The default 64-token
        // chunks leave too few pass-1 workgroups to occupy AMD GPUs; a sweep
        // over 8/16/32/64 selects 8 for both the 2B and 4B models.
        gqaChunkSize = 8u;
    }
    if (const char* chunkEnv = std::getenv("BP_GQA_CHUNK_SIZE")) {
        const int requested = std::atoi(chunkEnv);
        if (requested == 8 || requested == 16 || requested == 32 || requested == 64)
            gqaChunkSize = static_cast<uint32_t>(requested);
    }
    uint32_t maxChunks = (maxSeqLen + gqaChunkSize - 1) / gqaChunkSize;

    // Load kernels from embedded shaders
    auto& plRmsNorm    = (qwen35VulkanDecode || qwen35SubgroupSuite) ? getKernel("rms_norm_qwen_vulkan")
                                            : getKernel("rms_norm");
    auto& plAddRmsNorm = (qwen35VulkanDecode || qwen35SubgroupSuite) ? getKernel("add_rms_norm_qwen_vulkan")
                                            : getKernel("add_rms_norm");
    const CompiledPipeline* plQ8IntelUnpack = nullptr;
    if (isIntelAdapter) {
        std::string source(WGSL_Q8_MATMUL);
        const std::string wv0 =
            "let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),\n"
            "                                f32(extractBits(i32(pw0), 8u, 8u)),\n"
            "                                f32(extractBits(i32(pw0), 16u, 8u)),\n"
            "                                f32(extractBits(i32(pw0), 24u, 8u)));";
        const std::string wv1 =
            "let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),\n"
            "                                f32(extractBits(i32(pw1), 8u, 8u)),\n"
            "                                f32(extractBits(i32(pw1), 16u, 8u)),\n"
            "                                f32(extractBits(i32(pw1), 24u, 8u)));";
        if (auto pos = source.find(wv0); pos != std::string::npos)
            source.replace(pos, wv0.size(), "let wv0 = vec4<f32>(unpack4xI8(pw0));");
        if (auto pos = source.find(wv1); pos != std::string::npos)
            source.replace(pos, wv1.size(), "let wv1 = vec4<f32>(unpack4xI8(pw1));");
        plQ8IntelUnpack = &gpu->getOrCreatePipeline(
            "q8_matmul_intel_unpack4xi8", source, 6);
    }
    auto& plQ8Matmul = plQ8IntelUnpack ? *plQ8IntelUnpack
        : (cfg.arch == "qwen35" && gpu->backendType != WGPUBackendType_D3D12 &&
           !std::getenv("BP_Q35_GENERIC_KERNELS"))
            ? getKernel("q8_matmul_vec4")
            : getKernel("q8_matmul");
    auto& plQ8MatmulNorm = getKernel("q8_matmul_norm");

    // K-quant kernel selection
    bool useKQ = weightsUseNativeKQ;
    const char* kqKernelName = (weightQuantType == GGUF_TYPE_Q4_K) ? "q4k_matmul" :
                               (weightQuantType == GGUF_TYPE_Q5_K) ? "q5k_matmul" :
                               (weightQuantType == GGUF_TYPE_Q6_K) ? "q6k_matmul" : nullptr;
    // Q4K uses 32-thread WG (1 col/WG), Q5K/Q6K use 256-thread WG (8 cols/WG)
    uint32_t kqTileN = (weightQuantType == GGUF_TYPE_Q4_K) ? 1u : 8u;
    const CompiledPipeline* plKQ = nullptr;
    if (useKQ && kqKernelName) {
        plKQ = &getKernel(kqKernelName);
    }
    // The 256-thread Q4_K kernel processes eight output rows per workgroup.
    // It is consistently faster on NVIDIA decode (the 128-thread/four-row
    // variant remains the portable default for AMD and Intel).  Keep explicit
    // overrides so cross-device experiments can compare either path.
    const bool useQ4KDecode256 = std::getenv("BP_Q4K_DECODE_256") ||
        ((isNvidiaAdapter || isAmdAdapter) && !std::getenv("BP_Q4K_DECODE_128"));
    // Packed 4x8 integer dot products are conformant and faster on NVIDIA and
    // Intel. AMD exposes the operation but changes cared-model logits, so it
    // retains the floating-point Q4_K reduction.
    const bool useQ4KDp4a = (isNvidiaAdapter || isIntelAdapter || isAmdAdapter) &&
        !std::getenv("BP_Q4K_DISABLE_DP4A");
    auto kqPipelineFor = [&](GGUFType type) -> const CompiledPipeline* {
        switch (type) {
            case GGUF_TYPE_Q4_K: return &getKernel(useQ4KDp4a
                ? "q4k_matmul_dp4a"
                : useQ4KDecode256 ? "q4k_matmul" : "q4k_matmul_128");
            case GGUF_TYPE_Q5_K: return &getKernel("q5k_matmul");
            case GGUF_TYPE_Q6_K: return &getKernel("q6k_matmul");
            default: return nullptr;
        }
    };
    auto kqTileFor = [&](GGUFType type) {
        return type == GGUF_TYPE_Q4_K && !useQ4KDecode256 ? 4u : 8u;
    };

    const bool subgroupMatrixKernelReady =
        gpu->supportsSubgroupMatrix &&
        canUse512ThreadKernels &&
        canCompileEmbeddedKernel(*gpu, "q8_matmul_vulkan");
    fprintf(stderr, "  Subgroup matrix: %s\n",
           subgroupMatrixKernelReady ? "available (i8×i8→i32 MMA)" : "not available");
    auto& plQ8Fast     = getKernel("q8_matmul_fast");
    auto& plFusedRope  = getKernelHD(cfg.hasQkNorm ? "fused_qknorm_rope" : "fused_rope");
    // Intel's Windows subgroup path measured slightly slower than the
    // shared-memory reduction; NVIDIA and AMD both benefit from replacing
    // five workgroup barriers with the portable two-barrier subgroup merge.
    const bool useSubgroupChunkedAttention = gpu->supportsSubgroups &&
        gpu->adapterName.find("Intel") == std::string::npos;
    const char* chunkP1Kernel = useSubgroupChunkedAttention
        ? "gqa_chunked_pass1_subgroup" : "gqa_chunked_pass1";
    auto& plChunkP1    = getKernelHD(chunkP1Kernel);
    auto& plChunkP2    = getKernelHD("gqa_chunked_pass2");
    const CompiledPipeline* plQ4Decode = weightsAreNativeQ4
        ? &getKernel("matmul_q4_decode") : nullptr;
    static const char* Q4_PREQUANT_DOWN_WGSL = R"(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>B:array<u32>;
@group(0) @binding(3)var<storage,read>S:array<u32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id)wid:vec3<u32>,
        @builtin(local_invocation_id)lid:vec3<u32>){
 let N=P[1];let K=P[2];let warp=lid.x/32u;let lane=lid.x&31u;
 let row=wid.x*8u+warp;let nblocks=K/32u;let words=K/8u;
 var acc=0.0;
 if(row<N){
  for(var g=0u;g<K/256u;g++){
   let xb=lane/4u;let xq0=XQ[g*64u+lane*2u];let xq1=XQ[g*64u+lane*2u+1u];
   let qw=B[row*words+g*32u+lane];
   let b0=qw&255u;let b1=(qw>>8u)&255u;let b2=(qw>>16u)&255u;let b3=qw>>24u;
   let w0=(u32(b0&15u)-8u)&255u;let w1=(u32(b0>>4u)-8u)&255u;
   let w2=(u32(b1&15u)-8u)&255u;let w3=(u32(b1>>4u)-8u)&255u;
   let w4=(u32(b2&15u)-8u)&255u;let w5=(u32(b2>>4u)-8u)&255u;
   let w6=(u32(b3&15u)-8u)&255u;let w7=(u32(b3>>4u)-8u)&255u;
   let wq0=w0|(w1<<8u)|(w2<<16u)|(w3<<24u);
   let wq1=w4|(w5<<8u)|(w6<<16u)|(w7<<24u);
   let wb=g*8u+xb;let si=row*nblocks+wb;let sp=unpack2x16float(S[si/2u]);
   let ws=select(sp.x,sp.y,(si&1u)!=0u);
   acc+=f32(dot4I8Packed(xq0,wq0)+dot4I8Packed(xq1,wq1))*XS[wb]*ws;
  }
 }
 let sum=subgroupAdd(acc);if(lane==0u&&row<N){Y[row]=sum;}
}
)";
    const bool useNativeQ4PrequantDown = weightsAreNativeQ4 && isNvidiaAdapter &&
        !std::getenv("BP_Q4_DISABLE_PREQUANT_DOWN");
    const CompiledPipeline* plQ4PrequantDown = useNativeQ4PrequantDown
        ? &gpu->getOrCreatePipeline("q4_prequant_down", std::string(Q4_PREQUANT_DOWN_WGSL), 6)
        : nullptr;
    uint32_t Q4_DECODE_TILE_N = 32;
    int q4Cols = std::getenv("BP_Q4_COLS") ? std::atoi(std::getenv("BP_Q4_COLS"))
                                             : isIntelAdapter ? 2 : 1;
    if (weightsAreNativeQ4 && (q4Cols >= 1 && q4Cols <= 3)) {
            std::string src = getEmbeddedKernels().at("matmul_q4_decode").source;
            auto replaceAll = [&](const std::string& from, const std::string& to) {
                size_t pos = 0;
                while ((pos = src.find(from, pos)) != std::string::npos) {
                    src.replace(pos, from.size(), to);
                    pos += to.size();
                }
            };
            std::string cols = std::to_string(q4Cols);
            std::string tile = std::to_string(q4Cols * 8);
            replaceAll("COLS_PER_WARP: u32 = 4u", "COLS_PER_WARP: u32 = " + cols + "u");
            replaceAll("array<u32, 4>", "array<u32, " + cols + ">");
            replaceAll("array<bool, 4>", "array<bool, " + cols + ">");
            replaceAll("array<f32, 4>", "array<f32, " + cols + ">");
            replaceAll("wid.x * 32u", "wid.x * " + tile + "u");
            Q4_DECODE_TILE_N = q4Cols * 8;
            plQ4Decode = &gpu->getOrCreatePipeline(
                "matmul_q4_decode_c" + cols, src, 5);
            fprintf(stderr, "  Native Q4 decode: %d columns/warp (%u outputs/WG)\n",
                    q4Cols, Q4_DECODE_TILE_N);
    }
    const CompiledPipeline* plQ4Gateup = plQ4Decode;
    uint32_t Q4_GATEUP_TILE_N = Q4_DECODE_TILE_N;
    if (weightsAreNativeQ4) {
        int gateCols = std::getenv("BP_Q4_GATEUP_COLS")
            ? std::atoi(std::getenv("BP_Q4_GATEUP_COLS"))
            : isIntelAdapter ? 3 : 2;
        gateCols = std::max(1, std::min(3, gateCols));
        std::string src = getEmbeddedKernels().at("matmul_q4_decode").source;
        auto repl = [&](const std::string& from, const std::string& to) {
            size_t pos = 0;
            while ((pos = src.find(from, pos)) != std::string::npos) {
                src.replace(pos, from.size(), to); pos += to.size();
            }
        };
        const std::string cols = std::to_string(gateCols);
        const std::string tile = std::to_string(gateCols * 8);
        repl("COLS_PER_WARP: u32 = 4u", "COLS_PER_WARP: u32 = " + cols + "u");
        repl("array<u32, 4>", "array<u32, " + cols + ">");
        repl("array<bool, 4>", "array<bool, " + cols + ">");
        repl("array<f32, 4>", "array<f32, " + cols + ">");
        repl("wid.x * 32u", "wid.x * " + tile + "u");
        plQ4Gateup = &gpu->getOrCreatePipeline(
            "matmul_q4_decode_gateup_c" + cols, src, 5);
        Q4_GATEUP_TILE_N = gateCols * 8;
    }
    // The vocabulary projection is vastly wider than per-layer projections;
    // use the embedded four-column tile to amortize activation quantization.
    const CompiledPipeline* plQ4LmHead = weightsAreNativeQ4
        ? &getKernel("matmul_q4_decode") : nullptr;
    constexpr uint32_t Q4_LM_TILE_N = 32;
    static const char* Q4_A32_WGSL = R"(
enable subgroups;
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let N = P[1]; let K = P[2];
    let warp = lid.x / 32u; let lane = lid.x % 32u;
    let row = wid.x * 4u + warp;
    var acc = 0.0;
    if (row < N) {
        for (var k = lane; k < K; k += 32u) {
            let word = B[row * (K / 8u) + k / 8u];
            let q = i32((word >> ((k & 7u) * 4u)) & 15u) - 8;
            let block = row * (K / 32u) + k / 32u;
            let sp = unpack2x16float(S[block / 2u]);
            let scale = select(sp.x, sp.y, (block & 1u) != 0u);
            acc += X[k] * f32(q) * scale;
        }
    }
    let total = subgroupAdd(acc);
    if (lane == 0u && row < N) { Y[row] = total; }
}
)";
    const CompiledPipeline* plQ4A32 = weightsAreNativeQ4
        ? &gpu->getOrCreatePipeline("q4_matmul_a32", std::string(Q4_A32_WGSL), 5)
        : nullptr;
    static const char* Q4_A32_WIDE_WGSL = R"(
enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>B:array<u32>;
@group(0) @binding(2)var<storage,read>S:array<u32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>){
 let N=P[1];let K=P[2];let warp=lid.x/32u;let lane=lid.x&31u;let r0=wid.x*8u+warp*2u;var acc:array<f32,2>;
 for(var c=0u;c<2u;c++){let row=r0+c;if(row<N){for(var k=lane;k<K;k+=32u){let w=B[row*(K/8u)+k/8u];let q=i32((w>>((k&7u)*4u))&15u)-8;let b=row*(K/32u)+k/32u;let sp=unpack2x16float(S[b/2u]);let sc=select(sp.x,sp.y,(b&1u)==1u);acc[c]+=X[k]*f32(q)*sc;}}}
 for(var c=0u;c<2u;c++){let row=r0+c;let v=subgroupAdd(acc[c]);if(lane==0u&&row<N){Y[row]=v;}}
}
)";
    const bool useNarrowPleQ4 = std::getenv("BP_PLE_Q4_NARROW") ||
        (isNvidiaAdapter && !std::getenv("BP_PLE_Q4_WIDE"));
    const CompiledPipeline* plQ4A32Ple = weightsAreNativeQ4 &&
        !useNarrowPleQ4
        ? &gpu->getOrCreatePipeline("q4_matmul_a32_ple_wide", std::string(Q4_A32_WIDE_WGSL), 5)
        : plQ4A32;
    const uint32_t Q4_A32_PLE_TILE = useNarrowPleQ4 ? 4u : 8u;
    static const char* PLE_TOKEN_Q4_GATHER_WGSL = R"(
@group(0) @binding(0) var<storage, read> B: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<u32>;
@group(0) @binding(2) var<storage, read> Token: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i=gid.x;let K=P[0];if(i>=K){return;}let row=u32(max(Token[0],0));
 let word=B[row*(K/8u)+i/8u];let q=i32((word>>((i&7u)*4u))&15u)-8;
 let block=row*(K/32u)+i/32u;let sp=unpack2x16float(S[block/2u]);
 let scale=select(sp.x,sp.y,(block&1u)!=0u);Y[i]=f32(q)*scale*bitcast<f32>(P[1]);
}
)";
    static const char* PLE_TOKEN_Q4_ASYM_GATHER_WGSL = R"(
@group(0) @binding(0) var<storage, read> B: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<u32>;
@group(0) @binding(2) var<storage, read> Z: array<u32>;
@group(0) @binding(3) var<storage, read> Token: array<i32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let K = P[0];
    if (i >= K) { return; }
    let row = u32(max(Token[0], 0)); let V=P[2]; let D=P[3];
    let layer=i/D; let col=i%D; let packedRow=layer*V+row;
    let byteIndex=packedRow*(D/2u)+col/2u;
    let word=B[byteIndex/4u];
    let qv=i32((word>>(((byteIndex&3u)*8u+(col&1u)*4u)))&15u);
    let block = packedRow * (D / 32u) + col / 32u;
    let zpByteIndex=block/2u; let zpWord=Z[zpByteIndex/4u];
    let zpByte=(zpWord>>((zpByteIndex&3u)*8u))&255u;
    let zp=i32(select(zpByte&15u,zpByte>>4u,(block&1u)==1u));
    let sp = unpack2x16float(S[block / 2u]);
    let scale = select(sp.x, sp.y, (block & 1u) != 0u);
    Y[i] = f32(qv-zp) * scale * bitcast<f32>(P[1]);
}
)";
    static const char* PLE_PROJECT_COMBINE_WGSL = R"(
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
    const CompiledPipeline* plPleTokenGather = pleGpuPreprocess
        ? &gpu->getOrCreatePipeline(
            pleTokenEmbAsymmetric ? "ple_token_q4_asym_gather" : "ple_token_q4_gather",
            pleTokenEmbAsymmetric ? std::string(PLE_TOKEN_Q4_ASYM_GATHER_WGSL)
                                  : std::string(PLE_TOKEN_Q4_GATHER_WGSL),
            pleTokenEmbAsymmetric ? 6 : 5)
        : nullptr;
    const CompiledPipeline* plPleProjectCombine = pleGpuPreprocess
        ? &gpu->getOrCreatePipeline("ple_project_combine", std::string(PLE_PROJECT_COMBINE_WGSL), 4)
        : nullptr;
    static const char* RMS_NORM_ADD_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> Dst: array<f32>;
@group(0) @binding(1) var<storage, read> Src: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let N = P[0]; let eps = bitcast<f32>(P[1]);
    var ss = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = Src[j]; ss += v * v; }
    sums[tid] = ss;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (tid < stride) { sums[tid] += sums[tid + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride /= 2u;
    }
    let rms = inverseSqrt(sums[0] / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) {
        Dst[j] += Src[j] * rms * W[j];
    }
}
)";
    const CompiledPipeline* plRmsNormAdd = weightsAreNativeQ4
        ? &gpu->getOrCreatePipeline("rms_norm_add_inplace", std::string(RMS_NORM_ADD_WGSL), 4)
        : nullptr;
    GPUBuffer pleTokenGatherParams, pleModelProjParams, pleCombineParams;
    auto& plFp16Gemm   = getKernel("fp16_gemm");
    auto& plFp16Wide   = getKernel("fp16_gemm_wide");
    auto& plArgmax     = getKernel("argmax");
    auto& plEmbGather  = getKernel("embed_gather");
    // f16 embedding gather (embeddingGpuBuf is stored fp16 to halve VRAM).
    static const char* EMBED_GATHER_F16_WGSL = R"(
enable f16;
@group(0) @binding(0) var<storage, read> EmbeddingTable: array<f16>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let normalizer = bitcast<f32>(_params_[1]);
    let token = u32(TokenId[0]);
    let base_offset = token * E;
    let i = gid.x;
    if (i >= E) { return; }
    X[i] = f32(EmbeddingTable[base_offset + i]) * normalizer;
}
)";
    auto& plEmbGatherF16 = gpu->getOrCreatePipeline("embed_gather_f16",
        std::string(EMBED_GATHER_F16_WGSL), 4);
    // Q8 embedding gather straight from the tied Q8 LM head (weights+scales),
    // avoiding a separate 0.75–1.5 GB embedding buffer. Layout matches
    // repack_q8_0: weights[row*(E/4) + i/4] packs 4 int8; scales are fp16
    // packed 2-per-u32, one per 32-element block at row*(E/32)+i/32.
    static const char* EMBED_GATHER_Q8_WGSL = R"(
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<u32>;
@group(0) @binding(2) var<storage, read> TokenId: array<i32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let normalizer = bitcast<f32>(_params_[1]);
    let token = u32(TokenId[0]);
    let i = gid.x;
    if (i >= E) { return; }
    let wIdx = token * (E / 4u) + i / 4u;
    let packed = W[wIdx];
    let byte = (packed >> ((i % 4u) * 8u)) & 0xFFu;
    let q = (i32(byte) << 24u) >> 24u;   // sign-extend int8
    let gblock = token * (E / 32u) + i / 32u;
    let sp = unpack2x16float(S[gblock / 2u]);
    let scale = select(sp.x, sp.y, (gblock & 1u) != 0u);
    X[i] = f32(q) * scale * normalizer;
}
)";
    auto& plEmbGatherQ8 = gpu->getOrCreatePipeline("embed_gather_q8",
        std::string(EMBED_GATHER_Q8_WGSL), 5);
    auto& plEmbGatherKQ = getKernel("q6k_gather");
    const bool useGelu = (cfg.activation == ActivationType::GELU);
    auto& plDownSilu   = useGelu ? getKernelGelu("q8_down_silu_add")
                                 : getKernel("q8_down_silu_add");
    const bool decodeDp4aReady = (gpu->backendType == WGPUBackendType_D3D12) &&
                                 canUse256ThreadSubgroupKernels &&
                                 canCompileEmbeddedKernel(*gpu, "q8_matmul_decode_dp4a_d3d12");
    const CompiledPipeline* plQ8DecDp4a = decodeDp4aReady
        ? &getKernel("q8_matmul_decode_dp4a_d3d12") : nullptr;

    decodeFastVariantsAvailable = decodeFastQ8Eligible && !useKQ && !weightsAreNativeQ4 &&
                                  !std::getenv("BP_Q8_BASELINE");
    tuning.decodeUseFastQkv = decodeFastVariantsAvailable;
    tuning.decodeUseFastGateup = decodeFastVariantsAvailable;
    tuning.decodeUseFastOproj = false;
    tuning.decodeUseWideFp16 = decodeWideFp16Eligible;

    // Kernel selection per projection:
    auto& plQkv = tuning.decodeUseFastQkv ? plQ8Fast : plQ8Matmul;
    auto& plOp  = tuning.decodeUseFastOproj ? plQ8Fast : plQ8Matmul;
    auto& plGu  = tuning.decodeUseFastGateup ? plQ8Fast : plQ8Matmul;
    auto& plDnSilu = plDownSilu;

    useMMA = (gpu->backendType != WGPUBackendType_D3D12) && subgroupMatrixKernelReady;
    decodePoolCapacity = chooseDecodePoolDepth(*gpu);
    decodePoolDepth = decodePoolCapacity;
    if (const char* depthEnv = std::getenv("BP_DECODE_DEPTH")) {
        int forcedDepth = std::atoi(depthEnv);
        if (forcedDepth > 0)
            decodePoolDepth = std::max(1, std::min(forcedDepth, decodePoolCapacity));
    }
    decodeCbPoolBatch = chooseDecodeCbPoolBatch(*gpu, cfg);
    const char* lmHeadKind = lmHeadIsKQ ? "k-quant" : lmHeadIsQ4 ? "q4"
        : (lmHeadIsQ8 ? "q8" : (tuning.decodeUseWideFp16 ? "fp16_wide" : "fp16"));
        fprintf(stderr, "  Initial decode heuristic: qkv=%s oproj=%s gateup=%s lm_head=%s pool=%d batch=%d\n",
        tuning.decodeUseFastQkv ? "fast" : "base",
        tuning.decodeUseFastOproj ? "fast" : "base",
        tuning.decodeUseFastGateup ? "fast" : "base",
        lmHeadKind,
        decodePoolDepth, decodeCbPoolBatch);

    // Static params (shared between both sets — read-only)
    auto makeQ8Params = [&](const std::string& name, uint32_t K, uint32_t N) -> GPUBuffer {
        uint32_t data[4] = {K, N, 0, 0};
        auto buf = gpu->createBuffer(name, 16);
        gpu->writeBuffer(buf, data, 16);
        return buf;
    };
    auto makeQ4Params = [&](const std::string& name, uint32_t K, uint32_t N) -> GPUBuffer {
        uint32_t data[4] = {0, N, K, 0};
        auto buf = gpu->createBuffer(name, 16);
        gpu->writeBuffer(buf, data, 16);
        return buf;
    };
    uint32_t qkvParamOut = qkvOut;
    if (cfg.arch == "qwen35" && cfg.ssmInnerSize > 0) {
        qkvParamOut = cfg.ssmInnerSize + 2u * cfg.ssmGroupCount * cfg.ssmStateSize;
    }
    auto q8QkvParams   = makeQ8Params("p_qkv", cfg.nEmbd, qkvParamOut);
    GPUBuffer q8QkvNormParams;
    {
        uint32_t data[4] = {cfg.nEmbd, qkvParamOut, 0, 0};
        float eps = cfg.rmsNormEps; memcpy(&data[3], &eps, 4);
        q8QkvNormParams = gpu->createBuffer("p_qkv_norm", 16);
        gpu->writeBuffer(q8QkvNormParams, data, 16);
    }
    auto q8OprojParams = makeQ8Params("p_oproj", qDim, cfg.nEmbd);
    auto q8GuParams    = makeQ8Params("p_gu", cfg.nEmbd, 2 * cfg.intermediateSize);

    // Per-layer param buffers for variable-dim models
    std::vector<GPUBuffer> perLayerQkvParams, perLayerQkvNormParams;
    std::vector<GPUBuffer> perLayerOprojParams, perLayerGuParams, perLayerDnSiluParams;
    if (cfg.hasPerLayerDims) {
        perLayerQkvParams.resize(cfg.nLayer);
        perLayerQkvNormParams.resize(cfg.nLayer);
        perLayerOprojParams.resize(cfg.nLayer);
        perLayerGuParams.resize(cfg.nLayer);
        perLayerDnSiluParams.resize(cfg.nLayer);
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            auto& pl = cfg.perLayer[li];
            uint32_t plQkvOut = layerWeights[li].qOnly
                ? pl.qDim : pl.qDim + 2 * pl.kvDim;
            perLayerQkvParams[li] = makeQ8Params("p_qkv_" + std::to_string(li), cfg.nEmbd, plQkvOut);
            {
                uint32_t data[4] = {cfg.nEmbd, plQkvOut, 0, 0};
                float eps = cfg.rmsNormEps; memcpy(&data[3], &eps, 4);
                perLayerQkvNormParams[li] = gpu->createBuffer("p_qkv_norm_" + std::to_string(li), 16);
                gpu->writeBuffer(perLayerQkvNormParams[li], data, 16);
            }
            perLayerOprojParams[li] = makeQ8Params("p_oproj_" + std::to_string(li), pl.qDim, cfg.nEmbd);
            perLayerGuParams[li] = makeQ8Params("p_gu_" + std::to_string(li), cfg.nEmbd, 2 * pl.intermediateSize);
            {
                uint32_t data[4] = {pl.intermediateSize, cfg.nEmbd, pl.intermediateSize, 0};
                perLayerDnSiluParams[li] = gpu->createBuffer(
                    "p_dn_silu_" + std::to_string(li), 16, BUF_UNIFORM | BUF_COPY_DST);
                gpu->writeBuffer(perLayerDnSiluParams[li], data, 16);
            }
        }
    }
    // Fused down+silu params: [K=IM, N=E, IM, 0]
    GPUBuffer q8DnSiluParams;
    {
        uint32_t data[4] = {cfg.intermediateSize, cfg.nEmbd, cfg.intermediateSize, 0};
        q8DnSiluParams = gpu->createBuffer("p_dn_silu", 16, BUF_UNIFORM | BUF_COPY_DST);
        gpu->writeBuffer(q8DnSiluParams, data, 16);
    }

    // K-quant param buffers: [K, N, n_blocks, row_stride_words]
    auto makeKQParams = [&](const std::string& name, uint32_t K, uint32_t N,
                            uint32_t nBlocks, uint32_t rowStride) -> GPUBuffer {
        // Q5_K/Q6_K kernels also consume word 4 as an output offset. Keep one
        // common ABI for all K-quant kernels so Dawn's inferred minimum binding
        // size is satisfied even when the offset is zero.
        uint32_t data[5] = {K, N, nBlocks, rowStride, 0};
        auto buf = gpu->createBuffer(name, sizeof(data));
        gpu->writeBuffer(buf, data, sizeof(data));
        return buf;
    };
    GPUBuffer kqQkvParams, kqOprojParams, kqGuParams, kqDnParams, kqLmParams;
    if (useKQ) {
        kqQkvParams   = makeKQParams("p_kq_qkv", cfg.nEmbd, qkvParamOut, kqQkvNBlocks, kqQkvRowStride);
        kqOprojParams = makeKQParams("p_kq_oproj", qDim, cfg.nEmbd, kqONBlocks, kqORowStride);
        kqGuParams    = makeKQParams("p_kq_gu", cfg.nEmbd, 2 * cfg.intermediateSize, kqGuNBlocks, kqGuRowStride);
        kqDnParams    = makeKQParams("p_kq_dn", cfg.intermediateSize, cfg.nEmbd, kqDnNBlocks, kqDnRowStride);
    }
    if (lmHeadIsKQ)
        kqLmParams = makeKQParams("p_kq_lm", cfg.nEmbd, cfg.nVocab, kqLmNBlocks, kqLmRowStride);

    GPUBuffer rmsParams;
    {
        uint32_t rn[4];
        rn[0] = cfg.nEmbd; rn[1] = cfg.nEmbd;
        float eps = cfg.rmsNormEps; memcpy(&rn[2], &eps, 4);
        rn[3] = 0;
        rmsParams = gpu->createBuffer("p_rms", 16);
        gpu->writeBuffer(rmsParams, rn, 16);
    }

    GPUBuffer lmheadParams;
    {
        uint32_t fp[4] = {cfg.nEmbd, cfg.nVocab, 0, 0};
        lmheadParams = gpu->createBuffer("p_lmhead", 16);
        gpu->writeBuffer(lmheadParams, fp, 16);
    }

    uint32_t argmaxNumWg = std::max(1u, (cfg.nVocab + 1023u) / 1024u);
    GPUBuffer argmaxParams;
    {
        uint32_t p[4] = {cfg.nVocab, argmaxNumWg, 0, 0};
        argmaxParams = gpu->createBuffer("p_argmax", 16);
        gpu->writeBuffer(argmaxParams, p, 16);
    }
    GPUBuffer argmaxReduceParams;
    {
        uint32_t p[4] = {argmaxNumWg, 0, 0, 0};
        argmaxReduceParams = gpu->createBuffer("p_argmax_reduce", 16);
        gpu->writeBuffer(argmaxReduceParams, p, 16);
    }

    GPUBuffer embedParams;
    {
        uint32_t p[4] = {cfg.nEmbd, 0, 0, 0};
        float normalizer = 1.0f;
        memcpy(&p[1], &normalizer, 4);
        embedParams = gpu->createBuffer("p_embed", 16);
        gpu->writeBuffer(embedParams, p, 16);
    }
    GPUBuffer embedKQParams;
    if (lmHeadIsKQ && lmHeadKQType == GGUF_TYPE_Q6_K) {
        uint32_t p[2] = {cfg.nEmbd, kqLmRowStride};
        embedKQParams = gpu->createBuffer("p_embed_kq", sizeof(p));
        gpu->writeBuffer(embedKQParams, p, sizeof(p));
    }

    // Upload embedding table to GPU (shared). Two strategies to keep the
    // resident set small enough to stay under the D3D12 per-process memory
    // budget (much smaller than total VRAM on a shared desktop — exceeding it
    // surfaces as DXGI_ERROR_DEVICE_REMOVED from CreatePlacedResource and
    // silently kills the device):
    //   1. Tied Q8 LM head resident → gather straight from it (no extra buffer).
    //   2. Otherwise store a dedicated fp16 copy (half the fp32 size).
    if (lmHeadIsKQ && lmHeadKQType == GGUF_TYPE_Q6_K &&
        cfg.tieWordEmbeddings && lmHeadKQ.handle) {
        embeddingGpuIsF16 = false;
        embeddingGatherFromKQ = true;
        embeddingGpuBuf = GPUBuffer{};
        fprintf(stderr, "  Embedding gather: from tied native Q6_K LM head (no extra buffer)\n");
    } else if (lmHeadIsQ8 && cfg.tieWordEmbeddings && lmHeadQ8W.handle) {
        embeddingGpuIsF16 = false;         // signal: use Q8 gather from lm head
        embeddingGatherFromQ8 = true;
        embeddingGpuBuf = GPUBuffer{};     // no separate embedding buffer
        fprintf(stderr, "  Embedding gather: from tied Q8 LM head (no extra buffer)\n");
    } else {
        uint64_t nEl = embeddingCPU.size();
        embeddingGpuIsF16 = true;
        std::vector<uint16_t> f16(nEl);
        for (uint64_t j = 0; j < nEl; j++) f16[j] = f32_to_fp16(embeddingCPU[j]);
        uint64_t embBytes = nEl * 2;
        embeddingGpuBuf = gpu->createBuffer("embedding_gpu", embBytes);
        const uint64_t CHUNK = 128 * 1024 * 1024;
        for (uint64_t off = 0; off < embBytes; off += CHUNK) {
            uint64_t sz = std::min(CHUNK, embBytes - off);
            wgpuQueueWriteBuffer(gpu->queue, embeddingGpuBuf.handle, off,
                                 (const uint8_t*)f16.data() + off, sz);
        }
    }

    // Initialize dynamic param templates
    ropeParamData.resize(32, 0);
    {
        auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
        uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        p[0] = cfg.nHead;  p[1] = qDim;  p[2] = kvDim;
        p[3] = 0;  p[4] = ropeHalf;  p[5] = 0;
        float eps = cfg.rmsNormEps;
        memcpy(&p[6], &eps, 4);
        // Gemma 4 normalizes V per head (without a learned scale) before it is
        // written to the KV cache.  The fused QK-norm/RoPE kernel already
        // implements this via params.v_norm; leaving it disabled corrupts the
        // donor caches subsequently reused by the shared-KV layers.
        p[7] = cfg.arch == "gemma4" ? 1 : 0;
    }
    chunkedAttnParamData.resize(32, 0);
    {
        auto* p = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
        p[0] = cfg.nKvHeads * cfg.headDim;
        p[1] = cfg.nHead / cfg.nKvHeads;
        p[2] = 0;  p[3] = 0;  p[4] = 0;
        // Gemma 4 defines self.scaling = 1.0; unlike conventional attention it
        // must not apply the additional 1/sqrt(head_dim) factor here.
        float scale = cfg.arch == "gemma4" ? 1.0f
                                             : 1.0f / sqrtf((float)cfg.headDim);
        float neg_inf = -1e9f;
        memcpy(&p[5], &scale, 4);
        memcpy(&p[6], &neg_inf, 4);
        p[7] = maxChunks;
    }
    if (cfg.arch == "qwen35" && cfg.fullAttentionInterval > 0) {
        uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        uint32_t q[8] = {cfg.nHead, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], 0u, ropeHalf};
        uint32_t k[8] = {cfg.nKvHeads, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], 0u, ropeHalf};
        uint32_t kv[8] = {cfg.nKvHeads * cfg.headDim, 0u, 0, 0, 0, 0, 0, 0};
        q35RopeQParamsBuf = gpu->createBuffer("p_q35_rope_q", 32);
        q35RopeKParamsBuf = gpu->createBuffer("p_q35_rope_k", 32);
        q35KvWriteParamsBuf = gpu->createBuffer("p_q35_kv_write", 32);
        gpu->writeBuffer(q35RopeQParamsBuf, q, 32);
        gpu->writeBuffer(q35RopeKParamsBuf, k, 32);
        gpu->writeBuffer(q35KvWriteParamsBuf, kv, 32);
    }

    // Ensure layer norms exist (identity for models without QK norm)
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        if (!lw.qNorm.handle) {
            std::vector<float> ones(cfg.headDim, 1.0f);
            lw.qNorm = gpu->createBuffer("qnorm_id_" + std::to_string(i), cfg.headDim * 4);
            gpu->writeBuffer(lw.qNorm, ones.data(), cfg.headDim * 4);
        }
        if (!lw.kNorm.handle) {
            std::vector<float> ones(cfg.headDim, 1.0f);
            lw.kNorm = gpu->createBuffer("knorm_id_" + std::to_string(i), cfg.headDim * 4);
            gpu->writeBuffer(lw.kNorm, ones.data(), cfg.headDim * 4);
        }
    }

    // Shared argmax result buffer
    argmaxResultBuf = gpu->createBuffer("argmax_result", 4);
    argmaxPartialsBuf = gpu->createBuffer("argmax_partials", argmaxNumWg * 2u * 4u);

    // PLE buffers
    if (cfg.pleSize > 0) {
        uint32_t totalPleDim = cfg.pleSize * cfg.nLayer;
        pleInputBuf = gpu->createBuffer("ple_input", totalPleDim * 4);
        if (pleGpuPreprocess)
            pleProjRawBuf = gpu->createBuffer("ple_proj_raw", totalPleDim * 4);
        pleBuf = gpu->createBuffer("ple_buf", cfg.pleSize * 4);
        pleOutBuf = gpu->createBuffer("ple_out", cfg.nEmbd * 4);
    }

    // ─── Single set of intermediate buffers ───────────────────────────────
    // Sized to max per-layer dimensions for variable-dim models (Gemma 4).
    uint32_t maxQkvOutBuf = qkvOut;
    uint32_t maxQDimBuf = qDim;
    uint32_t maxIMBuf = cfg.intermediateSize;
    if (cfg.arch == "qwen35" && cfg.ssmInnerSize > 0) {
        uint32_t ssmConvChannels = cfg.ssmInnerSize + 2u * cfg.ssmGroupCount * cfg.ssmStateSize;
        maxQkvOutBuf = std::max(maxQkvOutBuf, ssmConvChannels);
        maxQkvOutBuf = std::max(maxQkvOutBuf, 2u * cfg.nHead * cfg.headDim);
        maxQDimBuf = std::max(maxQDimBuf, cfg.nHead * cfg.headDim);
    }
    for (auto& pl : cfg.perLayer) {
        uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
        if (plQkvOut > maxQkvOutBuf) maxQkvOutBuf = plQkvOut;
        if (pl.qDim > maxQDimBuf) maxQDimBuf = pl.qDim;
        if (pl.intermediateSize > maxIMBuf) maxIMBuf = pl.intermediateSize;
    }
    xBuf          = gpu->createBuffer("x", cfg.nEmbd * 4);
    normOutBuf    = gpu->createBuffer("norm_out", cfg.nEmbd * 4);
    qkvBuf        = gpu->createBuffer("qkv_out", maxQkvOutBuf * 4);
    qRotBuf       = gpu->createBuffer("q_rot", maxQDimBuf * 4);
    attnOutBuf    = gpu->createBuffer("attn_out", maxQDimBuf * 4);
    projOutBuf    = gpu->createBuffer("proj_out", cfg.nEmbd * 4);
    gateUpBuf     = gpu->createBuffer("gate_up", 2 * maxIMBuf * 4);
    // Scratch K/V write targets for shared-KV (Q-only) layers: the fused-rope
    // kernel always writes K/V somewhere; for those layers we discard its K/V
    // and read the real K/V from the source layer's cache instead.
    if (cfg.sharedKvLayers > 0) {
        uint32_t kvScratchDim = maxSeqLen * cfg.nKvHeads * cfg.headDim * 2; // fp16
        qOnlyScratchK = gpu->createBuffer("qonly_scratch_k", kvScratchDim);
        qOnlyScratchV = gpu->createBuffer("qonly_scratch_v", kvScratchDim);
    }

    // MoE intermediate buffers (allocate when MoE arch detected)
    if (cfg.numExperts > 0) {
        moeRouterOutBuf  = gpu->createBuffer("moe_router_out", cfg.numExperts * 4);
        moeIndicesBuf    = gpu->createBuffer("moe_indices",    cfg.numExpertsPerTok * 4);
        moeWeightsBuf    = gpu->createBuffer("moe_weights",    cfg.numExpertsPerTok * 4);
        moeExpertOutBuf  = gpu->createBuffer("moe_expert_out", cfg.nEmbd * 4);
        moeShexpGateUpBuf= gpu->createBuffer("moe_shexp_gu",   2 * cfg.moeIntermediateSize * 4);
        moeShexpActBuf   = gpu->createBuffer("moe_shexp_act",  cfg.moeIntermediateSize * 4);
        moeRoutedGateBuf = gpu->createBuffer("moe_routed_gate", cfg.moeIntermediateSize * 4);
        moeRoutedUpBuf   = gpu->createBuffer("moe_routed_up",   cfg.moeIntermediateSize * 4);
        moeRoutedActBuf  = gpu->createBuffer("moe_routed_act",  cfg.moeIntermediateSize * 4);
        fprintf(stderr, "  MoE intermediate buffers allocated: router=%uB indices=%uB weights=%uB expert_out=%uB shexp_gu=%uB shexp_act=%uB routed=3x%uB\n",
                cfg.numExperts * 4u, cfg.numExpertsPerTok * 4u, cfg.numExpertsPerTok * 4u,
                cfg.nEmbd * 4u, 2 * cfg.moeIntermediateSize * 4u, cfg.moeIntermediateSize * 4u,
                cfg.moeIntermediateSize * 4u);
    }
    // SSM persistent buffers per layer (only for SSM layers in hybrid archs)
    if (cfg.ssmInnerSize > 0) {
        ssmConvState.resize(cfg.nLayer);
        ssmHState.resize(cfg.nLayer);
        size_t convChannels = (cfg.arch == "qwen35")
            ? (size_t)cfg.ssmInnerSize + 2ull * cfg.ssmGroupCount * cfg.ssmStateSize
            : (size_t)cfg.ssmInnerSize;
        size_t convBytes = convChannels * cfg.ssmConvKernel * 4;
        size_t hElems = (size_t)cfg.ssmInnerSize * cfg.ssmStateSize;
        if (cfg.arch == "qwen35" && cfg.ssmTimeStepRank > 0) {
            size_t headV = cfg.ssmInnerSize / cfg.ssmTimeStepRank;
            hElems = (size_t)cfg.ssmTimeStepRank * headV * headV;
        }
        size_t hBytes = hElems * 4;
        uint32_t ssmLayerCount = 0;
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            if (cfg.isAttentionLayer(li)) continue;
            ssmConvState[li] = gpu->createBuffer("ssm_conv_state_L" + std::to_string(li), convBytes);
            ssmHState[li]    = gpu->createBuffer("ssm_h_state_L"    + std::to_string(li), hBytes);
            ssmLayerCount++;
        }
        fprintf(stderr, "  SSM state buffers: %u SSM layers x (conv=%zuB + h=%zuB) = %zu MB total\n",
                ssmLayerCount, convBytes, hBytes,
                (size_t)ssmLayerCount * (convBytes + hBytes) / (1024 * 1024));
        if (cfg.arch == "qwen35") {
            uint32_t convChannels = cfg.ssmInnerSize + 2u * cfg.ssmGroupCount * cfg.ssmStateSize;
            uint32_t qkDim = cfg.ssmGroupCount * cfg.ssmStateSize;
            q35ConvOutBuf  = gpu->createBuffer("q35_ssm_conv_out", convChannels * 4);
            q35SsmQBuf     = gpu->createBuffer("q35_ssm_q", qkDim * 4);
            q35SsmKBuf     = gpu->createBuffer("q35_ssm_k", qkDim * 4);
            q35SsmVBuf     = gpu->createBuffer("q35_ssm_v", cfg.ssmInnerSize * 4);
            q35SsmBetaBuf  = gpu->createBuffer("q35_ssm_beta", cfg.ssmTimeStepRank * 4);
            q35SsmAlphaBuf = gpu->createBuffer("q35_ssm_alpha", cfg.ssmTimeStepRank * 4);
            q35SsmGateBuf  = gpu->createBuffer("q35_ssm_gate", cfg.ssmTimeStepRank * 4);
            q35SsmYBuf     = gpu->createBuffer("q35_ssm_y", cfg.ssmInnerSize * 4);
            q35SsmNormBuf  = gpu->createBuffer("q35_ssm_normed", cfg.ssmInnerSize * 4);
            q35SsmZBuf     = gpu->createBuffer("q35_ssm_z", cfg.ssmInnerSize * 4);
        }
    }

    // qwen35moe attention intermediate buffers (sized for actual Q/K/V dims)
    if ((cfg.arch == "qwen35" || cfg.numExperts > 0) && cfg.fullAttentionInterval > 0) {
        // Discover sizes from first attention layer's qjW buffer.
        // qjW is sized [qOutDim * cfg.nEmbd] in Q8 (1 byte/elem + scale per 32);
        // qOutDim = 2 * qDim_actual. We use a generous bound: 2 * 4 * cfg.nEmbd
        // (covers qOutDim up to 4x nEmbd, fits qwen35moe's 8192 with nEmbd=2048).
        uint32_t maxQjDim = 4u * cfg.nEmbd;          // bound on 2*qDim_actual
        uint32_t maxQDim  = maxQjDim / 2u;
        uint32_t maxKvDim = 2u * cfg.nEmbd;          // bound on kvDim_actual
        q35QjBuf      = gpu->createBuffer("q35_qj", maxQjDim * 4);
        q35QBuf       = gpu->createBuffer("q35_q",  maxQDim * 4);
        q35GateBuf    = gpu->createBuffer("q35_gate", maxQDim * 4);
        q35KBuf       = gpu->createBuffer("q35_k",  maxKvDim * 4);
        q35VBuf       = gpu->createBuffer("q35_v",  maxKvDim * 4);
        q35AttnOutBuf = gpu->createBuffer("q35_attn_out", maxQDim * 4);
        // Cos/sin table for MRoPE: 4 sections × max 16 pairs × 2 (cos,sin) = 128 floats
        q35CosSinBuf  = gpu->createBuffer("q35_cossin", 128 * 4);
        fprintf(stderr, "  qwen35 attn buffers: qj=%uB q=%uB gate=%uB k=%uB v=%uB attn=%uB\n",
                maxQjDim*4u, maxQDim*4u, maxQDim*4u, maxKvDim*4u, maxKvDim*4u, maxQDim*4u);
    }

    // K-quant needs a separate SiLU-mul output buffer (used for down proj input)
    GPUBuffer siluMulOutBuf;
    bool hasNativeKQDown = useKQ;
    if (!hasNativeKQDown) {
        for (const auto& lw : layerWeights)
            if (lw.dnKQ.handle) { hasNativeKQDown = true; break; }
    }
    if (hasNativeKQDown) {
        siluMulOutBuf = gpu->createBuffer("silu_mul_out", maxIMBuf * 4);
        siluMulDebugBuf = siluMulOutBuf;
    }
    // llama.cpp quantizes the activation once before quantized matvec and
    // reuses it across output rows. Keep one scratch pair because decode
    // dispatches consume it sequentially.
    GPUBuffer kqActQ8Buf, kqActScaleBuf;
    if (useQ4KDp4a) {
        kqActQ8Buf = gpu->createBuffer("kq_act_q8", maxIMBuf);
        kqActScaleBuf = gpu->createBuffer("kq_act_scales", maxIMBuf / 8);
    }

    rstdBuf       = gpu->createBuffer("rstd", 16);
    logitsBuf     = gpu->createBuffer("logits", cfg.nVocab * 4);
    attnPartialsBuf = gpu->createBuffer("attn_partials",
        cfg.nHead * maxChunks * (cfg.headDim + 2) * 4);

    // Single set of dynamic params (writeBuffer is queue-sequenced)
    fusedRopeParamsBuf = gpu->createBuffer("p_frope", 32);
    gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);
    chunkedAttnParamsBuf = gpu->createBuffer("p_cattn", 32);
    gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);

    // Sliding window attention: separate param buffer with clamped T_total
    bool hasSWA = !cfg.layerAttnTypes.empty() && cfg.slidingWindow > 0
                  && !std::getenv("BP_NO_SWA");
    if (hasSWA) {
        chunkedAttnParamsBufSWA = gpu->createBuffer("p_cattn_swa", 32);
        gpu->writeBuffer(chunkedAttnParamsBufSWA, chunkedAttnParamData.data(), 32);
    }

    // Detect variable per-layer head dims (Gemma 4). When present, each layer
    // gets its own rope + attention param buffer with the layer's head dim
    // baked in; these are refreshed per token in updateDecodeParams.
    hasVariableHeadDim = false;
    if (cfg.hasPerLayerDims) {
        for (uint32_t li = 1; li < cfg.nLayer; li++)
            if (cfg.perLayer[li].headDim != cfg.perLayer[0].headDim)
                hasVariableHeadDim = true;
    }
    if (hasVariableHeadDim) {
        perLayerRopeParamBufs.resize(cfg.nLayer);
        perLayerAttnParamBufs.resize(cfg.nLayer);
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            perLayerRopeParamBufs[li] = gpu->createBuffer("p_frope_" + std::to_string(li), 32);
            perLayerAttnParamBufs[li] = gpu->createBuffer("p_cattn_" + std::to_string(li), 32);
        }
        fprintf(stderr, "  Variable head dims: per-layer rope/attn params (%u layers)\n", cfg.nLayer);
    }

    // ─── Build dispatch list (single — identical for every token) ─────────
    allDecodeDispatches.reserve(cfg.nLayer * 11 + 2);
    decodeDispatchIndices.assign(cfg.nLayer, {});
    decodeVariantBGs.assign(cfg.nLayer, {});
    ropeDispatchIndices.assign(cfg.nLayer, -1);
    attnP1DispatchIndices.assign(cfg.nLayer, -1);
    attnP2DispatchIndices.assign(cfg.nLayer, -1);
    q35QRoPEDispatchIndices.assign(cfg.nLayer, -1);
    q35KvWriteDispatchIndices.assign(cfg.nLayer, -1);
    argmaxDispatchIndex = -1;
    argmaxReduceDispatchIndex = -1;
    pleTokenGatherDispatchIndex = -1;

    if (pleGpuPreprocess && plPleTokenGather &&
        (pleTokenEmbAsymmetric || plQ4A32) && plPleProjectCombine) {
        uint32_t totalPleDim = cfg.pleSize * cfg.nLayer;
        uint32_t gatherData[4] = {totalPleDim, 0, cfg.nVocab, cfg.pleSize};
        float tokenScale = sqrtf((float)cfg.pleSize);
        memcpy(&gatherData[1], &tokenScale, 4);
        pleTokenGatherParams = gpu->createBuffer("p_ple_token_gather", 16);
        gpu->writeBuffer(pleTokenGatherParams, gatherData, 16);
        pleModelProjParams = makeQ4Params("p_ple_model_proj", cfg.nEmbd, totalPleDim);
        uint32_t combineData[4] = {cfg.pleSize, cfg.nLayer, 0, 0};
        memcpy(&combineData[2], &cfg.rmsNormEps, 4);
        pleCombineParams = gpu->createBuffer("p_ple_combine", 16);
        gpu->writeBuffer(pleCombineParams, combineData, 16);

        auto bgGather = pleTokenEmbAsymmetric
            ? makeBG(*plPleTokenGather, {
                {0, pleTokenEmbW}, {1, pleTokenEmbS}, {2, pleTokenEmbZ},
                {3, argmaxResultBuf}, {4, pleInputBuf}, {5, pleTokenGatherParams}})
            : makeBG(*plPleTokenGather, {
                {0, pleTokenEmbW}, {1, pleTokenEmbS}, {2, argmaxResultBuf},
                {3, pleInputBuf}, {4, pleTokenGatherParams}});
        pleTokenGatherDispatchIndex = (int)allDecodeDispatches.size();
        allDecodeDispatches.push_back({plPleTokenGather->pipeline, bgGather,
            (totalPleDim + 255) / 256, 1, 1, "ple_token_gather"});

        if (pleTokenEmbAsymmetric) {
            auto q8p = makeQ8Params("p_ple_model_proj_q8", cfg.nEmbd, totalPleDim);
            auto bgProj = makeBG(plQ8Matmul, {
                {0, xBuf}, {1, pleModelProjW}, {2, pleModelProjS},
                {3, zeroBiasV}, {4, pleProjRawBuf}, {5, q8p}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgProj,
                1, (totalPleDim + Q8_TILE - 1) / Q8_TILE, 1, "ple_model_proj"});
        } else {
            auto bgProj = makeBG(*plQ4A32, {
                {0, xBuf}, {1, pleModelProjW}, {2, pleModelProjS},
                {3, pleProjRawBuf}, {4, pleModelProjParams}});
            allDecodeDispatches.push_back({plQ4A32->pipeline, bgProj,
                (totalPleDim + 3) / 4, 1, 1, "ple_model_proj"});
        }

        auto bgCombine = makeBG(*plPleProjectCombine, {
            {0, pleProjRawBuf}, {1, pleProjNormW},
            {2, pleInputBuf}, {3, pleCombineParams}});
        allDecodeDispatches.push_back({plPleProjectCombine->pipeline, bgCombine,
            cfg.nLayer, 1, 1, "ple_project_combine"});
    }

    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& lw = layerWeights[i];
        auto& di = decodeDispatchIndices[i];
        auto& vbg = decodeVariantBGs[i];
        std::string L = "L" + std::to_string(i) + "/";
        auto& pl = cfg.perLayer[i];
        uint32_t plQkvOut = pl.qDim + 2 * pl.kvDim;
        uint32_t plQDim = pl.qDim;
        uint32_t plIM = pl.intermediateSize;
        auto& layerQkvP = cfg.hasPerLayerDims ? perLayerQkvParams[i] : q8QkvParams;
        auto& layerQkvNP = cfg.hasPerLayerDims ? perLayerQkvNormParams[i] : q8QkvNormParams;
        auto& layerOpP = cfg.hasPerLayerDims ? perLayerOprojParams[i] : q8OprojParams;
        auto& layerGuP = cfg.hasPerLayerDims ? perLayerGuParams[i] : q8GuParams;
        auto& layerDnP = cfg.hasPerLayerDims ? perLayerDnSiluParams[i] : q8DnSiluParams;
        bool q35CustomAttn = false;
        if (std::getenv("BP_TRACE_PIPELINE")) {
            const char* kind =
                (cfg.arch == "qwen35" && cfg.ssmInnerSize > 0 && !cfg.isAttentionLayer(i)) ? "qwen35-ssm" :
                (cfg.arch == "qwen35" && cfg.isAttentionLayer(i)) ? "qwen35-attn" : "standard";
            fprintf(stderr, "[trace] build layer %u (%s)\n", i, kind);
            fflush(stderr);
        }

        auto mkP32 = [&](const std::string& name, std::initializer_list<uint32_t> data) -> GPUBuffer {
            uint32_t buf[8] = {0};
            size_t i2 = 0; for (uint32_t v : data) { buf[i2++] = v; }
            auto b = gpu->createBuffer(name, 32);
            gpu->writeBuffer(b, buf, 32);
            return b;
        };

        if (cfg.arch == "qwen35" && cfg.ssmInnerSize > 0 && !cfg.isAttentionLayer(i)) {
            q35CustomAttn = true;
            uint32_t convChannels = cfg.ssmInnerSize + 2u * cfg.ssmGroupCount * cfg.ssmStateSize;
            uint32_t qkDimSsm = cfg.ssmGroupCount * cfg.ssmStateSize;
            uint32_t headV = cfg.ssmInnerSize / cfg.ssmTimeStepRank;
            bool useSsmKQ = lw.qkvKQ.handle != nullptr;

            {
                auto bg = makeBG(plRmsNorm, {
                    {0, xBuf}, {1, normOutBuf}, {2, lw.inputNorm},
                    {3, rstdBuf}, {4, rmsParams}});
                allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_norm"});
            }

            if (useSsmKQ) {
                auto* layerKQ = kqPipelineFor(lw.qkvKQType);
                uint32_t layerTile = kqTileFor(lw.qkvKQType);
                auto layerParams = makeKQParams("p_kq_ssm_qkv_" + std::to_string(i),
                    cfg.nEmbd, convChannels, lw.qkvKQNBlocks, lw.qkvKQRowStride);
                auto bg = makeBG(*layerKQ, {
                    {0, normOutBuf}, {1, lw.qkvKQ}, {2, zeroBiasQKV},
                    {3, qkvBuf}, {4, layerParams}});
                allDecodeDispatches.push_back({layerKQ->pipeline, bg,
                    1, (convChannels + layerTile - 1) / layerTile, 1, L+"ssm_qkv"});
            } else {
                auto p = mkP32("p_ssm_qkv_L"+std::to_string(i), {cfg.nEmbd, convChannels});
                auto bg = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                    {3, zeroBiasQKV}, {4, qkvBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (convChannels + Q8_TILE - 1) / Q8_TILE, 1, L+"ssm_qkv"});
            }

            {
                if (lw.attnGateKQ.handle) {
                    auto* layerKQ = kqPipelineFor(lw.attnGateKQType);
                    uint32_t tile = kqTileFor(lw.attnGateKQType);
                    auto p = makeKQParams("p_kq_ssm_z_" + std::to_string(i),
                        cfg.nEmbd, cfg.ssmInnerSize, lw.attnGateKQNBlocks,
                        lw.attnGateKQRowStride);
                    auto bg = makeBG(*layerKQ, {{0,normOutBuf},{1,lw.attnGateKQ},
                        {2,zeroBiasQKV},{3,q35SsmZBuf},{4,p}});
                    allDecodeDispatches.push_back({layerKQ->pipeline,bg,1,
                        (cfg.ssmInnerSize+tile-1)/tile,1,L+"ssm_z"});
                } else {
                    auto p = mkP32("p_ssm_z_L"+std::to_string(i), {cfg.nEmbd, cfg.ssmInnerSize});
                    auto bg = makeBG(plQ8Matmul, {
                        {0, normOutBuf}, {1, lw.attnGateW}, {2, lw.attnGateS},
                        {3, zeroBiasQKV}, {4, q35SsmZBuf}, {5, p}});
                    allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                        1, (cfg.ssmInnerSize + Q8_TILE - 1) / Q8_TILE, 1, L+"ssm_z"});
                }
            }
            bool useFusedBetaAlpha = lw.ssmBetaAlphaW.handle && lw.ssmBetaAlphaS.handle;
            if (useFusedBetaAlpha) {
                auto& plBetaAlphaGate = getKernel("qwen35_beta_alpha_gate_q8");
                auto p = mkP32("p_ssm_beta_alpha_gate_L"+std::to_string(i),
                               {cfg.nEmbd, cfg.ssmTimeStepRank});
                auto bg = makeBG(plBetaAlphaGate, {
                    {0, normOutBuf}, {1, lw.ssmBetaAlphaW}, {2, lw.ssmBetaAlphaS},
                    {3, lw.ssmDtBias}, {4, lw.ssmA}, {5, q35SsmBetaBuf},
                    {6, q35SsmGateBuf}, {7, p}});
                allDecodeDispatches.push_back({plBetaAlphaGate.pipeline, bg,
                    1, (2u * cfg.ssmTimeStepRank + Q8_TILE - 1) / Q8_TILE, 1,
                    L+"ssm_beta_alpha_gate"});
            } else {
                auto pBeta = mkP32("p_ssm_beta_L"+std::to_string(i), {cfg.nEmbd, cfg.ssmTimeStepRank});
                auto bgBeta = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.ssmBetaW}, {2, lw.ssmBetaS},
                    {3, zeroBiasQKV}, {4, q35SsmBetaBuf}, {5, pBeta}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgBeta,
                    1, (cfg.ssmTimeStepRank + Q8_TILE - 1) / Q8_TILE, 1, L+"ssm_beta"});

                auto pAlpha = mkP32("p_ssm_alpha_L"+std::to_string(i), {cfg.nEmbd, cfg.ssmTimeStepRank});
                auto bgAlpha = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.ssmAlphaW}, {2, lw.ssmAlphaS},
                    {3, zeroBiasQKV}, {4, q35SsmAlphaBuf}, {5, pAlpha}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgAlpha,
                    1, (cfg.ssmTimeStepRank + Q8_TILE - 1) / Q8_TILE, 1, L+"ssm_alpha"});
            }
            if (!useFusedBetaAlpha) {
                auto& plGate = getKernel("qwen35_alpha_beta_gate");
                auto p = mkP32("p_ssm_abg_L"+std::to_string(i),
                               {cfg.ssmTimeStepRank, 0u});
                auto bg = makeBG(plGate, {
                    {0, q35SsmBetaBuf}, {1, q35SsmAlphaBuf}, {2, lw.ssmDtBias},
                    {3, lw.ssmA}, {4, q35SsmBetaBuf}, {5, q35SsmGateBuf}, {6, p}});
                allDecodeDispatches.push_back({plGate.pipeline, bg,
                    (cfg.ssmTimeStepRank + 63) / 64, 1, 1, L+"ssm_abg"});
            }
            if ((qwen35VulkanDecode || qwen35SubgroupSuite) &&
                cfg.ssmStateSize == 128u && headV == 128u &&
                !std::getenv("BP_UNFUSED_Q35_CONV")) {
                static const char* CONV_SPLIT_WGSL = R"(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> State: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> CW: array<f32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Q: array<f32>;
@group(0) @binding(5) var<storage, read_write> KOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> V: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;
var<workgroup> sums: array<f32, 4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nk = P[0]; let nv = P[1]; let dk = P[2]; let dv = P[3]; let convK = P[4];
    let kind = wid.x; let h = wid.y; let d = lid.x;
    let dim = select(dk, dv, kind == 2u);
    let heads = select(nk, nv, kind == 2u);
    if (h >= heads) { return; }
    let qSize = nk * dk;
    let channelOffset = select(select(0u, qSize, kind == 1u), qSize * 2u, kind == 2u);
    let channel = channelOffset + h * dim + d;
    let base = channel * convK;
    var acc = Bias[channel];
    for (var j = 0u; j + 1u < convK; j++) {
        let old = State[base + j + 1u];
        State[base + j] = old;
        acc += CW[base + j] * old;
    }
    let newest = X[channel];
    State[base + convK - 1u] = newest;
    acc += CW[base + convK - 1u] * newest;
    let value = acc / (1.0 + exp(-acc));
    if (kind == 2u) {
        V[h * dv + d] = value;
        return;
    }
    let ws = subgroupAdd(value * value);
    if ((d & 31u) == 0u) { sums[d / 32u] = ws; }
    workgroupBarrier();
    let eps = bitcast<f32>(P[5]);
    let inv = 1.0 / max(sqrt(sums[0] + sums[1] + sums[2] + sums[3]), eps);
    if (kind == 0u) { Q[h * dk + d] = value * inv; }
    else { KOut[h * dk + d] = value * inv; }
}
)";
                auto& plConvSplit = gpu->getOrCreatePipeline(
                    "qwen35_conv_silu_split_l2", std::string(CONV_SPLIT_WGSL), 8);
                uint32_t epsBits; float eps = cfg.rmsNormEps; memcpy(&epsBits, &eps, 4);
                auto p = mkP32("p_ssm_conv_split_L" + std::to_string(i),
                    {cfg.ssmGroupCount, cfg.ssmTimeStepRank, cfg.ssmStateSize,
                     headV, cfg.ssmConvKernel, epsBits});
                auto bg = makeBG(plConvSplit, {
                    {0, ssmConvState[i]}, {1, qkvBuf}, {2, lw.ssmConv1dW},
                    {3, zeroBiasQKV}, {4, q35SsmQBuf}, {5, q35SsmKBuf},
                    {6, q35SsmVBuf}, {7, p}});
                allDecodeDispatches.push_back({plConvSplit.pipeline, bg,
                    3, std::max(cfg.ssmGroupCount, cfg.ssmTimeStepRank), 1,
                    L+"ssm_conv_split"});
            } else {
            {
                auto& plConvSilu = getKernel("qwen35_conv_update_silu");
                auto p = mkP32("p_ssm_conv_silu_L"+std::to_string(i), {convChannels, cfg.ssmConvKernel});
                auto bg = makeBG(plConvSilu, {
                    {0, ssmConvState[i]}, {1, qkvBuf}, {2, lw.ssmConv1dW},
                    {3, zeroBiasQKV}, {4, q35ConvOutBuf}, {5, p}});
                allDecodeDispatches.push_back({plConvSilu.pipeline, bg,
                    (convChannels + 255) / 256, 1, 1, L+"ssm_conv_silu"});
            }
            {
                auto& plSplit = getKernel("qwen35_split_qkv_l2");
                uint32_t epsBits; float eps = cfg.rmsNormEps; memcpy(&epsBits, &eps, 4);
                auto p = mkP32("p_ssm_split_L"+std::to_string(i),
                               {cfg.ssmGroupCount, cfg.ssmTimeStepRank, cfg.ssmStateSize, headV, epsBits});
                auto bg = makeBG(plSplit, {
                    {0, q35ConvOutBuf}, {1, q35SsmQBuf}, {2, q35SsmKBuf},
                    {3, q35SsmVBuf}, {4, p}});
                allDecodeDispatches.push_back({plSplit.pipeline, bg,
                    3, std::max(cfg.ssmGroupCount, cfg.ssmTimeStepRank), 1, L+"ssm_split"});
            }
            }
            {
                const bool amdValueMajorState = isAmdAdapter &&
                    (cfg.ssmTimeStepRank == 16u || cfg.ssmTimeStepRank == 32u);
                const bool nvidiaValueMajorState = isNvidiaAdapter &&
                    cfg.ssmTimeStepRank == 32u &&
                    (qwen35VulkanDecode || qwen35SubgroupSuite);
                const auto& plDelta = amdValueMajorState
                    ? gpu->getOrCreatePipeline("delta_net_decode_value_major",
                        deltaNetValueMajorSource(WGSL_DELTA_NET_DECODE), 8)
                    : nvidiaValueMajorState
                    ? gpu->getOrCreatePipeline("delta_net_decode_x2_value_major",
                        deltaNetValueMajorSource(WGSL_DELTA_NET_DECODE_X2), 8)
                    : (qwen35VulkanDecode || qwen35SubgroupSuite)
                        ? getKernel("delta_net_decode_x2") : getKernel("delta_net_decode");
                auto p = mkP32("p_ssm_delta_L"+std::to_string(i),
                               {cfg.ssmTimeStepRank, cfg.ssmGroupCount, cfg.ssmStateSize, headV});
                auto bg = makeBG(plDelta, {
                    {0, q35SsmQBuf}, {1, q35SsmKBuf}, {2, q35SsmVBuf},
                    {3, q35SsmBetaBuf}, {4, q35SsmGateBuf}, {5, ssmHState[i]},
                    {6, q35SsmYBuf}, {7, p}});
                const uint32_t deltaY = (qwen35VulkanDecode || qwen35SubgroupSuite)
                    ? ((headV + 1u) / 2u) : headV;
                allDecodeDispatches.push_back({plDelta.pipeline, bg,
                    cfg.ssmTimeStepRank, deltaY, 1, L+"ssm_delta"});
            }
            {
                auto& plNormGate = getKernel("qwen35_norm_gated");
                uint32_t epsBits; float eps = cfg.rmsNormEps; memcpy(&epsBits, &eps, 4);
                auto p = mkP32("p_ssm_norm_gate_L"+std::to_string(i),
                               {cfg.ssmTimeStepRank, headV, epsBits});
                auto bg = makeBG(plNormGate, {
                    {0, q35SsmYBuf}, {1, lw.ssmNorm}, {2, q35SsmZBuf},
                    {3, q35SsmNormBuf}, {4, p}});
                allDecodeDispatches.push_back({plNormGate.pipeline, bg,
                    cfg.ssmTimeStepRank, 1, 1, L+"ssm_norm_gate"});
            }
            {
                if (lw.ssmOutKQ.handle) {
                    auto* layerKQ = kqPipelineFor(lw.ssmOutKQType);
                    uint32_t tile = kqTileFor(lw.ssmOutKQType);
                    auto p = makeKQParams("p_kq_ssm_out_" + std::to_string(i),
                        cfg.ssmInnerSize, cfg.nEmbd, lw.ssmOutKQNBlocks,
                        lw.ssmOutKQRowStride);
                    auto bg = makeBG(*layerKQ, {{0,q35SsmNormBuf},{1,lw.ssmOutKQ},
                        {2,zeroBiasE},{3,projOutBuf},{4,p}});
                    allDecodeDispatches.push_back({layerKQ->pipeline,bg,1,
                        (cfg.nEmbd+tile-1)/tile,1,L+"ssm_out"});
                } else {
                    auto p = mkP32("p_ssm_out_L"+std::to_string(i), {cfg.ssmInnerSize, cfg.nEmbd});
                    auto bg = makeBG(plQ8Matmul, {
                        {0, q35SsmNormBuf}, {1, lw.ssmOutW}, {2, lw.ssmOutS},
                        {3, zeroBiasE}, {4, projOutBuf}, {5, p}});
                    allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                        1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"ssm_out"});
                }
            }
        }

        // ── Qwen3.5 full-attention layer ───────────────────────────────────
        // Joint Q+gate, separate K/V, QK normalization, MRoPE, gated output.
        // q35CustomAttn suppresses the standard attention path below.
        if (cfg.arch == "qwen35" && cfg.fullAttentionInterval > 0 && cfg.isAttentionLayer(i) &&
            (layerWeights[i].qjKQ.handle || layerWeights[i].qjW.handle) &&
            (layerWeights[i].kSepKQ.handle || layerWeights[i].kSepW.handle) &&
            (layerWeights[i].vSepKQ.handle || layerWeights[i].vSepW.handle)) {
            q35CustomAttn = true;
            auto& lwQ35 = layerWeights[i];
            if (i == 3) {
                fprintf(stderr, "  qwen35 attn dispatch wired for layer %u\n", i);
            }

            uint32_t qDimAct = pl.qDim;
            uint32_t qOutDim = 2u * qDimAct;
            uint32_t headDimAct = pl.headDim;
            uint32_t kvDimAct = pl.kvDim;

            // Full-attention layers need their own pre-attention norm just as
            // the recurrent layers do. Without this, Q/K/V consumed the stale
            // pre-FFN norm left by the preceding layer.
            {
                auto bg = makeBG(plRmsNorm, {
                    {0, xBuf}, {1, normOutBuf}, {2, lwQ35.inputNorm},
                    {3, rstdBuf}, {4, rmsParams}});
                allDecodeDispatches.push_back({plRmsNorm.pipeline, bg,
                    1, 1, 1, L+"rms_norm"});
            }

            // 2. Q matmul: normOutBuf @ qjW → q35QjBuf
            {
                if (lwQ35.qjKQ.handle) {
                    auto* kp=kqPipelineFor(lwQ35.qjKQType);auto tile=kqTileFor(lwQ35.qjKQType);
                    auto p=makeKQParams("p_kq_q35_q_"+std::to_string(i),cfg.nEmbd,qOutDim,lwQ35.qjKQNBlocks,lwQ35.qjKQRowStride);
                    auto bg=makeBG(*kp,{{0,normOutBuf},{1,lwQ35.qjKQ},{2,zeroBiasV},{3,q35QjBuf},{4,p}});
                    allDecodeDispatches.push_back({kp->pipeline,bg,1,(qOutDim+tile-1)/tile,1,L+"q35_q"});
                } else {
                    auto p = mkP32("p_q35_q_L"+std::to_string(i), {cfg.nEmbd, qOutDim});
                    auto bg = makeBG(plQ8Matmul, {{0,normOutBuf},{1,lwQ35.qjW},{2,lwQ35.qjS},{3,zeroBiasV},{4,q35QjBuf},{5,p}});
                    allDecodeDispatches.push_back({plQ8Matmul.pipeline,bg,1,(qOutDim+Q8_TILE-1)/Q8_TILE,1,L+"q35_q"});
                }
            }
            // 3. K matmul
            {
                if (lwQ35.kSepKQ.handle) {
                    auto* kp=kqPipelineFor(lwQ35.kSepKQType);auto tile=kqTileFor(lwQ35.kSepKQType);
                    auto p=makeKQParams("p_kq_q35_k_"+std::to_string(i),cfg.nEmbd,kvDimAct,lwQ35.kSepKQNBlocks,lwQ35.kSepKQRowStride);
                    auto bg=makeBG(*kp,{{0,normOutBuf},{1,lwQ35.kSepKQ},{2,zeroBiasV},{3,q35KBuf},{4,p}});
                    allDecodeDispatches.push_back({kp->pipeline,bg,1,(kvDimAct+tile-1)/tile,1,L+"q35_k"});
                } else {
                    auto p=mkP32("p_q35_k_L"+std::to_string(i),{cfg.nEmbd,kvDimAct});
                    auto bg=makeBG(plQ8Matmul,{{0,normOutBuf},{1,lwQ35.kSepW},{2,lwQ35.kSepS},{3,zeroBiasV},{4,q35KBuf},{5,p}});
                    allDecodeDispatches.push_back({plQ8Matmul.pipeline,bg,1,(kvDimAct+Q8_TILE-1)/Q8_TILE,1,L+"q35_k"});
                }
            }
            // 4. V matmul
            {
                if (lwQ35.vSepKQ.handle) {
                    auto* kp=kqPipelineFor(lwQ35.vSepKQType);auto tile=kqTileFor(lwQ35.vSepKQType);
                    auto p=makeKQParams("p_kq_q35_v_"+std::to_string(i),cfg.nEmbd,kvDimAct,lwQ35.vSepKQNBlocks,lwQ35.vSepKQRowStride);
                    auto bg=makeBG(*kp,{{0,normOutBuf},{1,lwQ35.vSepKQ},{2,zeroBiasV},{3,q35VBuf},{4,p}});
                    allDecodeDispatches.push_back({kp->pipeline,bg,1,(kvDimAct+tile-1)/tile,1,L+"q35_v"});
                } else {
                    auto p=mkP32("p_q35_v_L"+std::to_string(i),{cfg.nEmbd,kvDimAct});
                    auto bg=makeBG(plQ8Matmul,{{0,normOutBuf},{1,lwQ35.vSepW},{2,lwQ35.vSepS},{3,zeroBiasV},{4,q35VBuf},{5,p}});
                    allDecodeDispatches.push_back({plQ8Matmul.pipeline,bg,1,(kvDimAct+Q8_TILE-1)/Q8_TILE,1,L+"q35_v"});
                }
            }
            // 5. attn_split_qg: q35QjBuf → q35QBuf + q35GateBuf
            {
                auto& plSplit = getKernel("split_qg");
                auto p = mkP32("p_q35_split_L"+std::to_string(i), {cfg.nHead, headDimAct});
                auto bg = makeBG(plSplit, {
                    {0, q35QjBuf}, {1, q35QBuf}, {2, q35GateBuf}, {3, p}});
                allDecodeDispatches.push_back({plSplit.pipeline, bg,
                    (qDimAct + 255) / 256, 1, 1, L+"q35_split"});
            }

            // llama.cpp order for qwen35 full-attention layers:
            // Q/G split -> Q norm, K norm -> MRoPE(Q/K) -> attention -> sigmoid gate.
            {
                uint32_t epsBits = 0;
                memcpy(&epsBits, &cfg.rmsNormEps, 4);
                auto& plHeadNorm = getKernel("head_rmsnorm");
                auto p = mkP32("p_q35_qnorm_L"+std::to_string(i),
                               {cfg.nHead, headDimAct, epsBits});
                auto bg = makeBG(plHeadNorm, {{0, q35QBuf}, {1, lwQ35.qNorm}, {2, p}});
                allDecodeDispatches.push_back({plHeadNorm.pipeline, bg,
                    cfg.nHead, 1, 1, L+"q35_qnorm"});
            }
            {
                uint32_t epsBits = 0;
                memcpy(&epsBits, &cfg.rmsNormEps, 4);
                auto& plHeadNorm = getKernel("head_rmsnorm");
                auto p = mkP32("p_q35_knorm_L"+std::to_string(i),
                               {cfg.nKvHeads, headDimAct, epsBits});
                auto bg = makeBG(plHeadNorm, {{0, q35KBuf}, {1, lwQ35.kNorm}, {2, p}});
                allDecodeDispatches.push_back({plHeadNorm.pipeline, bg,
                    cfg.nKvHeads, 1, 1, L+"q35_knorm"});
            }
            {
                auto& plQ35Rope = getKernel("qwen35_rope_q_to_qrot");
                auto bg = makeBG(plQ35Rope, {
                    {0, q35QBuf}, {1, qRotBuf}, {2, ropeCosBuf}, {3, ropeSinBuf},
                    {4, q35RopeQParamsBuf}});
                q35QRoPEDispatchIndices[i] = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plQ35Rope.pipeline, bg,
                    (headDimAct + 255) / 256, cfg.nHead, 1, L+"q35_q_mrope"});
            }
            {
                auto& plKvWrite = getKernel("qwen35_kv_cache_write_rope");
                auto bg = makeBG(plKvWrite, {
                    {0, q35KBuf}, {1, q35VBuf}, {2, kvCache[i].K}, {3, kvCache[i].V},
                    {4, ropeCosBuf}, {5, ropeSinBuf}, {6, q35RopeKParamsBuf},
                    {7, q35KvWriteParamsBuf}});
                q35KvWriteDispatchIndices[i] = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plKvWrite.pipeline, bg,
                    (headDimAct + 255) / 256, cfg.nKvHeads, 1, L+"q35_kv_write"});
            }
            {
                auto bg = makeBG(plChunkP1, {
                    {0, qRotBuf}, {1, kvCache[i].K}, {2, kvCache[i].V},
                    {3, attnPartialsBuf}, {4, chunkedAttnParamsBuf}});
                attnP1DispatchIndices[i] = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plChunkP1.pipeline, bg,
                    cfg.nHead, maxChunks, 1, L+"q35_attn_p1"});
            }
            {
                auto bg = makeBG(plChunkP2, {
                    {0, attnPartialsBuf}, {1, attnOutBuf},
                    {2, chunkedAttnParamsBuf}});
                attnP2DispatchIndices[i] = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plChunkP2.pipeline, bg,
                    cfg.nHead, 1, 1, L+"q35_attn_p2"});
            }
            // 9. attn_gated_output
            {
                auto& plGate = getKernel("gated_output");
                auto p = mkP32("p_q35_gate_L"+std::to_string(i), {qDimAct});
                auto bg = makeBG(plGate, {
                    {0, attnOutBuf}, {1, q35GateBuf}, {2, q35AttnOutBuf}, {3, p}});
                allDecodeDispatches.push_back({plGate.pipeline, bg,
                    (qDimAct + 255) / 256, 1, 1, L+"q35_gated"});
            }
            // 10. wo matmul to projOutBuf. The shared post-attention block below
            // performs the residual add and post-attention norm.
            if (lwQ35.oKQ.handle) {
                auto* kp=kqPipelineFor(lwQ35.oKQType);auto tile=kqTileFor(lwQ35.oKQType);
                auto p=makeKQParams("p_kq_q35_o_"+std::to_string(i),qDimAct,cfg.nEmbd,lwQ35.oKQNBlocks,lwQ35.oKQRowStride);
                auto bg = makeBG(*kp, {
                    {0, q35AttnOutBuf}, {1, lwQ35.oKQ}, {2, zeroBiasE},
                    {3, projOutBuf}, {4, p}});
                allDecodeDispatches.push_back({kp->pipeline, bg,
                    1, (cfg.nEmbd + tile - 1) / tile, 1, L+"q35_kq_oproj"});
            } else if (lwQ35.oW.handle) {
                auto p = mkP32("p_q35_oproj_L"+std::to_string(i), {qDimAct, cfg.nEmbd});
                auto bg = makeBG(plQ8Matmul, {
                    {0, q35AttnOutBuf}, {1, lwQ35.oW}, {2, lwQ35.oS},
                    {3, zeroBiasE}, {4, projOutBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q35_oproj"});
            }
        }

        // Standard attention is used only when the architecture-specific path
        // above did not claim this layer.
        if (!q35CustomAttn) {
        // Shared KV: attention reads from source layer's cache
        uint32_t kvCacheLayer = (pl.kvSourceLayer >= 0) ? (uint32_t)pl.kvSourceLayer : i;

        // 1. RMSNorm (only for KQ path — Q8 path uses fused q8_matmul_norm).
        // Must run EVERY layer: the K-quant qkv matmul below reads normOutBuf,
        // which is otherwise clobbered by each layer's pre-FFN norm. Restricting
        // this to layer 0 left layers 1+ projecting QKV from the previous layer's
        // FFN-normed activation instead of this layer's attention-normed input —
        // producing fluent but factually wrong output for K-quant Gemma models.
        if ((useKQ || weightsAreNativeQ4)) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.inputNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_norm"});
        }

        // 2. QKV matmul
        {
            if (lw.qOnly) {
                // Shared-KV layer: only a Q projection exists. Produce Q into
                // qkvBuf's Q region; K/V are reused from the source layer's
                // cache (kvCacheLayer). rope writes its K/V output to a scratch
                // buffer so the shared cache is not corrupted.
                if (weightsAreNativeQ4 && plQ4Decode) {
                    auto p = makeQ4Params("p_q4_qonly_" + std::to_string(i), cfg.nEmbd, plQDim);
                    auto bg = makeBG(*plQ4Decode, {
                        {0, normOutBuf}, {1, lw.qOnlyW}, {2, lw.qOnlyS},
                        {3, qkvBuf}, {4, p}});
                    di.qkv = (int)allDecodeDispatches.size();
                    allDecodeDispatches.push_back({plQ4Decode->pipeline, bg,
                        (plQDim + Q4_DECODE_TILE_N - 1) / Q4_DECODE_TILE_N, 1, 1, L+"q4_q_only"});
                } else {
                    vbg.qkvBase = makeBG(plQ8MatmulNorm, {
                        {0, xBuf}, {1, lw.qOnlyW}, {2, lw.qOnlyS},
                        {3, zeroBiasQKV}, {4, qkvBuf}, {5, layerQkvNP},
                        {6, lw.inputNorm}});
                    di.qkv = (int)allDecodeDispatches.size();
                    allDecodeDispatches.push_back({plQ8MatmulNorm.pipeline, vbg.qkvBase,
                        1, (plQDim + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_q_only"});
                }
            } else if (useKQ) {
                auto bg = makeBG(*plKQ, {
                    {0, normOutBuf}, {1, lw.qkvKQ}, {2, zeroBiasQKV},
                    {3, qkvBuf}, {4, kqQkvParams}});
                di.qkv = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plKQ->pipeline, bg,
                    1, (plQkvOut + 7) / 8, 1, L+"kq_qkv"});
            } else if (weightsAreNativeQ4 && plQ4Decode) {
                auto p = makeQ4Params("p_q4_qkv_" + std::to_string(i), cfg.nEmbd, plQkvOut);
                auto bg = makeBG(*plQ4Decode, {
                    {0, normOutBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                    {3, qkvBuf}, {4, p}});
                di.qkv = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plQ4Decode->pipeline, bg,
                    (plQkvOut + Q4_DECODE_TILE_N - 1) / Q4_DECODE_TILE_N, 1, 1, L+"q4_qkv"});
            } else {
                vbg.qkvBase = makeBG(plQ8MatmulNorm, {
                    {0, xBuf}, {1, lw.qkvW}, {2, lw.qkvS},
                    {3, zeroBiasQKV}, {4, qkvBuf}, {5, layerQkvNP},
                    {6, lw.inputNorm}});
                di.qkv = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plQ8MatmulNorm.pipeline, vbg.qkvBase,
                    1, (plQkvOut + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_qkv"});
            }
        }

        {
            bool isSwaLayer = hasSWA && i < cfg.layerAttnTypes.size() &&
                              cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            auto& ropeCos = (isSwaLayer && ropeCosBufSWA.handle)
                            ? ropeCosBufSWA : ropeCosBuf;
            auto& ropeSin = (isSwaLayer && ropeSinBufSWA.handle)
                            ? ropeSinBufSWA : ropeSinBuf;
            // Shared-KV (Q-only) layers: rope Q into qRotBuf but send the
            // kernel's K/V writes to scratch so the shared source cache is
            // preserved (attention below reads the real K/V from it).
            auto& ropeK = lw.qOnly && qOnlyScratchK.handle ? qOnlyScratchK : kvCache[kvCacheLayer].K;
            auto& ropeV = lw.qOnly && qOnlyScratchV.handle ? qOnlyScratchV : kvCache[kvCacheLayer].V;
            // Per-layer head-dim kernel + params for variable-head-dim models.
            uint32_t hdL = cfg.perLayer[i].headDim;
            auto& plRope = hasVariableHeadDim
                ? getKernelHD(cfg.hasQkNorm ? "fused_qknorm_rope" : "fused_rope", hdL)
                : plFusedRope;
            auto& ropeParamBuf = hasVariableHeadDim ? perLayerRopeParamBufs[i] : fusedRopeParamsBuf;
            auto bg = makeBG(plRope, {
                {0, qkvBuf}, {1, qRotBuf},
                {2, ropeK}, {3, ropeV},
                {4, ropeCos}, {5, ropeSin},
                {6, lw.qNorm}, {7, lw.kNorm},
                {8, ropeParamBuf}});
            ropeDispatchIndices[i] = (int)allDecodeDispatches.size();
            decodeUsesFusedRopeParams = true;
            allDecodeDispatches.push_back({plRope.pipeline, bg,
                cfg.nHead + cfg.nKvHeads, 1, 1, L+"fused_rope"});
        }

        {
            bool isSWA = hasSWA && i < cfg.layerAttnTypes.size() &&
                         cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            uint32_t hdL = cfg.perLayer[i].headDim;
            auto& attnParams = hasVariableHeadDim ? perLayerAttnParamBufs[i]
                             : (isSWA ? chunkedAttnParamsBufSWA : chunkedAttnParamsBuf);
            auto& plP1 = hasVariableHeadDim ? getKernelHD(chunkP1Kernel, hdL) : plChunkP1;
            auto bg = makeBG(plP1, {
                {0, qRotBuf}, {1, kvCache[kvCacheLayer].K}, {2, kvCache[kvCacheLayer].V},
                {3, attnPartialsBuf}, {4, attnParams}});
            attnP1DispatchIndices[i] = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({plP1.pipeline, bg,
                cfg.nHead, maxChunks, 1, L+"attn_p1"});
        }

        {
            bool isSWA2 = hasSWA && i < cfg.layerAttnTypes.size() &&
                          cfg.layerAttnTypes[i] == AttnLayerType::SlidingWindow;
            uint32_t hdL = cfg.perLayer[i].headDim;
            auto& attnParams2 = hasVariableHeadDim ? perLayerAttnParamBufs[i]
                              : (isSWA2 ? chunkedAttnParamsBufSWA : chunkedAttnParamsBuf);
            auto& plP2 = hasVariableHeadDim ? getKernelHD("gqa_chunked_pass2", hdL) : plChunkP2;
            auto bg = makeBG(plP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, attnParams2}});
            attnP2DispatchIndices[i] = (int)allDecodeDispatches.size();
            allDecodeDispatches.push_back({plP2.pipeline, bg,
                cfg.nHead, 1, 1, L+"attn_p2"});
        }

        {
            if (useKQ) {
                auto bg = makeBG(*plKQ, {
                    {0, attnOutBuf}, {1, lw.oKQ}, {2, zeroBiasE},
                    {3, projOutBuf}, {4, kqOprojParams}});
                di.oproj = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plKQ->pipeline, bg,
                    1, (cfg.nEmbd + kqTileN - 1) / kqTileN, 1, L+"kq_oproj"});
            } else if (weightsAreNativeQ4 && plQ4Decode) {
                auto p = makeQ4Params("p_q4_oproj_" + std::to_string(i), plQDim, cfg.nEmbd);
                auto bg = makeBG(*plQ4Decode, {
                    {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                    {3, projOutBuf}, {4, p}});
                di.oproj = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plQ4Decode->pipeline, bg,
                    (cfg.nEmbd + Q4_DECODE_TILE_N - 1) / Q4_DECODE_TILE_N, 1, 1, L+"q4_oproj"});
            } else {
                vbg.oprojBase = makeBG(plQ8Matmul, {
                    {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                    {3, zeroBiasE}, {4, projOutBuf}, {5, layerOpP}});
                if (decodeFastVariantsAvailable) {
                    vbg.oprojFast = makeBG(plQ8Fast, {
                        {0, attnOutBuf}, {1, lw.oW}, {2, lw.oS},
                        {3, zeroBiasE}, {4, projOutBuf}, {5, layerOpP}});
                }
                auto bg = tuning.decodeUseFastOproj && vbg.oprojFast ? vbg.oprojFast : vbg.oprojBase;
                auto pipeline = tuning.decodeUseFastOproj && vbg.oprojFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
                di.oproj = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({pipeline, bg,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_oproj"});
            }
        }
        }

        // Post-attention:
        // - Qwen3.5: residual add first, then attn_post_norm feeds the FFN.
        // - Gemma sandwich: normalize attention output before residual add.
        // - Standard: residual add plus ffn_norm in one fused dispatch.
        if (cfg.arch == "qwen35" && lw.postNorm.handle) {
            const bool fuseGateupQ8 = isAmdAdapter && useQ4KDp4a &&
                lw.guKQ.handle && lw.guKQType == GGUF_TYPE_Q4_K;
            if (fuseGateupQ8) {
                static const char* ADD_RMS_Q8_WGSL = R"(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read_write> X:array<f32>;
@group(0) @binding(1) var<storage,read> R:array<f32>;
@group(0) @binding(2) var<storage,read_write> Y:array<f32>;
@group(0) @binding(3) var<storage,read> W:array<f32>;
@group(0) @binding(4) var<storage,read_write> Rstd:array<f32>;
@group(0) @binding(5) var<storage,read_write> XQ:array<u32>;
@group(0) @binding(6) var<storage,read_write> XS:array<f32>;
struct Params{stride:i32,N:i32,eps:f32};
@group(0) @binding(7) var<storage,read> P:Params;
var<workgroup> sums:array<f32,8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane:u32,
        @builtin(subgroup_size) sg_size:u32){
 let tid=lid.x;let N=u32(P.N);var ss=0.0;
 for(var k=tid;k<N;k+=256u){let v=X[k]+R[k];X[k]=v;ss+=v*v;}
 let part=subgroupAdd(ss);let sg=tid/sg_size;
 if(sg_lane==0u){sums[sg]=part;}workgroupBarrier();
 if(tid==0u){var total=0.0;let nsg=(256u+sg_size-1u)/sg_size;
  for(var s=0u;s<nsg;s++){total+=sums[s];}
  let r=inverseSqrt(total/f32(N)+P.eps);sums[0]=r;Rstd[0]=r;
 }workgroupBarrier();let rstd=sums[0];
 for(var base=0u;base<N;base+=256u){
  let k=base+tid;var y=0.0;if(k<N){y=(X[k]*rstd)*W[k];Y[k]=y;}
  var amax=abs(y);amax=max(amax,subgroupShuffleXor(amax,16u));
  amax=max(amax,subgroupShuffleXor(amax,8u));amax=max(amax,subgroupShuffleXor(amax,4u));
  amax=max(amax,subgroupShuffleXor(amax,2u));amax=max(amax,subgroupShuffleXor(amax,1u));
  let scale=amax/127.0;let lane=tid&31u;let block32=tid/32u;
  if(lane==0u){XS[base/32u+block32]=scale;}
  let safe=select(1.0,scale,scale!=0.0);let qi=clamp(i32(round(y/safe)),-127,127);
  let pack_lane=lane&3u;let pack_group=lane/4u;
  var packed=u32(qi&255)<<(pack_lane*8u);packed|=subgroupShuffleXor(packed,1u);
  packed|=subgroupShuffleXor(packed,2u);
  if(pack_lane==0u){XQ[base/4u+block32*8u+pack_group]=packed;}
 }
}
)";
                auto& plAddRmsQ8 = gpu->getOrCreatePipeline(
                    "q35_add_rms_q8_amd", ADD_RMS_Q8_WGSL, 8);
                auto bg = makeBG(plAddRmsQ8, {
                    {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                    {3, lw.postNorm}, {4, rstdBuf}, {5, kqActQ8Buf},
                    {6, kqActScaleBuf}, {7, rmsParams}});
                allDecodeDispatches.push_back({plAddRmsQ8.pipeline, bg,
                    1, 1, 1, L+"q35_add_attn_post_norm_q8"});
            } else {
                auto bg = makeBG(plAddRmsNorm, {
                    {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                    {3, lw.postNorm}, {4, rstdBuf}, {5, rmsParams}});
                allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg,
                    1, 1, 1, L+"q35_add_attn_post_norm"});
            }
        } else if (cfg.hasSandwichNorm && lw.postNorm.handle) {
            if (weightsAreNativeQ4 && !std::getenv("BP_UNFUSED_SANDWICH")) {
                static const char* FUSED_SANDWICH_ATTN_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Attn: array<f32>;
@group(0) @binding(2) var<storage, read> PostW: array<f32>;
@group(0) @binding(3) var<storage, read> FfnW: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let N = P[0]; let eps = bitcast<f32>(P[1]);
    var ss = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = Attn[j]; ss += v * v; }
    sums[tid] = ss;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { sums[tid] += sums[tid + stride]; }
        workgroupBarrier();
    }
    let postRms = inverseSqrt(sums[0] / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) {
        X[j] += Attn[j] * postRms * PostW[j];
    }
    storageBarrier();
    workgroupBarrier();
    ss = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; ss += v * v; }
    sums[tid] = ss;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { sums[tid] += sums[tid + stride]; }
        workgroupBarrier();
    }
    let ffnRms = inverseSqrt(sums[0] / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { Out[j] = X[j] * ffnRms * FfnW[j]; }
}
)";
                auto& plFusedSandwich = gpu->getOrCreatePipeline(
                    "fused_sandwich_attn", std::string(FUSED_SANDWICH_ATTN_WGSL), 6);
                GPUBuffer fusedP;
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                memcpy(&data[1], &cfg.rmsNormEps, 4);
                fusedP = gpu->createBuffer("p_fused_sandwich_" + std::to_string(i), 16);
                gpu->writeBuffer(fusedP, data, 16);
                auto bg = makeBG(plFusedSandwich, {
                    {0, xBuf}, {1, projOutBuf}, {2, lw.postNorm},
                    {3, lw.ffnNorm}, {4, normOutBuf}, {5, fusedP}});
                allDecodeDispatches.push_back({plFusedSandwich.pipeline, bg,
                    1, 1, 1, L+"fused_post_add_ffn_norm"});
            } else {
            // Sandwich: RMSNorm(projOutBuf) in-place, then xBuf += projOutBuf, then norm for FFN
            // Pure workgroup shared-memory tree reduction — no subgroup ops.
            // The subgroup+workgroup-shared variant crashed the D3D12 driver
            // (DXGI_ERROR_DEVICE_REMOVED at the first pipeline that uses it)
            // when compiled for Gemma 4's sandwich norms.
            static const char* RMS_NORM_INPLACE_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
var<workgroup> wg_sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0];
    let eps = bitcast<f32>(_p_[1]);
    let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var i = tid; i < N; i += 256u) { let v = X[i]; sum_sq += v * v; }
    wg_sums[tid] = sum_sq;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) { wg_sums[tid] = wg_sums[tid] + wg_sums[tid + stride]; }
        workgroupBarrier();
    }
    let total = wg_sums[0];
    let rms = 1.0 / sqrt(total / f32(N) + eps);
    for (var i = tid; i < N; i += 256u) { X[i] = X[i] * rms * W[i]; }
}
)";
            auto& plRmsIP = gpu->getOrCreatePipeline("rms_norm_inplace",
                std::string(RMS_NORM_INPLACE_WGSL), 3);
            GPUBuffer rmsIPParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                rmsIPParams = gpu->createBuffer("p_rmsip_post_" + std::to_string(i), 16);
                gpu->writeBuffer(rmsIPParams, data, 16);
            }
            // Post-attention sandwich norm (in-place on projOutBuf)
            auto bgPostNorm = makeBG(plRmsIP, {
                {0, projOutBuf}, {1, lw.postNorm}, {2, rmsIPParams}});
            allDecodeDispatches.push_back({plRmsIP.pipeline, bgPostNorm, 1, 1, 1, L+"post_norm"});

            // Residual add: xBuf += projOutBuf
            static const char* ADD_INPLACE_WGSL2 = R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0];
    let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)";
            auto& plAddIP = gpu->getOrCreatePipeline("add_inplace_sw",
                std::string(ADD_INPLACE_WGSL2), 3);
            GPUBuffer addParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                addParams = gpu->createBuffer("p_add_post_" + std::to_string(i), 16);
                gpu->writeBuffer(addParams, data, 16);
            }
            auto bgAdd = makeBG(plAddIP, {
                {0, xBuf}, {1, projOutBuf}, {2, addParams}});
            allDecodeDispatches.push_back({plAddIP.pipeline, bgAdd,
                (cfg.nEmbd + 255) / 256, 1, 1, L+"add_res1"});

            // Pre-FFN RMSNorm (using ffnNorm)
            auto bgFfnNorm = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, lw.ffnNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bgFfnNorm, 1, 1, 1, L+"ffn_norm"});
            }
        } else {
            // Standard: fused residual-add + pre-FFN norm
            auto bg = makeBG(plAddRmsNorm, {
                {0, xBuf}, {1, projOutBuf}, {2, normOutBuf},
                {3, lw.postAttnNorm}, {4, rstdBuf}, {5, rmsParams}});
            allDecodeDispatches.push_back({plAddRmsNorm.pipeline, bg, 1, 1, 1, L+"add_rms"});
        }

        // ── MoE FFN dispatch (Phase 3d) ─────────────────────────────────────
        // Sequence per MoE layer per decode step:
        //   router_matmul  → moeRouterOutBuf
        //   topk_softmax   → moeIndicesBuf, moeWeightsBuf
        //   compute_offsets → moeGateOffsets, moeUpOffsets, moeDownOffsets
        //   for k in 0..numExpertsPerTok:
        //     iq2s_matmul_moe(slot=k, gate_offsets) → moeRoutedGateBuf
        //     iq2s_matmul_moe(slot=k, up_offsets)   → moeRoutedUpBuf
        //     silu_mul(gate,up)                     → moeRoutedActBuf
        //     iq3s_matmul_moe(slot=k, down_offsets) → moeExpertOutBuf
        //     weighted_accumulate(slot=k)           : xBuf += w[k] * moeExpertOutBuf
        //   shared_expert: 4 Q8 dispatches + add into xBuf
        if (cfg.numExperts > 0) {
            // ── per-layer buffers we need that aren't in member state ──
            // These persist for the life of this build (each layer gets its own).
            uint32_t IMe = cfg.moeIntermediateSize;
            uint32_t IMs = cfg.moeSharedIntermediateSize;
            uint32_t topk = cfg.numExpertsPerTok;

            // 3 per-direction offset buffers (k slots each, u32 each = 4*k bytes)
            GPUBuffer gateOffsets = gpu->createBuffer("moe_gate_off_L"+std::to_string(i), topk*4);
            GPUBuffer upOffsets   = gpu->createBuffer("moe_up_off_L"+std::to_string(i),   topk*4);
            GPUBuffer downOffsets = gpu->createBuffer("moe_down_off_L"+std::to_string(i), topk*4);

            // Param buffers (16 B each)
            auto mkP = [&](const std::string& name, std::initializer_list<uint32_t> data) -> GPUBuffer {
                uint32_t buf[8] = {0};
                size_t i2 = 0; for (uint32_t v : data) { buf[i2++] = v; }
                auto b = gpu->createBuffer(name, 32);
                gpu->writeBuffer(b, buf, 32);
                return b;
            };

            // 1. Router matmul (Q8): normOutBuf @ routerW → moeRouterOutBuf
            {
                auto p = mkP("p_router_L"+std::to_string(i), {cfg.nEmbd, cfg.numExperts});
                auto bg = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.routerW}, {2, lw.routerS},
                    {3, zeroBiasV}, {4, moeRouterOutBuf}, {5, p}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                    1, (cfg.numExperts + Q8_TILE - 1) / Q8_TILE, 1, L+"moe_router"});
            }

            // 2. topk_softmax: moeRouterOutBuf → moeIndicesBuf, moeWeightsBuf
            {
                auto& plTopK = getKernel("moe_topk_softmax");
                auto p = mkP("p_topk_L"+std::to_string(i), {cfg.numExperts, topk, 1u});
                auto bg = makeBG(plTopK, {
                    {0, moeRouterOutBuf}, {1, moeIndicesBuf}, {2, moeWeightsBuf}, {3, p}});
                allDecodeDispatches.push_back({plTopK.pipeline, bg, 1, 1, 1, L+"moe_topk"});
            }

            // 3. compute_offsets: moeIndicesBuf → gate/up/down offsets
            {
                auto& plCO = getKernel("moe_compute_offsets");
                auto p = mkP("p_co_L"+std::to_string(i), {topk, IMe, cfg.nEmbd});
                auto bg = makeBG(plCO, {
                    {0, moeIndicesBuf}, {1, gateOffsets}, {2, upOffsets}, {3, downOffsets}, {4, p}});
                allDecodeDispatches.push_back({plCO.pipeline, bg, 1, 1, 1, L+"moe_co"});
            }

            // 4. Per-expert loop (8 slots)
            // For each slot, dispatch: gate (IQ2_S), up (IQ2_S), silu_mul, down, accumulate.
            // We assume gate/up are IQ2_S and down is IQ3_S — most layers in this model.
            // (TODO: per-layer down quant detection; this hardcodes IQ3_S which is wrong
            //  for layers 34, 38, 39 where down is IQ4_XS.)
            auto& plIQ2Moe = getKernel("iq2s_matmul_moe");
            auto& plIQ3Moe = getKernel("iq3s_matmul_moe");
            auto& plIQ4XSMoe = getKernel("iq4xs_matmul_moe");
            auto& plSiluMul = getKernel("silu_mul_fused");  // existing fused kernel
            auto& plWAcc = getKernel("moe_weighted_accumulate_decode");

            // Compute n_blocks + row_stride for the fused expert buffers.
            // For IQ2_S blocks of 82 bytes padded to 84 = 21 words; rowStride = 21*nBlocks.
            // K-dim of gate/up = nEmbd = 2048; K/QK_K = 8 blocks per row → rowStride=168 words.
            uint32_t iq2_nBlocks_e   = cfg.nEmbd / 256u;            // K dim is nEmbd for gate/up
            uint32_t iq2_rowStride_e = iq2_nBlocks_e * 21u;
            uint32_t iq3_nBlocks_im  = IMe / 256u;                  // K dim is IMe for down
            uint32_t iq3_rowStride_im= iq3_nBlocks_im * 28u;
            uint32_t iq4xs_rowStride_im = iq3_nBlocks_im * 34u;
            // Pick down dispatch based on this layer's recorded quant type.
            bool downIsIQ4XS = (moeExpertsDownType.size() > i && moeExpertsDownType[i] == (uint32_t)GGUF_TYPE_IQ4_XS);
            auto& plDownMoe = downIsIQ4XS ? plIQ4XSMoe : plIQ3Moe;
            uint32_t downRowStride = downIsIQ4XS ? iq4xs_rowStride_im : iq3_rowStride_im;

            for (uint32_t k = 0; k < topk; k++) {
                std::string ks = std::to_string(k);
                // 4a. gate: normOutBuf @ expertsGateW(slot=k) → moeRoutedGateBuf
                {
                    auto p = mkP("p_gate_L"+std::to_string(i)+"_s"+ks,
                                 {cfg.nEmbd, IMe, iq2_nBlocks_e, iq2_rowStride_e, 0u, k});
                    auto bg = makeBG(plIQ2Moe, {
                        {0, normOutBuf}, {1, lw.expertsGateW}, {2, iq2sCodebookBuf},
                        {3, zeroBiasGU}, {4, moeRoutedGateBuf}, {5, gateOffsets}, {6, p}});
                    allDecodeDispatches.push_back({plIQ2Moe.pipeline, bg,
                        1, (IMe + 8u - 1u) / 8u, 1, L+"moe_gate_s"+ks});
                }
                // 4b. up: → moeRoutedUpBuf
                {
                    auto p = mkP("p_up_L"+std::to_string(i)+"_s"+ks,
                                 {cfg.nEmbd, IMe, iq2_nBlocks_e, iq2_rowStride_e, 0u, k});
                    auto bg = makeBG(plIQ2Moe, {
                        {0, normOutBuf}, {1, lw.expertsUpW}, {2, iq2sCodebookBuf},
                        {3, zeroBiasGU}, {4, moeRoutedUpBuf}, {5, upOffsets}, {6, p}});
                    allDecodeDispatches.push_back({plIQ2Moe.pipeline, bg,
                        1, (IMe + 8u - 1u) / 8u, 1, L+"moe_up_s"+ks});
                }
                // 4c. silu_mul(gate, up) → moeRoutedActBuf
                {
                    auto p = mkP("p_silumul_L"+std::to_string(i)+"_s"+ks, {IMe});
                    // silu_mul_fused expects [gate, up] packed in gateUpBuf typically;
                    // for MoE we pass them as separate inputs. The kernel signature
                    // may differ — using shared_silu + shared_mul as fallback.
                    auto& plSilu = getKernel("shared_silu");
                    auto& plMul  = getKernel("shared_mul");
                    auto bgSilu = makeBG(plSilu,
                        {{0, moeRoutedGateBuf}, {1, moeRoutedGateBuf}, {2, p}});
                    allDecodeDispatches.push_back({plSilu.pipeline, bgSilu,
                        (IMe + 255) / 256, 1, 1, L+"moe_silu_s"+ks});
                    auto bgMul = makeBG(plMul,
                        {{0, moeRoutedGateBuf}, {1, moeRoutedUpBuf}, {2, moeRoutedActBuf}, {3, p}});
                    allDecodeDispatches.push_back({plMul.pipeline, bgMul,
                        (IMe + 255) / 256, 1, 1, L+"moe_mul_s"+ks});
                }
                // 4d. down: moeRoutedActBuf @ expertsDownW(slot=k) → moeExpertOutBuf
                {
                    auto p = mkP("p_down_L"+std::to_string(i)+"_s"+ks,
                                 {IMe, cfg.nEmbd, iq3_nBlocks_im, downRowStride, 0u, k});
                    // IQ3_S kernel needs codebook binding; IQ4_XS does not.
                    GPUBuffer codebookBuf = downIsIQ4XS ? zeroBiasE : iq3sCodebookBuf; // codebook irrelevant for iq4xs
                    if (downIsIQ4XS) {
                        auto bg = makeBG(plDownMoe, {
                            {0, moeRoutedActBuf}, {1, lw.expertsDownW},
                            {2, zeroBiasE}, {3, moeExpertOutBuf}, {4, downOffsets}, {5, p}});
                        allDecodeDispatches.push_back({plDownMoe.pipeline, bg,
                            1, (cfg.nEmbd + 8u - 1u) / 8u, 1, L+"moe_down_iq4xs_s"+ks});
                    } else {
                        auto bg = makeBG(plDownMoe, {
                            {0, moeRoutedActBuf}, {1, lw.expertsDownW}, {2, iq3sCodebookBuf},
                            {3, zeroBiasE}, {4, moeExpertOutBuf}, {5, downOffsets}, {6, p}});
                        allDecodeDispatches.push_back({plDownMoe.pipeline, bg,
                            1, (cfg.nEmbd + 8u - 1u) / 8u, 1, L+"moe_down_iq3s_s"+ks});
                    }
                }
                // 4e. weighted_accumulate: xBuf += weights[k] * moeExpertOutBuf
                {
                    auto p = mkP("p_wacc_L"+std::to_string(i)+"_s"+ks, {cfg.nEmbd, k});
                    auto bg = makeBG(plWAcc,
                        {{0, xBuf}, {1, moeExpertOutBuf}, {2, moeWeightsBuf}, {3, p}});
                    allDecodeDispatches.push_back({plWAcc.pipeline, bg,
                        (cfg.nEmbd + 255) / 256, 1, 1, L+"moe_wacc_s"+ks});
                }
            }

            // 5. Shared expert (Q8 dense)
            {
                auto pGate = mkP("p_sh_gate_L"+std::to_string(i), {cfg.nEmbd, IMs});
                auto bgGate = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.shexpGateW}, {2, lw.shexpGateS},
                    {3, zeroBiasGU}, {4, moeShexpGateUpBuf}, {5, pGate}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgGate,
                    1, (IMs + Q8_TILE - 1) / Q8_TILE, 1, L+"shexp_gate"});

                // up writes to moeShexpGateUpBuf offset IMs — but our buffer alloc is 2*IMs.
                // For simplicity we use a separate small slice via creating another buffer.
                // (Could pack via y_offset param; using moeShexpActBuf as up temp for now.)
                auto pUp = mkP("p_sh_up_L"+std::to_string(i), {cfg.nEmbd, IMs});
                auto bgUp = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.shexpUpW}, {2, lw.shexpUpS},
                    {3, zeroBiasGU}, {4, moeShexpActBuf}, {5, pUp}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgUp,
                    1, (IMs + Q8_TILE - 1) / Q8_TILE, 1, L+"shexp_up"});

                // silu(gate) * up → gate buf (in-place)
                auto pSiluP = mkP("p_sh_silu_L"+std::to_string(i), {IMs});
                auto& plSilu = getKernel("shared_silu");
                auto& plMul  = getKernel("shared_mul");
                auto bgSilu = makeBG(plSilu,
                    {{0, moeShexpGateUpBuf}, {1, moeShexpGateUpBuf}, {2, pSiluP}});
                allDecodeDispatches.push_back({plSilu.pipeline, bgSilu,
                    (IMs + 255) / 256, 1, 1, L+"shexp_silu"});
                auto bgMul = makeBG(plMul,
                    {{0, moeShexpGateUpBuf}, {1, moeShexpActBuf},
                     {2, moeShexpGateUpBuf}, {3, pSiluP}});
                allDecodeDispatches.push_back({plMul.pipeline, bgMul,
                    (IMs + 255) / 256, 1, 1, L+"shexp_mul"});

                // down: moeShexpGateUpBuf @ shexpDownW → moeExpertOutBuf, then add to xBuf
                auto pDn = mkP("p_sh_dn_L"+std::to_string(i), {IMs, cfg.nEmbd});
                auto bgDn = makeBG(plQ8Matmul, {
                    {0, moeShexpGateUpBuf}, {1, lw.shexpDownW}, {2, lw.shexpDownS},
                    {3, zeroBiasE}, {4, moeExpertOutBuf}, {5, pDn}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgDn,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"shexp_dn"});

                // residual add: xBuf += moeExpertOutBuf (use a constant-weight accumulator)
                auto pAdd = mkP("p_sh_add_L"+std::to_string(i), {cfg.nEmbd, 0u});
                // Reuse plWAcc but with a "weights" buffer that's 1.0 at index 0.
                // For simplicity create a tiny [1] buffer with value 1.0.
                GPUBuffer one = gpu->createBuffer("one_L"+std::to_string(i), 4);
                float oneF = 1.0f;
                gpu->writeBuffer(one, &oneF, 4);
                auto bgAdd = makeBG(plWAcc,
                    {{0, xBuf}, {1, moeExpertOutBuf}, {2, one}, {3, pAdd}});
                allDecodeDispatches.push_back({plWAcc.pipeline, bgAdd,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"shexp_add"});
            }

            continue;  // skip the dense FFN block below
        }

        {
            if (lw.guKQ.handle) {
                auto* layerKQ = kqPipelineFor(lw.guKQType);
                uint32_t layerTile = kqTileFor(lw.guKQType);
                auto layerParams = makeKQParams("p_kq_gu_" + std::to_string(i),
                    cfg.nEmbd, 2 * plIM, lw.guKQNBlocks, lw.guKQRowStride);
                const bool prequantQ4K = useQ4KDp4a && lw.guKQType == GGUF_TYPE_Q4_K;
                if (prequantQ4K) {
                    const bool packedByPostNorm = cfg.arch == "qwen35" &&
                        isAmdAdapter && lw.postNorm.handle;
                    if (!packedByPostNorm) {
                        auto& plQuant = getKernel("q8_quantize_dp4a");
                        auto bgQuant = makeBG(plQuant, {
                            {0, normOutBuf}, {1, kqActQ8Buf},
                            {2, kqActScaleBuf}, {3, layerParams}});
                        allDecodeDispatches.push_back({plQuant.pipeline, bgQuant,
                            (cfg.nEmbd + 255) / 256, 1, 1, L+"kq_gateup_quant"});
                    }
                    const bool useIntelReduc16 = isIntelAdapter && cfg.arch == "qwen35";
                    layerKQ = &getKernel(useIntelReduc16
                        ? "q4k_matmul_prequant_dp4a_reduc16"
                        : "q4k_matmul_prequant_dp4a");
                    layerTile = useIntelReduc16 ? 16 : 8;
                }
                auto bg = prequantQ4K
                    ? makeBG(*layerKQ, {{0, kqActQ8Buf}, {1, kqActScaleBuf},
                        {2, lw.guKQ}, {3, zeroBiasGU}, {4, gateUpBuf}, {5, layerParams}})
                    : makeBG(*layerKQ, {{0, normOutBuf}, {1, lw.guKQ},
                        {2, zeroBiasGU}, {3, gateUpBuf}, {4, layerParams}});
                di.gateup = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({layerKQ->pipeline, bg,
                    1, (2 * plIM + layerTile - 1) / layerTile, 1, L+"kq_gateup"});
            } else if (weightsAreNativeQ4 && plQ4Gateup) {
                auto p = makeQ4Params("p_q4_gateup_" + std::to_string(i), cfg.nEmbd, 2 * plIM);
                auto bg = makeBG(*plQ4Gateup, {
                    {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                    {3, gateUpBuf}, {4, p}});
                di.gateup = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({plQ4Gateup->pipeline, bg,
                    (2 * plIM + Q4_GATEUP_TILE_N - 1) / Q4_GATEUP_TILE_N, 1, 1, L+"q4_gateup"});
            } else {
                vbg.gateupBase = makeBG(plQ8Matmul, {
                    {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                    {3, zeroBiasGU}, {4, gateUpBuf}, {5, layerGuP}});
                if (decodeFastVariantsAvailable) {
                    vbg.gateupFast = makeBG(plQ8Fast, {
                        {0, normOutBuf}, {1, lw.guW}, {2, lw.guS},
                        {3, zeroBiasGU}, {4, gateUpBuf}, {5, layerGuP}});
                }
                auto bg = tuning.decodeUseFastGateup && vbg.gateupFast ? vbg.gateupFast : vbg.gateupBase;
                auto pipeline = tuning.decodeUseFastGateup && vbg.gateupFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
                di.gateup = (int)allDecodeDispatches.size();
                allDecodeDispatches.push_back({pipeline, bg,
                    1, (2 * plIM + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_gateup"});
            }
        }

        // 9. Down projection + SiLU + residual add
        {
            if (lw.dnKQ.handle) {
                // K-quant: activation+mul → K-quant matmul → add residual (3 dispatches)
                auto* layerKQ = kqPipelineFor(lw.dnKQType);
                uint32_t layerTile = kqTileFor(lw.dnKQType);
                auto layerParams = makeKQParams("p_kq_dn_" + std::to_string(i),
                    plIM, cfg.nEmbd, lw.dnKQNBlocks, lw.dnKQRowStride);
                auto& plActMul = useGelu ? getKernel("gelu_mul_fused")
                                          : getKernel("silu_mul_fused");
                GPUBuffer siluParams;
                {
                    uint32_t data[4] = {cfg.intermediateSize, 0, 0, 0};
                    siluParams = gpu->createBuffer("p_silu_" + std::to_string(i), 16);
                    gpu->writeBuffer(siluParams, data, 16);
                }
                auto bgSilu = makeBG(plActMul, {
                    {0, gateUpBuf}, {1, siluMulOutBuf}, {2, siluParams}});
                allDecodeDispatches.push_back({plActMul.pipeline, bgSilu,
                    (cfg.intermediateSize + 127) / 128, 1, 1, L+"act_mul"});

                const bool prequantQ4K = useQ4KDp4a && lw.dnKQType == GGUF_TYPE_Q4_K;
                if (prequantQ4K) {
                    auto& plQuant = getKernel("q8_quantize_dp4a");
                    auto bgQuant = makeBG(plQuant, {
                        {0, siluMulOutBuf}, {1, kqActQ8Buf},
                        {2, kqActScaleBuf}, {3, layerParams}});
                    allDecodeDispatches.push_back({plQuant.pipeline, bgQuant,
                        (plIM + 255) / 256, 1, 1, L+"kq_down_quant"});
                    const bool useIntelReduc16 = isIntelAdapter && cfg.arch == "qwen35";
                    layerKQ = &getKernel(useIntelReduc16
                        ? "q4k_matmul_prequant_dp4a_reduc16"
                        : "q4k_matmul_prequant_dp4a");
                    layerTile = useIntelReduc16 ? 16 : 8;
                }
                auto bgDn = prequantQ4K
                    ? makeBG(*layerKQ, {{0, kqActQ8Buf}, {1, kqActScaleBuf},
                        {2, lw.dnKQ}, {3, zeroBiasE}, {4, projOutBuf}, {5, layerParams}})
                    : makeBG(*layerKQ, {{0, siluMulOutBuf}, {1, lw.dnKQ},
                        {2, zeroBiasE}, {3, projOutBuf}, {4, layerParams}});
                allDecodeDispatches.push_back({layerKQ->pipeline, bgDn,
                    1, (cfg.nEmbd + layerTile - 1) / layerTile, 1, L+"kq_down"});

                // Add residual: xBuf[i] += projOutBuf[i] using inline add_inplace kernel
                static const char* ADD_INPLACE_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0];
    let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)";
                auto& plAddIP = gpu->getOrCreatePipeline("add_inplace",
                    std::string(ADD_INPLACE_WGSL), 3);
                GPUBuffer addIPParams;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    addIPParams = gpu->createBuffer("p_addip_" + std::to_string(i), 16);
                    gpu->writeBuffer(addIPParams, data, 16);
                }
                // Post-FFN sandwich norm (before residual add)
                if (cfg.hasSandwichNorm && lw.postFfwNorm.handle) {
                    auto& plRmsIP2 = gpu->getOrCreatePipeline("rms_norm_inplace",
                        std::string(R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
var<workgroup> wg_sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0]; let eps = bitcast<f32>(_p_[1]); let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; sum_sq += v * v; }
    wg_sums[tid] = sum_sq;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { wg_sums[tid] += wg_sums[tid + stride]; }
        workgroupBarrier();
    }
    let rms = 1.0 / sqrt(wg_sums[0] / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { X[j] = X[j] * rms * W[j]; }
}
)"), 3);
                    GPUBuffer rmsP;
                    {
                        uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                        float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                        rmsP = gpu->createBuffer("p_rmsip_ffw_" + std::to_string(i), 16);
                        gpu->writeBuffer(rmsP, data, 16);
                    }
                    auto bgFfwNorm = makeBG(plRmsIP2, {
                        {0, projOutBuf}, {1, lw.postFfwNorm}, {2, rmsP}});
                    allDecodeDispatches.push_back({plRmsIP2.pipeline, bgFfwNorm,
                        1, 1, 1, L+"post_ffw_norm"});
                }

                auto bgAdd = makeBG(plAddIP, {
                    {0, xBuf}, {1, projOutBuf}, {2, addIPParams}});
                allDecodeDispatches.push_back({plAddIP.pipeline, bgAdd,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"residual_add"});
            } else if (cfg.hasSandwichNorm && lw.postFfwNorm.handle) {
                // Sandwich norm Q8 path: split into act → matmul → norm → add
                // 1. Activation: gateUpBuf → siluMulOutBuf (reuse K-quant temp buffer)
                if (!siluMulOutBuf.handle)
                    siluMulOutBuf = gpu->createBuffer("silu_mul_out", maxIMBuf * 4);
                siluMulDebugBuf = siluMulOutBuf;
                auto& plActMulSW = useGelu ? getKernel("gelu_mul_fused")
                                            : getKernel("silu_mul_fused");
                GPUBuffer actParams;
                {
                    uint32_t data[4] = {plIM, 0, 0, 0};
                    actParams = gpu->createBuffer("p_act_sw_" + std::to_string(i), 16);
                    gpu->writeBuffer(actParams, data, 16);
                }
                auto bgAct = makeBG(plActMulSW, {
                    {0, gateUpBuf}, {1, siluMulOutBuf}, {2, actParams}});
                allDecodeDispatches.push_back({plActMulSW.pipeline, bgAct,
                    (plIM + 127) / 128, 1, 1, L+"act_mul_sw"});

                // 2. Down matmul: siluMulOutBuf → projOutBuf (no residual add).
                // Use a dedicated STORAGE params buffer: plQ8Matmul declares its
                // params binding as var<storage>, but layerDnP / perLayerDnSiluParams
                // is a UNIFORM buffer (for q8_down_silu_add). Binding a uniform
                // buffer to a storage slot builds an invalid D3D12 descriptor and
                // removes the device (DXGI_ERROR_DEVICE_REMOVED).
                if (weightsAreNativeQ4 && plQ4Decode) {
                    auto p = makeQ4Params("p_q4_down_" + std::to_string(i), plIM, cfg.nEmbd);
                    if (plQ4PrequantDown) {
                        auto qp = makeQ8Params("p_q4_down_quant_" + std::to_string(i), plIM, 0);
                        auto& plQuant = getKernel("q8_quantize_dp4a");
                        auto bgQuant = makeBG(plQuant, {
                            {0, siluMulOutBuf}, {1, kqActQ8Buf},
                            {2, kqActScaleBuf}, {3, qp}});
                        allDecodeDispatches.push_back({plQuant.pipeline, bgQuant,
                            (plIM + 255) / 256, 1, 1, L+"q4_down_quant"});
                        auto bgDn = makeBG(*plQ4PrequantDown, {
                            {0, kqActQ8Buf}, {1, kqActScaleBuf},
                            {2, lw.dnW}, {3, lw.dnS}, {4, projOutBuf}, {5, p}});
                        allDecodeDispatches.push_back({plQ4PrequantDown->pipeline, bgDn,
                            (cfg.nEmbd + 7) / 8, 1, 1, L+"q4_down_sw"});
                    } else {
                        auto bgDn = makeBG(*plQ4Decode, {
                            {0, siluMulOutBuf}, {1, lw.dnW}, {2, lw.dnS},
                            {3, projOutBuf}, {4, p}});
                        allDecodeDispatches.push_back({plQ4Decode->pipeline, bgDn,
                            (cfg.nEmbd + Q4_DECODE_TILE_N - 1) / Q4_DECODE_TILE_N, 1, 1, L+"q4_down_sw"});
                    }
                } else {
                    auto dnMatmulP = makeQ8Params("p_dn_sw_" + std::to_string(i), plIM, cfg.nEmbd);
                    auto bgDn = makeBG(plQ8Matmul, {
                        {0, siluMulOutBuf}, {1, lw.dnW}, {2, lw.dnS},
                        {3, zeroBiasE}, {4, projOutBuf}, {5, dnMatmulP}});
                    allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgDn,
                        1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_down_sw"});
                }

                // 3. Post-FFN sandwich norm (in-place on projOutBuf)
                if (plRmsNormAdd && lw.postFfwNorm.handle) {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    memcpy(&data[1], &cfg.rmsNormEps, 4);
                    auto p = gpu->createBuffer("p_ffw_norm_add_" + std::to_string(i), 16);
                    gpu->writeBuffer(p, data, 16);
                    auto bg = makeBG(*plRmsNormAdd, {
                        {0, xBuf}, {1, projOutBuf}, {2, lw.postFfwNorm}, {3, p}});
                    allDecodeDispatches.push_back({plRmsNormAdd->pipeline, bg,
                        1, 1, 1, L+"fused_ffw_norm_add"});
                } else {
                auto& plRmsIPfw = gpu->getOrCreatePipeline("rms_norm_inplace",
                    std::string(R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
var<workgroup> wg_sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0]; let eps = bitcast<f32>(_p_[1]); let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; sum_sq += v * v; }
    wg_sums[tid] = sum_sq;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { wg_sums[tid] += wg_sums[tid + stride]; }
        workgroupBarrier();
    }
    let rms = 1.0 / sqrt(wg_sums[0] / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { X[j] = X[j] * rms * W[j]; }
}
)"), 3);
                GPUBuffer rmsP2;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                    rmsP2 = gpu->createBuffer("p_rmsip_ffw2_" + std::to_string(i), 16);
                    gpu->writeBuffer(rmsP2, data, 16);
                }
                auto bgFfwNorm = makeBG(plRmsIPfw, {
                    {0, projOutBuf}, {1, lw.postFfwNorm}, {2, rmsP2}});
                allDecodeDispatches.push_back({plRmsIPfw.pipeline, bgFfwNorm,
                    1, 1, 1, L+"post_ffw_norm"});

                // 4. Residual add: xBuf += projOutBuf
                auto& plAddIPsw = gpu->getOrCreatePipeline("add_inplace",
                    std::string(R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0]; let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)"), 3);
                GPUBuffer addP2;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    addP2 = gpu->createBuffer("p_add_ffw_" + std::to_string(i), 16);
                    gpu->writeBuffer(addP2, data, 16);
                }
                auto bgAdd2 = makeBG(plAddIPsw, {
                    {0, xBuf}, {1, projOutBuf}, {2, addP2}});
                allDecodeDispatches.push_back({plAddIPsw.pipeline, bgAdd2,
                    (cfg.nEmbd + 255) / 256, 1, 1, L+"add_res2"});
                }
            } else {
                auto bg = makeBG(plDnSilu, {
                    {0, gateUpBuf}, {1, lw.dnW}, {2, lw.dnS},
                    {3, zeroBiasE}, {4, xBuf}, {5, layerDnP}});
                allDecodeDispatches.push_back({plDnSilu.pipeline, bg,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"q8_down_silu_add"});
            }
        }

        // PLE (Per-Layer Embedding) injection
        if (cfg.pleSize > 0 && lw.pleInpGateW.handle && lw.pleProjW.handle) {
            uint32_t pleDim = cfg.pleSize;
            GPUBuffer geluP;
            {
                uint32_t data[4] = {pleDim, i * pleDim, 0, 0};
                geluP = gpu->createBuffer("p_ple_gelu_" + std::to_string(i), 16);
                gpu->writeBuffer(geluP, data, 16);
            }
            // 1. inp_gate matmul: xBuf [E] → pleBuf [pleSize]
            if (pleWeightsUseFp16 && plQ4A32Ple) {
                auto p = makeQ4Params("p_q4_ple_gate_" + std::to_string(i), cfg.nEmbd, pleDim);
                auto bgGate = makeBG(*plQ4A32Ple, {
                    {0, xBuf}, {1, lw.pleInpGateW}, {2, lw.pleInpGateS},
                    {3, pleBuf}, {4, p}});
                allDecodeDispatches.push_back({plQ4A32Ple->pipeline, bgGate,
                    (pleDim + Q4_A32_PLE_TILE - 1) / Q4_A32_PLE_TILE, 1, 1, L+"q4_ple_gate"});
            } else {
                GPUBuffer pleGateP = makeQ8Params("p_ple_gate_" + std::to_string(i), cfg.nEmbd, pleDim);
                auto bgGate = makeBG(plQ8Matmul, {
                    {0, xBuf}, {1, lw.pleInpGateW}, {2, lw.pleInpGateS},
                    {3, zeroBiasE}, {4, pleBuf}, {5, pleGateP}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgGate,
                    1, (pleDim + Q8_TILE - 1) / Q8_TILE, 1, L+"ple_gate"});
            }

            // 2. Fused GELU + multiply by this layer's PLE input slice.
            auto& plPleGeluMul = getKernel("ple_gelu_mul");
            auto bgGeluMul = makeBG(plPleGeluMul, {
                {0, pleBuf}, {1, pleInputBuf}, {2, geluP}});
            allDecodeDispatches.push_back({plPleGeluMul.pipeline, bgGeluMul,
                (pleDim + 255) / 256, 1, 1, L+"ple_gelu_mul"});

            // 4. Back-projection: pleBuf [pleSize] → pleOutBuf [E]
            if (pleWeightsUseFp16 && plQ4A32Ple) {
                auto p = makeQ4Params("p_q4_ple_proj_" + std::to_string(i), pleDim, cfg.nEmbd);
                auto bgProj = makeBG(*plQ4A32Ple, {
                    {0, pleBuf}, {1, lw.pleProjW}, {2, lw.pleProjS},
                    {3, pleOutBuf}, {4, p}});
                allDecodeDispatches.push_back({plQ4A32Ple->pipeline, bgProj,
                    (cfg.nEmbd + Q4_A32_PLE_TILE - 1) / Q4_A32_PLE_TILE, 1, 1, L+"q4_ple_proj"});
            } else {
                GPUBuffer pleProjP = makeQ8Params("p_ple_proj_" + std::to_string(i), pleDim, cfg.nEmbd);
                auto bgProj = makeBG(plQ8Matmul, {
                    {0, pleBuf}, {1, lw.pleProjW}, {2, lw.pleProjS},
                    {3, zeroBiasE}, {4, pleOutBuf}, {5, pleProjP}});
                allDecodeDispatches.push_back({plQ8Matmul.pipeline, bgProj,
                    1, (cfg.nEmbd + Q8_TILE - 1) / Q8_TILE, 1, L+"ple_proj"});
            }

            // 5. RMSNorm on pleOutBuf (in-place)
            if (plRmsNormAdd && lw.plePostNorm.handle) {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                memcpy(&data[1], &cfg.rmsNormEps, 4);
                auto p = gpu->createBuffer("p_ple_norm_add_" + std::to_string(i), 16);
                gpu->writeBuffer(p, data, 16);
                auto bg = makeBG(*plRmsNormAdd, {
                    {0, xBuf}, {1, pleOutBuf}, {2, lw.plePostNorm}, {3, p}});
                allDecodeDispatches.push_back({plRmsNormAdd->pipeline, bg,
                    1, 1, 1, L+"fused_ple_norm_add"});
            } else {
            if (lw.plePostNorm.handle) {
                auto& plRmsIPple = gpu->getOrCreatePipeline("rms_norm_inplace",
                    std::string(R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
var<workgroup> wg_sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _p_[0]; let eps = bitcast<f32>(_p_[1]); let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < N; j += 256u) { let v = X[j]; sum_sq += v * v; }
    wg_sums[tid] = sum_sq;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { wg_sums[tid] += wg_sums[tid + stride]; }
        workgroupBarrier();
    }
    let rms = 1.0 / sqrt(wg_sums[0] / f32(N) + eps);
    for (var j = tid; j < N; j += 256u) { X[j] = X[j] * rms * W[j]; }
}
)"), 3);
                GPUBuffer rmsPlP;
                {
                    uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                    float eps = cfg.rmsNormEps; memcpy(&data[1], &eps, 4);
                    rmsPlP = gpu->createBuffer("p_ple_norm_" + std::to_string(i), 16);
                    gpu->writeBuffer(rmsPlP, data, 16);
                }
                auto bgPleNorm = makeBG(plRmsIPple, {
                    {0, pleOutBuf}, {1, lw.plePostNorm}, {2, rmsPlP}});
                allDecodeDispatches.push_back({plRmsIPple.pipeline, bgPleNorm,
                    1, 1, 1, L+"ple_norm"});
            }

            // 6. Residual add: xBuf += pleOutBuf
            auto& plAddPLE = gpu->getOrCreatePipeline("add_inplace",
                std::string(R"(
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0]; let i = gid.x;
    if (i < N) { dst[i] = dst[i] + src[i]; }
}
)"), 3);
            GPUBuffer addPleP;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                addPleP = gpu->createBuffer("p_ple_add_" + std::to_string(i), 16);
                gpu->writeBuffer(addPleP, data, 16);
            }
            auto bgPleAdd = makeBG(plAddPLE, {
                {0, xBuf}, {1, pleOutBuf}, {2, addPleP}});
            allDecodeDispatches.push_back({plAddPLE.pipeline, bgPleAdd,
                (cfg.nEmbd + 255) / 256, 1, 1, L+"ple_add"});
            }
        }

        // Per-layer output scale
        if (lw.outScale.handle) {
            static const char* SCALE_BUF_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> _p_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _p_[0];
    let i = gid.x;
    if (i < N) { X[i] = X[i] * Scale[0]; }
}
)";
            auto& plScale = gpu->getOrCreatePipeline("scale_buf",
                std::string(SCALE_BUF_WGSL), 3);
            GPUBuffer scaleParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                scaleParams = gpu->createBuffer("p_scale_" + std::to_string(i), 16);
                gpu->writeBuffer(scaleParams, data, 16);
            }
            auto bgScale = makeBG(plScale, {
                {0, xBuf}, {1, lw.outScale}, {2, scaleParams}});
            allDecodeDispatches.push_back({plScale.pipeline, bgScale,
                (cfg.nEmbd + 255) / 256, 1, 1, L+"out_scale"});
        }

        // 10. RMSNorm for next layer (only needed for KQ path — Q8 path uses fused kernel)
        bool nextNeedsPrecomputedNorm = i + 1 < cfg.nLayer &&
            (cfg.arch != "qwen35" || cfg.isAttentionLayer(i + 1));
        if (useKQ && nextNeedsPrecomputedNorm) {
            auto bg = makeBG(plRmsNorm, {
                {0, xBuf}, {1, normOutBuf}, {2, layerWeights[i+1].inputNorm},
                {3, rstdBuf}, {4, rmsParams}});
            allDecodeDispatches.push_back({plRmsNorm.pipeline, bg, 1, 1, 1, L+"rms_next"});
        }
    }

    // Final RMSNorm
    {
        static const char* FINAL_RMS_PORTABLE_WGSL = R"(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid=lid.x; let N=P[0]; let eps=bitcast<f32>(P[1]); var ss=0.0;
    for(var i=tid;i<N;i+=256u){let v=X[i];ss+=v*v;}
    sums[tid]=ss;workgroupBarrier();
    for(var stride=128u;stride>0u;stride>>=1u){
        if(tid<stride){sums[tid]+=sums[tid+stride];}workgroupBarrier();
    }
    let r=inverseSqrt(sums[0]/f32(N)+eps);
    if(tid==0u){Rstd[0]=r;}
    for(var i=tid;i<N;i+=256u){Y[i]=X[i]*r*W[i];}
}
)";
        auto& plFinalRms = gpu->getOrCreatePipeline(
            "final_rms_portable", std::string(FINAL_RMS_PORTABLE_WGSL), 5);
        auto bg = makeBG(plFinalRms, {
            {0, xBuf}, {1, normOutBuf}, {2, finalNormW},
            {3, rstdBuf}, {4, rmsParams}});
        allDecodeDispatches.push_back({plFinalRms.pipeline, bg, 1, 1, 1, "final_rms"});
    }

    // LM head
    if (lmHeadIsKQ) {
        const bool usePackedQ6Dp4a = lmHeadKQType == GGUF_TYPE_Q6_K &&
            (isAmdAdapter || ((isIntelAdapter || isNvidiaAdapter) &&
                              cfg.arch == "qwen35")) &&
            useQ4KDp4a;
        bool useWideQ6 = lmHeadKQType == GGUF_TYPE_Q6_K &&
            gpu->backendType == WGPUBackendType_Vulkan &&
            !std::getenv("BP_Q6_LM_NARROW");
        auto* lmKQ = usePackedQ6Dp4a ? &getKernel(isIntelAdapter
                                   ? "q6k_matmul_prequant_dp4a_reduc16"
                                   : "q6k_matmul_prequant_dp4a")
                               : useWideQ6 ? &getKernel("q6k_matmul_wide")
                               : kqPipelineFor(lmHeadKQType);
        uint32_t tile = usePackedQ6Dp4a && isIntelAdapter ? 16u
                      : useWideQ6 ? 16u : kqTileFor(lmHeadKQType);
        if (usePackedQ6Dp4a) {
            auto& plQuant = getKernel("q8_quantize_dp4a");
            auto bgQuant = makeBG(plQuant, {
                {0, normOutBuf}, {1, kqActQ8Buf},
                {2, kqActScaleBuf}, {3, kqLmParams}});
            allDecodeDispatches.push_back({plQuant.pipeline, bgQuant,
                (cfg.nEmbd + 255u) / 256u, 1, 1, "lm_head_quant"});
        }
        auto bg = usePackedQ6Dp4a
            ? makeBG(*lmKQ, {{0, kqActQ8Buf}, {1, kqActScaleBuf},
                {2, lmHeadKQ}, {3, zeroBiasV}, {4, logitsBuf}, {5, kqLmParams}})
            : makeBG(*lmKQ, {{0, normOutBuf}, {1, lmHeadKQ}, {2, zeroBiasV},
                {3, logitsBuf}, {4, kqLmParams}});
        allDecodeDispatches.push_back({lmKQ->pipeline, bg,
            1, (cfg.nVocab + tile - 1) / tile, 1, "lm_head"});
    } else if (lmHeadIsQ4 && plQ4LmHead) {
        auto p = makeQ4Params("p_lmhead_q4", cfg.nEmbd, cfg.nVocab);
        auto bg = makeBG(*plQ4LmHead, {
            {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
            {3, logitsBuf}, {4, p}});
        allDecodeDispatches.push_back({plQ4LmHead->pipeline, bg,
            (cfg.nVocab + Q4_LM_TILE_N - 1) / Q4_LM_TILE_N, 1, 1, "lm_head"});
    } else if (lmHeadIsQ8) {
        if (plQ8DecDp4a) {
            // DP4A decode path: Params{M, N, K, pad}, TILE_N=32
            uint32_t pData[4] = {1u, cfg.nVocab, cfg.nEmbd, 0};
            auto lmDpParams = gpu->createBuffer("p_lmhead_q8_dp4a", 16);
            gpu->writeBuffer(lmDpParams, pData, 16);
            auto bg = makeBG(*plQ8DecDp4a, {
                {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
                {3, zeroBiasV}, {4, logitsBuf}, {5, lmDpParams}});
            allDecodeDispatches.push_back({plQ8DecDp4a->pipeline, bg,
                (cfg.nVocab + 31u) / 32u, 1, 1, "lm_head"});
        } else {
            auto q8LmParams = makeQ8Params("p_lmhead_q8", cfg.nEmbd, cfg.nVocab);
            auto bg = makeBG(plQ8Matmul, {
                {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
                {3, zeroBiasV}, {4, logitsBuf}, {5, q8LmParams}});
            allDecodeDispatches.push_back({plQ8Matmul.pipeline, bg,
                1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1, "lm_head"});
        }
    } else {
        const auto& plLmHead = tuning.decodeUseWideFp16 ? plFp16Wide : plFp16Gemm;
        uint32_t FP16_TILE = tuning.decodeUseWideFp16 ? 32u : 8u;
        auto bg = makeBG(plLmHead, {
            {0, normOutBuf}, {1, lmHeadW}, {2, zeroBiasV},
            {3, logitsBuf}, {4, lmheadParams}});
        allDecodeDispatches.push_back({plLmHead.pipeline, bg,
            1, (cfg.nVocab + FP16_TILE - 1) / FP16_TILE, 1, "lm_head"});
    }

    // Logit softcapping (Gemma 2/4): tanh(logits/cap) * cap
    if (cfg.logitSoftcap > 0.0f) {
        static const char* LOGIT_SOFTCAP_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> _params_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let cap = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= N) { return; }
    Y[i] = tanh(Y[i] / cap) * cap;
}
)";
        auto& plSoftcap = gpu->getOrCreatePipeline("logit_softcap",
            std::string(LOGIT_SOFTCAP_WGSL), 2);
        GPUBuffer softcapParams;
        {
            uint32_t data[4] = {cfg.nVocab, 0, 0, 0};
            memcpy(&data[1], &cfg.logitSoftcap, 4);
            softcapParams = gpu->createBuffer("p_softcap", 16);
            gpu->writeBuffer(softcapParams, data, 16);
        }
        auto bg = makeBG(plSoftcap, {
            {0, logitsBuf}, {1, softcapParams}});
        softcapPipeline = plSoftcap.pipeline;
        softcapBG = bg;
        softcapDispatchX = (cfg.nVocab + 255) / 256;
        allDecodeDispatches.push_back({softcapPipeline, bg,
            softcapDispatchX, 1, 1, "logit_softcap"});
    }

    // GPU argmax
    {
        auto& plArgmaxReduce = getKernel("argmax_reduce");
        auto bg = makeBG(plArgmax, {
            {0, logitsBuf}, {1, argmaxResultBuf}, {2, argmaxParams}, {3, argmaxPartialsBuf}});
        argmaxDispatchIndex = (int)allDecodeDispatches.size();
        allDecodeDispatches.push_back({plArgmax.pipeline, bg, argmaxNumWg, 1, 1, "argmax"});
        auto bgReduce = makeBG(plArgmaxReduce, {
            {0, argmaxPartialsBuf}, {1, argmaxResultBuf}, {2, argmaxReduceParams}});
        argmaxReduceDispatchIndex = (int)allDecodeDispatches.size();
        allDecodeDispatches.push_back({plArgmaxReduce.pipeline, bgReduce, 1, 1, 1, "argmax_reduce"});
    }

    // Auto-decode: embed_gather + full pipeline
    {
        // Gather the token embedding into xBuf (f32). Prefer gathering from the
        // tied Q8 LM head (no extra buffer); otherwise from the fp16 copy.
        std::vector<std::pair<uint32_t, GPUBuffer>> gatherBg =
            embeddingGatherFromKQ
            ? std::vector<std::pair<uint32_t, GPUBuffer>>{
                  {0, lmHeadKQ}, {1, argmaxResultBuf}, {2, xBuf}, {3, embedKQParams}}
            : embeddingGatherFromQ8
            ? std::vector<std::pair<uint32_t, GPUBuffer>>{
                  {0, lmHeadQ8W}, {1, lmHeadQ8S}, {2, argmaxResultBuf},
                  {3, xBuf}, {4, embedParams}}
            : std::vector<std::pair<uint32_t, GPUBuffer>>{
                  {0, embeddingGpuBuf}, {1, argmaxResultBuf},
                  {2, xBuf}, {3, embedParams}};
        auto& plGather = embeddingGatherFromKQ ? plEmbGatherKQ
            : (embeddingGatherFromQ8 ? plEmbGatherQ8 : plEmbGatherF16);
        auto bg = makeBG(plGather, gatherBg);
        autoDecodeDispatches.clear();
        autoDecodeDispatches.push_back({plGather.pipeline, bg,
            (cfg.nEmbd + 255) / 256, 1, 1, "embed_gather"});

        // Embedding scaling (Gemma: multiply by sqrt(n_embd))
        if (cfg.embeddingScale > 0.0f) {
            static const char* SCALE_INPLACE_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> _params_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let scale = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= N) { return; }
    X[i] = X[i] * scale;
}
)";
            auto& plScale = gpu->getOrCreatePipeline("scale_inplace",
                std::string(SCALE_INPLACE_WGSL), 2);
            GPUBuffer scaleParams;
            {
                uint32_t data[4] = {cfg.nEmbd, 0, 0, 0};
                memcpy(&data[1], &cfg.embeddingScale, 4);
                scaleParams = gpu->createBuffer("p_emb_scale", 16);
                gpu->writeBuffer(scaleParams, data, 16);
            }
            auto scaleBg = makeBG(plScale, {
                {0, xBuf}, {1, scaleParams}});
            autoDecodeDispatches.push_back({plScale.pipeline, scaleBg,
                (cfg.nEmbd + 255) / 256, 1, 1, "embed_scale"});
        }

        autoDecodeDispatches.insert(autoDecodeDispatches.end(),
            allDecodeDispatches.begin(), allDecodeDispatches.end());
    }

    fprintf(stderr, "  %zu decode dispatches (%u layers)\n",
           allDecodeDispatches.size(), cfg.nLayer);

    // ─── Create staging pool ──────────────────────────────────────────────
    pool.resize(decodePoolCapacity);
    for (int s = 0; s < decodePoolCapacity; s++) {
        auto& ps = pool[s];
        WGPUBufferDescriptor bd{};
        bd.usage = BUF_MAP_READ | BUF_COPY_DST;
        bd.size = 4;
        char label[32]; snprintf(label, 32, "staging_%d", s);
        bd.label = {label, (uint32_t)strlen(label)};
        ps.stagingBuf = wgpuDeviceCreateBuffer(gpu->device, &bd);
        ps.tokenInBuf = gpu->createBuffer("decode_token_in_" + std::to_string(s), 4);
        ps.tokenOutBuf = gpu->createBuffer("decode_token_out_" + std::to_string(s), 4);
        int32_t zeroToken = 0;
        gpu->writeBuffer(ps.tokenInBuf, &zeroToken, 4);
        gpu->writeBuffer(ps.tokenOutBuf, &zeroToken, 4);

        // Per-slot param buffers
        char rl[32]; snprintf(rl, 32, "p_frope_%d", s);
        ps.ropeParamsBuf = gpu->createBuffer(std::string(rl), 32);
        gpu->writeBuffer(ps.ropeParamsBuf, ropeParamData.data(), 32);
        char al[32]; snprintf(al, 32, "p_cattn_%d", s);
        ps.attnParamsBuf = gpu->createBuffer(std::string(al), 32);
        gpu->writeBuffer(ps.attnParamsBuf, chunkedAttnParamData.data(), 32);

        if (q35RopeQParamsBuf.handle) {
            uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
            uint32_t q[8] = {cfg.nHead, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], 0u, ropeHalf};
            uint32_t k[8] = {cfg.nKvHeads, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], 0u, ropeHalf};
            uint32_t kv[8] = {cfg.nKvHeads * cfg.headDim, 0u, 0, 0, 0, 0, 0, 0};
            char ql[32]; snprintf(ql, 32, "p_q35_rope_q_%d", s);
            char kl[32]; snprintf(kl, 32, "p_q35_rope_k_%d", s);
            char vl[32]; snprintf(vl, 32, "p_q35_kv_write_%d", s);
            ps.q35RopeQParamsBuf = gpu->createBuffer(std::string(ql), 32);
            ps.q35RopeKParamsBuf = gpu->createBuffer(std::string(kl), 32);
            ps.q35KvWriteParamsBuf = gpu->createBuffer(std::string(vl), 32);
            gpu->writeBuffer(ps.q35RopeQParamsBuf, q, 32);
            gpu->writeBuffer(ps.q35RopeKParamsBuf, k, 32);
            gpu->writeBuffer(ps.q35KvWriteParamsBuf, kv, 32);
        }

        // SWA per-slot param buffer
        if (hasSWA) {
            char sl[32]; snprintf(sl, 32, "p_cattn_swa_%d", s);
            ps.attnParamsBufSWA = gpu->createBuffer(std::string(sl), 32);
            gpu->writeBuffer(ps.attnParamsBufSWA, chunkedAttnParamData.data(), 32);
        }

        if (hasVariableHeadDim) {
            ps.layerRopeParamsBufs.resize(cfg.nLayer);
            ps.layerAttnParamsBufs.resize(cfg.nLayer);
            constexpr uint64_t kParamStride = 256;
            ps.layerRopeParamsPacked = gpu->createBuffer(
                "p_frope_packed_" + std::to_string(s), cfg.nLayer * kParamStride);
            ps.layerAttnParamsPacked = gpu->createBuffer(
                "p_cattn_packed_" + std::to_string(s), cfg.nLayer * kParamStride);
            for (uint32_t li = 0; li < cfg.nLayer; li++) {
                ps.layerRopeParamsBufs[li] = {ps.layerRopeParamsPacked.handle, 32,
                                               li * kParamStride};
                ps.layerAttnParamsBufs[li] = {ps.layerAttnParamsPacked.handle, 32,
                                               li * kParamStride};
            }
        }

        // Clone autoDecodeDispatches with per-slot bind groups for
        // dispatches that reference dynamic param buffers.
        ps.dispatches = autoDecodeDispatches;
        // Prefix dispatches before allDecodeDispatches: embed_gather [+ embed_scale]
        autoDecodePrefixCount = (cfg.embeddingScale > 0.0f) ? 2 : 1;
        int prefixCount = autoDecodePrefixCount;
        auto& plQ35RopeToQRot = getKernel("qwen35_rope_q_to_qrot");
        auto& plQ35KvWriteRope = getKernel("qwen35_kv_cache_write_rope");
        if (embeddingGatherFromKQ) {
            ps.dispatches[0].bindGroup = makeBG(plEmbGatherKQ, {
                {0, lmHeadKQ}, {1, ps.tokenInBuf}, {2, xBuf}, {3, embedKQParams}});
        } else if (embeddingGatherFromQ8) {
            ps.dispatches[0].bindGroup = makeBG(plEmbGatherQ8, {
                {0, lmHeadQ8W}, {1, lmHeadQ8S}, {2, ps.tokenInBuf},
                {3, xBuf}, {4, embedParams}});
        } else {
            ps.dispatches[0].bindGroup = makeBG(plEmbGatherF16, {
                {0, embeddingGpuBuf}, {1, ps.tokenInBuf},
                {2, xBuf}, {3, embedParams}});
        }
        if (pleTokenGatherDispatchIndex >= 0 && plPleTokenGather) {
            int idx = pleTokenGatherDispatchIndex + prefixCount;
            ps.dispatches[idx].bindGroup = pleTokenEmbAsymmetric
                ? makeBG(*plPleTokenGather, {
                    {0, pleTokenEmbW}, {1, pleTokenEmbS}, {2, pleTokenEmbZ},
                    {3, ps.tokenInBuf}, {4, pleInputBuf}, {5, pleTokenGatherParams}})
                : makeBG(*plPleTokenGather, {
                    {0, pleTokenEmbW}, {1, pleTokenEmbS}, {2, ps.tokenInBuf},
                    {3, pleInputBuf}, {4, pleTokenGatherParams}});
        }
        if (argmaxDispatchIndex >= 0) {
            int idx = argmaxDispatchIndex + prefixCount;
            ps.dispatches[idx].bindGroup = makeBG(plArgmax, {
                {0, logitsBuf}, {1, ps.tokenOutBuf}, {2, argmaxParams}, {3, argmaxPartialsBuf}});
        }
        if (argmaxReduceDispatchIndex >= 0) {
            int idx = argmaxReduceDispatchIndex + prefixCount;
            auto& plArgmaxReduce = getKernel("argmax_reduce");
            ps.dispatches[idx].bindGroup = makeBG(plArgmaxReduce, {
                {0, argmaxPartialsBuf}, {1, ps.tokenOutBuf}, {2, argmaxReduceParams}});
        }
        for (uint32_t layer = 0; layer < cfg.nLayer; layer++) {
            bool layerIsSWA = hasSWA && layer < cfg.layerAttnTypes.size() &&
                              cfg.layerAttnTypes[layer] == AttnLayerType::SlidingWindow;
            uint32_t hdL = cfg.perLayer[layer].headDim;
            uint32_t kvCacheLayer = cfg.perLayer[layer].kvSourceLayer >= 0
                ? (uint32_t)cfg.perLayer[layer].kvSourceLayer : layer;
            if (ropeDispatchIndices[layer] >= 0) {
                int ropeIdx = ropeDispatchIndices[layer] + prefixCount;
                auto& plRope = hasVariableHeadDim
                    ? getKernelHD(cfg.hasQkNorm ? "fused_qknorm_rope" : "fused_rope", hdL)
                    : plFusedRope;
                auto& ropeCos = layerIsSWA && ropeCosBufSWA.handle ? ropeCosBufSWA : ropeCosBuf;
                auto& ropeSin = layerIsSWA && ropeSinBufSWA.handle ? ropeSinBufSWA : ropeSinBuf;
                auto& lw = layerWeights[layer];
                auto& ropeK = lw.qOnly && qOnlyScratchK.handle ? qOnlyScratchK : kvCache[kvCacheLayer].K;
                auto& ropeV = lw.qOnly && qOnlyScratchV.handle ? qOnlyScratchV : kvCache[kvCacheLayer].V;
                auto& ropeP = hasVariableHeadDim ? ps.layerRopeParamsBufs[layer] : ps.ropeParamsBuf;
                ps.dispatches[ropeIdx].bindGroup = makeBG(plRope, {
                    {0, qkvBuf}, {1, qRotBuf},
                    {2, ropeK}, {3, ropeV}, {4, ropeCos}, {5, ropeSin},
                    {6, lw.qNorm}, {7, lw.kNorm}, {8, ropeP}});
            }

            if (q35QRoPEDispatchIndices[layer] >= 0 && ps.q35RopeQParamsBuf.handle) {
                int qIdx = q35QRoPEDispatchIndices[layer] + prefixCount;
                ps.dispatches[qIdx].bindGroup = makeBG(plQ35RopeToQRot, {
                    {0, q35QBuf}, {1, qRotBuf}, {2, ropeCosBuf}, {3, ropeSinBuf},
                    {4, ps.q35RopeQParamsBuf}});
            }

            if (q35KvWriteDispatchIndices[layer] >= 0 && ps.q35KvWriteParamsBuf.handle) {
                int kvIdx = q35KvWriteDispatchIndices[layer] + prefixCount;
                ps.dispatches[kvIdx].bindGroup = makeBG(plQ35KvWriteRope, {
                    {0, q35KBuf}, {1, q35VBuf}, {2, kvCache[layer].K}, {3, kvCache[layer].V},
                    {4, ropeCosBuf}, {5, ropeSinBuf}, {6, ps.q35RopeKParamsBuf},
                    {7, ps.q35KvWriteParamsBuf}});
            }

            if (attnP1DispatchIndices[layer] < 0 || attnP2DispatchIndices[layer] < 0) {
                continue;
            }
            int p1Idx = attnP1DispatchIndices[layer] + prefixCount;
            int p2Idx = attnP2DispatchIndices[layer] + prefixCount;

            auto& slotAttnBuf = hasVariableHeadDim ? ps.layerAttnParamsBufs[layer]
                : (layerIsSWA ? ps.attnParamsBufSWA : ps.attnParamsBuf);
            auto& plP1 = hasVariableHeadDim
                ? getKernelHD(useSubgroupChunkedAttention
                    ? "gqa_chunked_pass1_subgroup" : "gqa_chunked_pass1", hdL)
                : plChunkP1;
            auto& plP2 = hasVariableHeadDim ? getKernelHD("gqa_chunked_pass2", hdL) : plChunkP2;

            ps.dispatches[p1Idx].bindGroup = makeBG(plP1, {
                {0, qRotBuf}, {1, kvCache[kvCacheLayer].K}, {2, kvCache[kvCacheLayer].V},
                {3, attnPartialsBuf}, {4, slotAttnBuf}});

            ps.dispatches[p2Idx].bindGroup = makeBG(plP2, {
                {0, attnPartialsBuf}, {1, attnOutBuf},
                {2, slotAttnBuf}});
        }

    }
    for (int s = 0; s < decodePoolCapacity; s++)
        refillCBPool(s);
    // Slot 0 also has a prompt-only plan that stops before final RMS/logits.
    // Intermediate known prompt tokens need hidden/KV state, not vocabulary work.
    refillKnownPrefillCBPool(0);
    fprintf(stderr, "  Pool: %d slots × %d pre-recorded CBs\n",
            decodePoolCapacity, decodeCbPoolBatch);

    // Pre-allocate prefill resources (buffers + bind groups at maxSeqLen)
    // Skip for K-quant models — prefill kernels only support Q8/fp32 weights.
    // Also skip for shared-KV models (Gemma 4): the batched prefill path
    // assumes every layer has a full QKV projection, which shared-KV (Q-only)
    // layers lack. Fall back to serial decode for prefill there.
    bool isKQ = weightsUseNativeKQ;
    bool hasSharedKv = cfg.sharedKvLayers > 0;
    // Batched prefill assumes standard 2-norm layers with a full QKV per layer.
    // Gemma sandwich-norm models (and shared-KV) don't fit it; use serial
    // decode. qwen35 keeps its own path and batched prefill.
    bool isGemma = cfg.arch.rfind("gemma", 0) == 0;
    bool gemmaSerial = isGemma && (cfg.hasSandwichNorm || hasSharedKv);
    if (cfg.arch == "qwen35") {
        const bool intelQwen4B = gpu->adapterName.find("Intel") != std::string::npos &&
                                cfg.nEmbd > 2048;
        if (!std::getenv("BP_QWEN35_SERIAL_PREFILL") && !intelQwen4B) {
            initQwen35PrefillResources();
        } else {
            fprintf(stderr, "  Prefill: Qwen3.5 serial path\n");
        }
    } else if (!isKQ && !gemmaSerial) {
        initPrefillResources();
    } else if (gemmaSerial) {
        const bool intelAdapter = gpu->adapterName.find("Intel") != std::string::npos;
        if (cfg.arch == "gemma4" && !weightsUseNativeKQ && gpu->supportsSubgroups &&
            !intelAdapter) {
            initGemmaPrefillResources();
        } else {
            fprintf(stderr, "  Prefill: skipped (Gemma sandwich/shared-KV, using serial decode)\n");
        }
    } else {
        fprintf(stderr, "  Prefill: skipped (K-quant weights, using serial decode)\n");
    }
}

// ─── Pre-allocate prefill resources ──────────────────────────────────────────

void ModelRunner::initQwen35PrefillResources() {
    const uint32_t C=qwen35Pf.capacity,E=cfg.nEmbd;
    const uint32_t convChannels=cfg.ssmInnerSize+2u*cfg.ssmGroupCount*cfg.ssmStateSize;
    const uint32_t qdim=cfg.nHead*cfg.headDim,kvdim=cfg.nKvHeads*cfg.headDim;
    const uint32_t im=cfg.intermediateSize;
    auto mk=[&](const char* n,uint64_t elems){return gpu->createBuffer(n,std::max<uint64_t>(4,elems*4));};
    qwen35Pf.tokens=mk("qpf_tokens",C);qwen35Pf.x=mk("qpf_x",(uint64_t)C*E);
    qwen35Pf.norm=mk("qpf_norm",(uint64_t)C*E);qwen35Pf.proj=mk("qpf_proj",(uint64_t)C*E);
    qwen35Pf.gateup=mk("qpf_gateup",(uint64_t)C*2u*im);qwen35Pf.act=mk("qpf_act",(uint64_t)C*im);
    qwen35Pf.rstd=mk("qpf_rstd",C);qwen35Pf.qkv=mk("qpf_qkv",(uint64_t)C*convChannels);
    qwen35Pf.z=mk("qpf_z",(uint64_t)C*cfg.ssmInnerSize);qwen35Pf.beta=mk("qpf_beta",(uint64_t)C*cfg.ssmTimeStepRank);
    qwen35Pf.gate=mk("qpf_gate",(uint64_t)C*cfg.ssmTimeStepRank);qwen35Pf.conv=mk("qpf_conv",(uint64_t)C*convChannels);
    qwen35Pf.sq=mk("qpf_sq",(uint64_t)C*cfg.ssmGroupCount*cfg.ssmStateSize);
    qwen35Pf.sk=mk("qpf_sk",(uint64_t)C*cfg.ssmGroupCount*cfg.ssmStateSize);
    qwen35Pf.sv=mk("qpf_sv",(uint64_t)C*cfg.ssmInnerSize);qwen35Pf.sy=mk("qpf_sy",(uint64_t)C*cfg.ssmInnerSize);
    qwen35Pf.snorm=mk("qpf_snorm",(uint64_t)C*cfg.ssmInnerSize);
    qwen35Pf.qj=mk("qpf_qj",(uint64_t)C*2u*qdim);qwen35Pf.aq=mk("qpf_aq",(uint64_t)C*qdim);
    qwen35Pf.ag=mk("qpf_ag",(uint64_t)C*qdim);qwen35Pf.ak=mk("qpf_ak",(uint64_t)C*kvdim);
    qwen35Pf.av=mk("qpf_av",(uint64_t)C*kvdim);qwen35Pf.qrot=mk("qpf_qrot",(uint64_t)C*qdim);
    qwen35Pf.attn=mk("qpf_attn",(uint64_t)C*qdim);qwen35Pf.aout=mk("qpf_aout",(uint64_t)C*qdim);
    qwen35Pf.paramArena=gpu->createBuffer("qpf_param_arena",128u*1024u,
        BUF_STORAGE|BUF_UNIFORM|BUF_COPY_DST);
    (void)getKernel("q6k_gather_batched");(void)getKernel("q8_matmul_batched_dp4a");
    (void)getKernel("q4k_matmul_batched4");
    (void)getKernel("q4k_matmul_batched8");
    (void)getKernel("q5k_matmul_batched4");
    (void)getKernel("q6k_matmul_batched4");
    (void)getKernel("qwen35_conv_scan_silu");(void)getKernel("qwen35_split_qkv_l2_batched");
    (void)getKernel("qwen35_conv_scan_split_l2");
    (void)getKernel("delta_net_scan_x2");(void)getKernel("qwen35_norm_gated_batched");
    (void)getKernel("qwen35_split_qg_batched");(void)getKernel("head_rmsnorm_batched");
    (void)getKernel("qwen35_rope_kv_batched");(void)getKernel("gated_output_batched");
    (void)getKernel("qwen35_alpha_beta_gate_batched");(void)getKernel("silu_mul_batched");
    (void)getKernel("rms_norm_batched");(void)getKernel("add_rms_norm_batched");
    (void)getKernel("gemma_norm_add_batched");
    (void)getKernel("add_inplace_batched");
    for(const auto& pl:cfg.perLayer)if(pl.headDim)(void)getKernelHD(
        gpu->backendType == WGPUBackendType_D3D12 ? "causal_attn" : "flash_attn_vulkan",
        pl.headDim);
    qwen35Pf.ready=true;
    fprintf(stderr,"  Qwen3.5 hybrid batched prefill: enabled (chunk=%u)\n",C);
}

void ModelRunner::initGemmaPrefillResources() {
    const uint32_t C = gemmaPf.capacity;
    uint32_t maxQkv = 0, maxQ = 0, maxIM = 0;
    for (const auto& pl : cfg.perLayer) {
        maxQ = std::max(maxQ, pl.qDim);
        maxQkv = std::max(maxQkv, pl.qDim + 2u * pl.kvDim);
        maxIM = std::max(maxIM, pl.intermediateSize);
    }
    auto mk = [&](const char* name, uint64_t elems, uint64_t bytes = 4) {
        return gpu->createBuffer(name, std::max<uint64_t>(4, elems * bytes));
    };
    gemmaPf.tokens = mk("gpf_tokens", C);
    gemmaPf.x      = mk("gpf_x", C * cfg.nEmbd);
    gemmaPf.norm   = mk("gpf_norm", C * cfg.nEmbd);
    gemmaPf.qkv    = mk("gpf_qkv", C * maxQkv);
    gemmaPf.qrot   = mk("gpf_qrot", C * maxQ);
    gemmaPf.attn   = mk("gpf_attn", C * maxQ);
    gemmaPf.proj   = mk("gpf_proj", C * cfg.nEmbd);
    gemmaPf.gateup = mk("gpf_gateup", C * 2ull * maxIM);
    gemmaPf.act    = mk("gpf_act", C * maxIM);
    gemmaPf.rstd   = mk("gpf_rstd", C);
    if (cfg.pleSize > 0) {
        uint64_t totalPle = (uint64_t)cfg.pleSize * cfg.nLayer;
        gemmaPf.pleSignal = mk("gpf_ple_signal", C * totalPle);
        gemmaPf.pleRaw    = mk("gpf_ple_raw", C * totalPle);
        gemmaPf.pleGate   = mk("gpf_ple_gate", C * cfg.pleSize);
        gemmaPf.pleOut    = mk("gpf_ple_out", C * cfg.nEmbd);
    }
    // Force compilation at load so unsupported shader features fail before
    // the first prompt rather than midway through inference.
    (void)getKernel("matmul_q4_batched");
    (void)getKernel(useDP4A ? "q8_matmul_batched_dp4a" : "q8_matmul_d3d12");
    (void)getKernel("q4_gather_batched");
    (void)getKernel("gemma_sandwich_attn_batched");
    (void)getKernel("gemma_norm_add_batched");
    (void)getKernel("gelu_mul_batched");
    (void)getKernel("ple_gelu_mul_batched");
    (void)getKernel("ple_combine_batched");
    for (const auto& pl : cfg.perLayer)
        if (pl.headDim) (void)getKernelHD(gpu->backendType == WGPUBackendType_D3D12
            ? "causal_attn" : "flash_attn_vulkan", pl.headDim);
    gemmaPf.ready = true;
    fprintf(stderr, "  Gemma batched prefill: enabled (chunk=%u)\n", C);
}

void ModelRunner::initPrefillResources() {
    uint32_t T = maxSeqLen;
    uint32_t qDimL  = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;
    const bool useGelu = (cfg.activation == ActivationType::GELU);
    uint32_t Q8_TILE = 8;

    // Intermediate buffers sized to maxSeqLen
    pfCache.pX    = gpu->createBuffer("pf_x",    T * cfg.nEmbd * 4);
    pfCache.pNorm = gpu->createBuffer("pf_norm", T * cfg.nEmbd * 4);
    pfCache.pQkv  = gpu->createBuffer("pf_qkv",  T * qkvOutL * 4);
    pfCache.pQRot = gpu->createBuffer("pf_qrot", T * qDimL * 4);
    pfCache.pAttn = gpu->createBuffer("pf_attn", T * qDimL * 4);
    pfCache.pProj = gpu->createBuffer("pf_proj", T * cfg.nEmbd * 4);
    pfCache.pGU   = gpu->createBuffer("pf_gu",   T * 2 * cfg.intermediateSize * 4);
    pfCache.pRstd = gpu->createBuffer("pf_rstd", T * 4);
    pfCache.pNormQ  = gpu->createBuffer("pf_norm_q",  T * cfg.nEmbd);
    pfCache.pNormQS = gpu->createBuffer("pf_norm_qs", T * (cfg.nEmbd / 32) * 4);
    pfCache.pAttnQ  = gpu->createBuffer("pf_attn_q",  T * qDimL);
    pfCache.pAttnQS = gpu->createBuffer("pf_attn_qs", T * (qDimL / 32) * 4);
    pfCache.pGUQ    = gpu->createBuffer("pf_gu_q",    T * cfg.intermediateSize);
    pfCache.pGUQS   = gpu->createBuffer("pf_gu_qs",   T * (cfg.intermediateSize / 32) * 4);

    // Global param buffers (written per-call with actual T)
    // Both backends use var<uniform> for early loop exit
    bool isVulkan = (gpu->backendType != WGPUBackendType_D3D12);
    uint64_t paramUsage = BUF_UNIFORM | BUF_COPY_DST;
    pfCache.pQkvP = gpu->createBuffer("pp_qkv", 16, paramUsage);
    pfCache.pOpP  = gpu->createBuffer("pp_op",  16, paramUsage);
    pfCache.pGuP  = gpu->createBuffer("pp_gu",  16, paramUsage);
    pfCache.pDnP  = gpu->createBuffer("pp_dn",  16, paramUsage);
    pfCache.pNormQP = gpu->createBuffer("pp_norm_q", 16, paramUsage);
    pfCache.pAttnQP = gpu->createBuffer("pp_attn_q", 16, paramUsage);
    pfCache.pGUQP   = gpu->createBuffer("pp_gu_q",   16, paramUsage);
    pfCache.pLmP  = gpu->createBuffer("pp_lm",  16);
    {
        uint32_t d[4] = {cfg.nEmbd, cfg.nEmbd, 0, 0};
        float eps = cfg.rmsNormEps; memcpy(&d[2], &eps, 4);
        pfCache.pRmsP = gpu->createBuffer("pp_rms", 16);
        gpu->writeBuffer(pfCache.pRmsP, d, 16);
    }

    // Per-layer param buffers
    pfCache.ropeParams.resize(cfg.nLayer);
    pfCache.attnParams.resize(cfg.nLayer);
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        pfCache.ropeParams[li] = gpu->createBuffer(
            "pp_rope_L" + std::to_string(li), 32);
        // Attention params always uniform (enables early exit on both backends)
        pfCache.attnParams[li] = gpu->createBuffer(
            "pp_attn_L" + std::to_string(li), 32,
            BUF_UNIFORM | BUF_COPY_DST);
    }

    // Get kernels — MMA on Vulkan, prequantized DP4A on D3D12 when available
    auto& kRmsB    = getKernel("rms_norm_batched");
    auto& kAddRmsB = getKernel("add_rms_norm_batched");
    auto& kRopeB   = getKernelHD(cfg.hasQkNorm ? "rope_batched_simple" : "rope_batched");

    const auto& limits = effectiveLimits(*gpu);
    const bool canUse512ThreadKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 512u &&
        limits.maxComputeWorkgroupStorageSize >= 32u * 1024u;
    const bool canUse256ThreadSubgroupKernels =
        gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 256u &&
        limits.maxComputeWorkgroupStorageSize >= 16u * 1024u;

    // Select matmul + attention kernels based on capability, not backend alone.
    useMMA = (gpu->backendType != WGPUBackendType_D3D12) &&
             gpu->supportsSubgroupMatrix &&
             canUse512ThreadKernels;

    // Detect DP4A support on D3D12 (dot4I8Packed → 4× INT8 throughput)
    useDP4A = false;
    if (!useMMA) {
        useDP4A = canUse256ThreadSubgroupKernels &&
                  canCompileEmbeddedKernel(*gpu, "q8_matmul_dp4a_d3d12");
        fprintf(stderr, "  DP4A (dot4I8Packed): %s\n",
               useDP4A ? "available" : "not available");
    }

    bool usePrequant = !useMMA && useDP4A;
    tuning.prefillUseWidePrequant = usePrequant && canUse256ThreadSubgroupKernels;
    tuning.prefillUseWidePrequantAdd = usePrequant && canUse256ThreadSubgroupKernels;
    tuning.prefillMatM = useMMA ? 64u : 8u;
    tuning.prefillMatN = useMMA ? 64u : 32u;
    tuning.prefillWideMatM = tuning.prefillUseWidePrequant ? 4u : tuning.prefillMatM;
    tuning.prefillWideMatN = tuning.prefillUseWidePrequant ? 64u : tuning.prefillMatN;
    tuning.prefillDnM = tuning.prefillUseWidePrequantAdd ? 4u : tuning.prefillMatM;
    tuning.prefillDnN = tuning.prefillUseWidePrequantAdd ? 64u : tuning.prefillMatN;
    tuning.prefillAttnBlockQ = useMMA ? 16u : 4u;

    const char* wideMatKernel = tuning.prefillUseWidePrequant
        ? "q8_matmul_prequant_wide_d3d12"
        : nullptr;
    const char* wideAddKernel = tuning.prefillUseWidePrequantAdd
        ? "q8_matmul_prequant_add_wide_d3d12"
        : nullptr;
    const char* matKernel = useMMA ? "q8_matmul_vulkan"
                          : (usePrequant ? "q8_matmul_prequant_d3d12" : "q8_matmul_d3d12");
    const char* dnSiluKernel = useMMA ? "q8_down_silu_add_vulkan"
                             : (usePrequant ? (tuning.prefillUseWidePrequantAdd
                                                ? "q8_matmul_prequant_add_wide_d3d12"
                                                : "q8_matmul_prequant_add_d3d12")
                                            : "q8_down_silu_add_d3d12");
    const char* attnKernel = useMMA ? "flash_attn_vulkan" : "causal_attn";
    const CompiledPipeline* kQuant = usePrequant ? &getKernel("quantize_fp32_rows_d3d12") : nullptr;
    const CompiledPipeline* kSiluQ = usePrequant ? &(useGelu ? getKernelGelu("silu_quantize_rows_d3d12") : getKernel("silu_quantize_rows_d3d12")) : nullptr;
    const CompiledPipeline* kMatWide = tuning.prefillUseWidePrequant
        ? &getKernel(wideMatKernel)
        : nullptr;
    const CompiledPipeline* kDnSiluWide = tuning.prefillUseWidePrequantAdd
        ? &getKernel(wideAddKernel)
        : nullptr;
    auto& kMat     = getKernel(matKernel);
    auto& kDnSilu  = useGelu ? getKernelGelu(dnSiluKernel) : getKernel(dnSiluKernel);
    auto& kAttn    = getKernelHD(attnKernel);
    fprintf(stderr, "  Prefill tuning: mat=%s qkv/gateup=%s down=%s attn=%s tiles=%ux%u wide=%ux%u pool=%d\n",
           matKernel,
           tuning.prefillUseWidePrequant ? wideMatKernel : matKernel,
           dnSiluKernel,
           attnKernel,
           tuning.prefillMatM, tuning.prefillMatN,
           tuning.prefillWideMatM, tuning.prefillWideMatN,
           decodePoolDepth);

    // Build bind groups (stable — buffer handles don't change)
    pfCache.layerBGs.resize(cfg.nLayer);
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        auto& lw = layerWeights[li];
        auto& bg = pfCache.layerBGs[li];

        if (!lw.qkvW.handle && !lw.qkvKQ.handle) {
            continue;
        }

        bg.rms = makeBG(kRmsB, {
            {0, pfCache.pX}, {1, pfCache.pNorm}, {2, lw.inputNorm},
            {3, pfCache.pRstd}, {4, pfCache.pRmsP}});

        if (usePrequant) {
            bg.qnorm = makeBG(*kQuant, {
                {0, pfCache.pNorm}, {1, pfCache.pNormQ}, {2, pfCache.pNormQS},
                {3, pfCache.pNormQP}});

            bg.qkv = makeBG(tuning.prefillUseWidePrequant ? *kMatWide : kMat, {
                {0, pfCache.pNormQ}, {1, pfCache.pNormQS}, {2, lw.qkvW},
                {3, lw.qkvS}, {4, zeroBiasQKV}, {5, pfCache.pQkv}, {6, pfCache.pQkvP}});
        } else {
            bg.qkv = makeBG(kMat, {
                {0, pfCache.pNorm}, {1, lw.qkvW}, {2, lw.qkvS},
                {3, zeroBiasQKV}, {4, pfCache.pQkv}, {5, pfCache.pQkvP}});
        }

        bg.rope = makeBG(kRopeB, {
            {0, pfCache.pQkv}, {1, pfCache.pQRot},
            {2, kvCache[li].K}, {3, kvCache[li].V},
            {4, ropeCosBuf}, {5, ropeSinBuf},
            {6, lw.qNorm}, {7, lw.kNorm},
            {8, pfCache.ropeParams[li]}});

        bg.attn = makeBG(kAttn, {
            {0, pfCache.pQRot}, {1, kvCache[li].K}, {2, kvCache[li].V},
            {3, pfCache.pAttn}, {4, pfCache.attnParams[li]}});

        if (usePrequant) {
            bg.attnq = makeBG(*kQuant, {
                {0, pfCache.pAttn}, {1, pfCache.pAttnQ}, {2, pfCache.pAttnQS},
                {3, pfCache.pAttnQP}});

            bg.oproj = makeBG(kMat, {
                {0, pfCache.pAttnQ}, {1, pfCache.pAttnQS}, {2, lw.oW},
                {3, lw.oS}, {4, zeroBiasE}, {5, pfCache.pProj}, {6, pfCache.pOpP}});
        } else {
            bg.oproj = makeBG(kMat, {
                {0, pfCache.pAttn}, {1, lw.oW}, {2, lw.oS},
                {3, zeroBiasE}, {4, pfCache.pProj}, {5, pfCache.pOpP}});
        }

        bg.addrms = makeBG(kAddRmsB, {
            {0, pfCache.pX}, {1, pfCache.pProj}, {2, pfCache.pNorm},
            {3, lw.postAttnNorm}, {4, pfCache.pRstd}, {5, pfCache.pRmsP}});

        if (usePrequant) {
            bg.gateup = makeBG(tuning.prefillUseWidePrequant ? *kMatWide : kMat, {
                {0, pfCache.pNormQ}, {1, pfCache.pNormQS}, {2, lw.guW},
                {3, lw.guS}, {4, zeroBiasGU}, {5, pfCache.pGU}, {6, pfCache.pGuP}});

            bg.siluq = makeBG(*kSiluQ, {
                {0, pfCache.pGU}, {1, pfCache.pGUQ}, {2, pfCache.pGUQS},
                {3, pfCache.pGUQP}});

            bg.downsilu = makeBG(kDnSilu, {
                {0, pfCache.pGUQ}, {1, pfCache.pGUQS}, {2, lw.dnW},
                {3, lw.dnS}, {4, zeroBiasE}, {5, pfCache.pX}, {6, pfCache.pDnP}});
        } else {
            bg.gateup = makeBG(kMat, {
                {0, pfCache.pNorm}, {1, lw.guW}, {2, lw.guS},
                {3, zeroBiasGU}, {4, pfCache.pGU}, {5, pfCache.pGuP}});

            bg.downsilu = makeBG(kDnSilu, {
                {0, pfCache.pGU}, {1, lw.dnW}, {2, lw.dnS},
                {3, zeroBiasE}, {4, pfCache.pX}, {5, pfCache.pDnP}});
        }
    }

    pfCache.finalRmsBG = makeBG(kRmsB, {
        {0, pfCache.pX}, {1, pfCache.pNorm}, {2, finalNormW},
        {3, pfCache.pRstd}, {4, pfCache.pRmsP}});

    // LM head bind group
    if (lmHeadIsQ8) {
        pfCache.lmBG = makeBG(getKernel("q8_matmul"), {
            {0, normOutBuf}, {1, lmHeadQ8W}, {2, lmHeadQ8S},
            {3, zeroBiasV}, {4, logitsBuf}, {5, pfCache.pLmP}});
    }

    // Argmax bind group. The argmax kernel declares 4 bindings (the 4th is the
    // Partials scratch buffer used by the multi-WG decode path); the prefill
    // single-WG dispatch still needs to bind all four or Dawn rejects the BG.
    {
        GPUBuffer argmaxP = gpu->createBuffer("pf_argmax_p", 16);
        uint32_t p[4] = {cfg.nVocab, 1u, 0, 0};
        gpu->writeBuffer(argmaxP, p, 16);
        pfCache.argmaxBG = makeBG(getKernel("argmax"), {
            {0, logitsBuf}, {1, argmaxResultBuf}, {2, argmaxP}, {3, argmaxPartialsBuf}});
    }

    // ── Build pre-recorded indirect dispatch table ───────────────────
    // Pipelines and bind groups are static; only grid sizes change with T.
    // We store a fixed sequence of (pipeline, bindGroup, indirectOffset) entries
    // and update the indirect buffer contents before each prefill call.
    {
        uint32_t perLayerDispatches = usePrequant ? 12u : 8u;
        uint32_t nDispatches = cfg.nLayer * perLayerDispatches + 1;
        pfCache.indirectTable.clear();
        pfCache.indirectTable.reserve(nDispatches);

        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            auto& bg = pfCache.layerBGs[li];
            auto addEntry = [&](WGPUComputePipeline pl, WGPUBindGroup bg_,
                                const std::string& name) {
                uint64_t off = pfCache.indirectTable.size() * 12;
                pfCache.indirectTable.push_back({pl, bg_, off, name});
            };
            addEntry(kRmsB.pipeline,    bg.rms,      "pf_rms");
            if (usePrequant) addEntry(kQuant->pipeline, bg.qnorm,   "pf_qnorm");
            addEntry(tuning.prefillUseWidePrequant ? kMatWide->pipeline : kMat.pipeline,
                     bg.qkv,      "pf_qkv");
            addEntry(kRopeB.pipeline,   bg.rope,     "pf_rope");
            addEntry(kAttn.pipeline,    bg.attn,     "pf_attn");
            if (usePrequant) addEntry(kQuant->pipeline, bg.attnq,   "pf_attn_quant");
            addEntry(kMat.pipeline,     bg.oproj,    "pf_oproj");
            addEntry(kAddRmsB.pipeline, bg.addrms,   "pf_add_rms");
            if (usePrequant) addEntry(kQuant->pipeline, bg.qnorm,   "pf_qnorm_ffn");
            addEntry(tuning.prefillUseWidePrequant ? kMatWide->pipeline : kMat.pipeline,
                     bg.gateup,   "pf_gateup");
            if (usePrequant) addEntry(kSiluQ->pipeline, bg.siluq,   "pf_silu_quant");
            addEntry((tuning.prefillUseWidePrequantAdd && kDnSiluWide) ? kDnSiluWide->pipeline : kDnSilu.pipeline,
                     bg.downsilu, "pf_down_silu");
        }
        {
            uint64_t off = pfCache.indirectTable.size() * 12;
            pfCache.indirectTable.push_back({kRmsB.pipeline, pfCache.finalRmsBG,
                                              off, "pf_final_rms"});
        }

        // Create the indirect buffer: [gx, gy, gz] × nDispatches
        pfCache.indirectBuf = gpu->createBuffer("pf_indirect",
            nDispatches * 12, BUF_INDIRECT | BUF_COPY_DST);
    }

    pfCache.ready = true;
}

// ─── Pre-record command buffer pool ──────────────────────────────────────────

void ModelRunner::refillCBPool(int slot) {
    auto& ps = pool[slot];

    // Release any remaining CBs
    for (int i = ps.cbIdx; i < (int)ps.cbPool.size(); i++)
        wgpuCommandBufferRelease(ps.cbPool[i]);

    // Each "token" needs nGroups CBs. Pre-record decodeCbPoolBatch tokens worth.
    int totalCBs = decodeCbPoolBatch * nGroups;
    ps.cbPool.resize(totalCBs);
    ps.cbIdx = 0;

    // Compute group boundaries: split per-slot dispatches into nGroups chunks
    auto& dispatches = ps.dispatches;
    int total = (int)dispatches.size();
    std::vector<int> groupStart(nGroups + 1);
    for (int g = 0; g <= nGroups; g++)
        groupStart[g] = g * total / nGroups;

    auto encodeGroup = [&](int gBegin, int gEnd, bool addCopy) -> WGPUCommandBuffer {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

        if (passPerDispatch) {
            for (int d = gBegin; d < gEnd; d++) {
                auto& di = dispatches[d];
                WGPUComputePassDescriptor cpD{};
                auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
                wgpuComputePassEncoderSetPipeline(pass, di.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, di.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, di.gx, di.gy, di.gz);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
            }
        } else {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            for (int d = gBegin; d < gEnd; d++) {
                auto& di = dispatches[d];
                wgpuComputePassEncoderSetPipeline(pass, di.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, di.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, di.gx, di.gy, di.gz);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        if (addCopy) {
            int ringDepth = std::max(1, decodePoolDepth);
            int nextSlot = (slot + 1) % ringDepth;
            if (nextSlot < (int)pool.size() && pool[nextSlot].tokenInBuf.handle) {
                wgpuCommandEncoderCopyBufferToBuffer(enc, ps.tokenOutBuf.handle, 0,
                                                     pool[nextSlot].tokenInBuf.handle, 0, 4);
            }
            wgpuCommandEncoderCopyBufferToBuffer(enc, ps.tokenOutBuf.handle, 0,
                                                ps.stagingBuf, 0, 4);
        }

        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuCommandEncoderRelease(enc);
        return cb;
    };

    for (int i = 0; i < decodeCbPoolBatch; i++) {
        for (int g = 0; g < nGroups; g++) {
            bool isLast = (g == nGroups - 1);
            ps.cbPool[i * nGroups + g] = encodeGroup(
                groupStart[g], groupStart[g + 1], isLast);
        }
    }
}

void ModelRunner::refillKnownPrefillCBPool(int slot) {
    auto& ps = pool[slot];
    for (int i = ps.knownPrefillCbIdx;
         i < (int)ps.knownPrefillCBPool.size(); i++)
        if (ps.knownPrefillCBPool[i])
            wgpuCommandBufferRelease(ps.knownPrefillCBPool[i]);
    ps.knownPrefillCBPool.assign(decodeCbPoolBatch, nullptr);
    ps.knownPrefillCbIdx = 0;

    int stop = (int)ps.dispatches.size();
    for (int i = 0; i < (int)ps.dispatches.size(); i++) {
        if (ps.dispatches[i].name == "final_rms") { stop = i; break; }
    }

    for (int batch = 0; batch < decodeCbPoolBatch; batch++) {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        if (passPerDispatch) {
            for (int d = 0; d < stop; d++) {
                auto& di = ps.dispatches[d];
                WGPUComputePassDescriptor cpD{};
                auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
                wgpuComputePassEncoderSetPipeline(pass, di.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, di.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, di.gx, di.gy, di.gz);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
            }
        } else {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            for (int d = 0; d < stop; d++) {
                auto& di = ps.dispatches[d];
                wgpuComputePassEncoderSetPipeline(pass, di.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, di.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, di.gx, di.gy, di.gz);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
        WGPUCommandBufferDescriptor cbD{};
        ps.knownPrefillCBPool[batch] = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuCommandEncoderRelease(enc);
    }
}

void ModelRunner::autotuneDecodeDepth() {
    if (!gpu || gpu->backendType != WGPUBackendType_D3D12) return;
    // Qwen 3.5 interleaves SSM and full-attention layers whose recurrent state
    // is mutated by benchmarkDecodeConfig. Replaying several speculative token
    // streams during startup leaves that state inconsistent and caused a
    // cross-vendor 0xC0000005 immediately after autotuning. Keep the warmed,
    // hardware-selected pool depth until the tuner can snapshot/restore every
    // Qwen 3.5 recurrent buffer between candidates.
    if (cfg.arch == "qwen35") return;
    if (decodePoolCapacity < 3) return;

    const int originalDepth = decodePoolDepth;
    const int maxDepth = std::min(decodePoolCapacity, 8);
    const int nTokens = 24;
    const double epsilon = 0.015;
    double bestMsPerTok = 1e30;
    int bestDepth = decodePoolDepth;

    for (int depth = 2; depth <= maxDepth; depth++) {
        double msPerTok = benchmarkDecodeConfig(depth, nTokens, 2);
        if (msPerTok < bestMsPerTok * (1.0 - epsilon) ||
            (fabs(msPerTok - bestMsPerTok) <= bestMsPerTok * epsilon && depth < bestDepth)) {
            bestMsPerTok = msPerTok;
            bestDepth = depth;
        }
    }

    decodePoolDepth = bestDepth;
    for (int s = 0; s < decodePoolDepth; s++)
        refillCBPool(s);
    resetKVCache();

    fprintf(stderr, "  Decode depth autotune: %d -> %d (%.2f ms/tok)\n",
           originalDepth, decodePoolDepth, bestMsPerTok);
}

double ModelRunner::benchmarkDecodeConfig(int depth, int nTokens, int repeats) {
    int32_t seedToken = 0;
    double totalMsPerTok = 0.0;
    for (int rep = 0; rep < repeats; rep++) {
        resetKVCache();
        seedDecodeTokenInputs(seedToken);
        for (int s = 0; s < depth; s++)
            refillCBPool(s);

        auto t0 = std::chrono::steady_clock::now();
        int submitted = 0;
        int completed = 0;
        int primeCount = std::min(depth, nTokens);
        for (int i = 0; i < primeCount; i++) {
            submitDecode((uint32_t)i, i);
            submitted++;
        }
        while (completed < submitted) {
            int slot = completed % depth;
            (void)readArgmax(slot);
            completed++;
            if (submitted < nTokens) {
                submitDecode((uint32_t)submitted, slot);
                submitted++;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        totalMsPerTok += std::chrono::duration<double, std::milli>(t1 - t0).count() / nTokens;
    }
    return totalMsPerTok / std::max(repeats, 1);
}

void ModelRunner::applyDecodeKernelSelection(bool useFastQkv, bool useFastOproj,
                                             bool useFastGateup) {
    tuning.decodeUseFastQkv = useFastQkv && decodeFastVariantsAvailable;
    tuning.decodeUseFastOproj = useFastOproj && decodeFastVariantsAvailable;
    tuning.decodeUseFastGateup = useFastGateup && decodeFastVariantsAvailable;

    const auto& plQ8Matmul = getKernel("q8_matmul");
    const auto& plQ8MatmulNorm = getKernel("q8_matmul_norm");
    const auto& plQ8Fast = getKernel("q8_matmul_fast");
    for (uint32_t i = 0; i < cfg.nLayer; i++) {
        auto& di = decodeDispatchIndices[i];
        auto& vbg = decodeVariantBGs[i];

        if (di.qkv >= 0 && vbg.qkvBase) {
            auto& d = allDecodeDispatches[di.qkv];
            d.pipeline = tuning.decodeUseFastQkv && vbg.qkvFast ? plQ8Fast.pipeline : plQ8MatmulNorm.pipeline;
            d.bindGroup = tuning.decodeUseFastQkv && vbg.qkvFast ? vbg.qkvFast : vbg.qkvBase;
            autoDecodeDispatches[di.qkv + autoDecodePrefixCount] = d;
        }
        if (di.oproj >= 0 && vbg.oprojBase) {
            auto& d = allDecodeDispatches[di.oproj];
            d.pipeline = tuning.decodeUseFastOproj && vbg.oprojFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastOproj && vbg.oprojFast ? vbg.oprojFast : vbg.oprojBase;
            autoDecodeDispatches[di.oproj + autoDecodePrefixCount] = d;
        }
        if (di.gateup >= 0 && vbg.gateupBase) {
            auto& d = allDecodeDispatches[di.gateup];
            d.pipeline = tuning.decodeUseFastGateup && vbg.gateupFast ? plQ8Fast.pipeline : plQ8Matmul.pipeline;
            d.bindGroup = tuning.decodeUseFastGateup && vbg.gateupFast ? vbg.gateupFast : vbg.gateupBase;
            autoDecodeDispatches[di.gateup + autoDecodePrefixCount] = d;
        }
    }

    for (int s = 0; s < decodePoolDepth; s++) {
        auto& ps = pool[s];
        if (!ps.dispatches.empty()) {
            for (uint32_t i = 0; i < cfg.nLayer; i++) {
                auto& di = decodeDispatchIndices[i];
                if (di.qkv >= 0 && decodeVariantBGs[i].qkvBase)
                    ps.dispatches[di.qkv + autoDecodePrefixCount] = autoDecodeDispatches[di.qkv + autoDecodePrefixCount];
                if (di.oproj >= 0 && decodeVariantBGs[i].oprojBase)
                    ps.dispatches[di.oproj + autoDecodePrefixCount] = autoDecodeDispatches[di.oproj + autoDecodePrefixCount];
                if (di.gateup >= 0 && decodeVariantBGs[i].gateupBase)
                    ps.dispatches[di.gateup + autoDecodePrefixCount] = autoDecodeDispatches[di.gateup + autoDecodePrefixCount];
            }
        }
        refillCBPool(s);
    }
}

void ModelRunner::autotuneDecodeKernels() {
    if (!gpu || gpu->backendType != WGPUBackendType_D3D12) return;
    if (!decodeFastVariantsAvailable) return;

    const int nTokens = 12;
    const double epsilon = 0.015;
    double bestMsPerTok = 1e30;
    bool bestQkv = tuning.decodeUseFastQkv;
    bool bestOproj = tuning.decodeUseFastOproj;
    bool bestGateup = tuning.decodeUseFastGateup;

    for (int mask = 0; mask < 8; mask++) {
        bool useFastQkv = (mask & 1) != 0;
        bool useFastOproj = (mask & 2) != 0;
        bool useFastGateup = (mask & 4) != 0;
        applyDecodeKernelSelection(useFastQkv, useFastOproj, useFastGateup);
        double msPerTok = benchmarkDecodeConfig(decodePoolDepth, nTokens, 1);
        int currentFastCount = (int)useFastQkv + (int)useFastOproj + (int)useFastGateup;
        int bestFastCount = (int)bestQkv + (int)bestOproj + (int)bestGateup;
        if (msPerTok < bestMsPerTok * (1.0 - epsilon) ||
            (fabs(msPerTok - bestMsPerTok) <= bestMsPerTok * epsilon && currentFastCount < bestFastCount)) {
            bestMsPerTok = msPerTok;
            bestQkv = useFastQkv;
            bestOproj = useFastOproj;
            bestGateup = useFastGateup;
        }
    }

    applyDecodeKernelSelection(bestQkv, bestOproj, bestGateup);
    resetKVCache();
    fprintf(stderr, "  Decode kernel autotune: qkv=%s oproj=%s gateup=%s (%.2f ms/tok)\n",
           bestQkv ? "fast" : "base",
           bestOproj ? "fast" : "base",
           bestGateup ? "fast" : "base",
           bestMsPerTok);
}

std::string ModelRunner::decodeAutotuneCachePath() const {
    namespace fs = std::filesystem;
    fs::path modelPath(ggufPath);
    fs::path modelDir = modelPath.parent_path();
    std::string modelName = modelDir.filename().string();
    while (modelName == "weights" || modelName == "decoder" ||
           modelName == "onnx-webgpu") {
        modelDir = modelDir.parent_path();
        modelName = modelDir.filename().string();
    }

    fs::path repoRoot;
    auto findRepo = [&](fs::path start) {
        for (auto p = fs::absolute(start); !p.empty() && p != p.parent_path();
             p = p.parent_path()) {
            for (const auto& candidate : {p, p / "backpack"}) {
                if (fs::is_directory(candidate / "gitignore") &&
                    fs::exists(candidate / "runtime" / "CMakeLists.txt")) {
                    return candidate;
                }
            }
        }
        return fs::path{};
    };
    repoRoot = findRepo(modelPath);
    if (repoRoot.empty()) repoRoot = findRepo(fs::current_path());
    if (repoRoot.empty())
        return {};

    fs::path cacheDir = repoRoot / "gitignore" / "models" / modelName;
    fs::create_directories(cacheDir);
    return (cacheDir / ("decode_autotune_" + backendName(gpu->backendType) + ".txt")).string();
}

std::string ModelRunner::decodeAutotuneCacheKey() const {
    std::ostringstream oss;
    oss << "backend=" << backendName(gpu->backendType)
        << ";gpu=" << gpu->adapterName
        << ";desc=" << gpu->adapterDescription
        << ";arch=" << cfg.arch
        << ";layers=" << cfg.nLayer
        << ";embd=" << cfg.nEmbd
        << ";head_dim=" << cfg.headDim
        << ";kv_heads=" << cfg.nKvHeads
        << ";intermediate=" << cfg.intermediateSize
        << ";native_kq=" << (weightsUseNativeKQ ? 1 : 0)
        << ";native_q4=" << (weightsAreNativeQ4 ? 1 : 0)
        << ";invocations=" << effectiveLimits(*gpu).maxComputeInvocationsPerWorkgroup
        << ";wgmem=" << effectiveLimits(*gpu).maxComputeWorkgroupStorageSize
        << ";pool_cap=" << decodePoolCapacity
        << ";batch=" << decodeCbPoolBatch;
    return oss.str();
}

bool ModelRunner::loadDecodeAutotuneCache() {
    std::string path = decodeAutotuneCachePath();
    if (path.empty()) return false;

    std::ifstream in(path);
    if (!in) return false;

    std::string key;
    std::string line;
    int cachedDepth = -1;
    int cachedQkv = -1, cachedOproj = -1, cachedGateup = -1;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string name = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        if (name == "key") key = value;
        else if (name == "depth") cachedDepth = atoi(value.c_str());
        else if (name == "qkv_fast") cachedQkv = atoi(value.c_str());
        else if (name == "oproj_fast") cachedOproj = atoi(value.c_str());
        else if (name == "gateup_fast") cachedGateup = atoi(value.c_str());
    }

    if (key != decodeAutotuneCacheKey()) return false;
    if (cachedDepth < 2 || cachedDepth > decodePoolCapacity) return false;
    if (cachedQkv < 0 || cachedOproj < 0 || cachedGateup < 0) return false;

    decodePoolDepth = cachedDepth;
    applyDecodeKernelSelection(cachedQkv != 0, cachedOproj != 0, cachedGateup != 0);
    resetKVCache();
    fprintf(stderr, "  Decode autotune cache: loaded from %s\n", path.c_str());
    return true;
}

void ModelRunner::saveDecodeAutotuneCache() const {
    std::string path = decodeAutotuneCachePath();
    if (path.empty()) return;

    std::ofstream out(path, std::ios::trunc);
    if (!out) return;
    out << "key=" << decodeAutotuneCacheKey() << "\n";
    out << "depth=" << decodePoolDepth << "\n";
    out << "qkv_fast=" << (tuning.decodeUseFastQkv ? 1 : 0) << "\n";
    out << "oproj_fast=" << (tuning.decodeUseFastOproj ? 1 : 0) << "\n";
    out << "gateup_fast=" << (tuning.decodeUseFastGateup ? 1 : 0) << "\n";
}

void ModelRunner::printActiveDecodeTuning(const char* prefix) const {
    const char* lmHeadKind = lmHeadIsKQ ? "k-quant" : lmHeadIsQ4 ? "q4"
        : (lmHeadIsQ8 ? "q8" : (tuning.decodeUseWideFp16 ? "fp16_wide" : "fp16"));
    fprintf(stderr, "%s: depth=%d/%d qkv=%s oproj=%s gateup=%s lm_head=%s batch=%d\n",
           prefix,
           decodePoolDepth,
           decodePoolCapacity,
           tuning.decodeUseFastQkv ? "fast" : "base",
           tuning.decodeUseFastOproj ? "fast" : "base",
           tuning.decodeUseFastGateup ? "fast" : "base",
           lmHeadKind,
           decodeCbPoolBatch);
}

void ModelRunner::destroy() {
    if (!gpu) return;

    std::unordered_set<void*> releasedBindGroups;
    auto releaseBG = [&](WGPUBindGroup& bg) {
        if (!bg) return;
        void* key = reinterpret_cast<void*>(bg);
        if (releasedBindGroups.insert(key).second)
            wgpuBindGroupRelease(bg);
        bg = nullptr;
    };

    for (auto& d : allDecodeDispatches)
        releaseBG(d.bindGroup);
    for (auto& d : autoDecodeDispatches)
        releaseBG(d.bindGroup);

    for (auto& bg : decodeVariantBGs) {
        releaseBG(bg.qkvBase);
        releaseBG(bg.qkvFast);
        releaseBG(bg.oprojBase);
        releaseBG(bg.oprojFast);
        releaseBG(bg.gateupBase);
        releaseBG(bg.gateupFast);
    }

    for (auto& bg : pfCache.layerBGs) {
        releaseBG(bg.rms);
        releaseBG(bg.qnorm);
        releaseBG(bg.qkv);
        releaseBG(bg.rope);
        releaseBG(bg.attn);
        releaseBG(bg.attnq);
        releaseBG(bg.oproj);
        releaseBG(bg.addrms);
        releaseBG(bg.gateup);
        releaseBG(bg.siluq);
        releaseBG(bg.downsilu);
    }
    releaseBG(pfCache.finalRmsBG);
    releaseBG(pfCache.lmBG);
    releaseBG(pfCache.argmaxBG);

    for (auto& slot : pool) {
        for (int i = slot.cbIdx; i < (int)slot.cbPool.size(); i++) {
            if (slot.cbPool[i])
                wgpuCommandBufferRelease(slot.cbPool[i]);
        }
        slot.cbPool.clear();
        slot.cbIdx = 0;
        for (int i = slot.knownPrefillCbIdx;
             i < (int)slot.knownPrefillCBPool.size(); i++) {
            if (slot.knownPrefillCBPool[i])
                wgpuCommandBufferRelease(slot.knownPrefillCBPool[i]);
        }
        slot.knownPrefillCBPool.clear();
        slot.knownPrefillCbIdx = 0;
        if (slot.stagingBuf) {
            wgpuBufferRelease(slot.stagingBuf);
            slot.stagingBuf = nullptr;
        }
        if (slot.ropeParamsBuf.handle) {
            wgpuBufferRelease(slot.ropeParamsBuf.handle);
            slot.ropeParamsBuf.handle = nullptr;
        }
        if (slot.attnParamsBuf.handle) {
            wgpuBufferRelease(slot.attnParamsBuf.handle);
            slot.attnParamsBuf.handle = nullptr;
        }
        if (slot.attnParamsBufSWA.handle) {
            wgpuBufferRelease(slot.attnParamsBufSWA.handle);
            slot.attnParamsBufSWA.handle = nullptr;
        }
        if (slot.q35RopeQParamsBuf.handle) {
            wgpuBufferRelease(slot.q35RopeQParamsBuf.handle);
            slot.q35RopeQParamsBuf.handle = nullptr;
        }
        if (slot.q35RopeKParamsBuf.handle) {
            wgpuBufferRelease(slot.q35RopeKParamsBuf.handle);
            slot.q35RopeKParamsBuf.handle = nullptr;
        }
        if (slot.q35KvWriteParamsBuf.handle) {
            wgpuBufferRelease(slot.q35KvWriteParamsBuf.handle);
            slot.q35KvWriteParamsBuf.handle = nullptr;
        }
        if (slot.tokenInBuf.handle) {
            wgpuBufferRelease(slot.tokenInBuf.handle);
            slot.tokenInBuf.handle = nullptr;
        }
        if (slot.tokenOutBuf.handle) {
            wgpuBufferRelease(slot.tokenOutBuf.handle);
            slot.tokenOutBuf.handle = nullptr;
        }
        slot.dispatches.clear();
    }
    pool.clear();

    if (profiler) {
        profiler->destroy();
        delete profiler;
        profiler = nullptr;
    }
}

// ─── Inference ───────────────────────────────────────────────────────────────

void ModelRunner::uploadEmbedding(int32_t tokenId) {
    if (tokenId < 0 || (uint32_t)tokenId >= cfg.nVocab) tokenId = 0;
    if (pleGpuPreprocess && argmaxResultBuf.handle)
        gpu->writeBuffer(argmaxResultBuf, &tokenId, 4);
    const float* emb = embeddingCPU.data() + tokenId * cfg.nEmbd;
    if (std::getenv("BP_DUMP_BUFFER_STATS")) {
        dumpFloatArrayStats("embeddingCPU[token]", emb, cfg.nEmbd);
    }
    if (cfg.embeddingScale > 0.0f) {
        std::vector<float> scaled(cfg.nEmbd);
        for (uint32_t i = 0; i < cfg.nEmbd; i++)
            scaled[i] = emb[i] * cfg.embeddingScale;
        gpu->writeBuffer(xBuf, scaled.data(), cfg.nEmbd * 4);
    } else {
        gpu->writeBuffer(xBuf, emb, cfg.nEmbd * 4);
    }

    // PLE: compute per-layer input for this token, matching Gemma 4's
    // project_per_layer_inputs:
    //   proj = per_layer_model_proj(embedding)           # [nLayer*pleDim]
    //   proj = RMSNorm_per_slice(proj, per_layer_proj_norm)
    //   tok  = per_layer_token_embd[token] * sqrt(pleDim)
    //   per_layer_input = (proj + tok) / sqrt(2)
    if (cfg.pleSize > 0 && !pleEmbCPU.empty() && pleInputBuf.handle) {
        uint32_t pleDim = cfg.pleSize;
        uint32_t totalPleDim = pleDim * cfg.nLayer;
        float pleScale = sqrtf((float)pleDim);

        std::vector<float> perLayerInput(totalPleDim, 0.0f);

        // token-side signal: per_layer_token_embd[token] * sqrt(pleDim)
        if ((size_t)tokenId * totalPleDim + totalPleDim <= pleEmbCPU.size()) {
            for (uint32_t j = 0; j < totalPleDim; j++)
                perLayerInput[j] = pleEmbCPU[(size_t)tokenId * totalPleDim + j] * pleScale;
        }

        // CPU reference projection. This retains the native Q4 values; the Q8
        // GPU projection is observably too lossy for Gemma 4's close logits.
        if (pleModelProjCPU.size() == (size_t)totalPleDim * cfg.nEmbd) {
            const float* embSrc = embeddingCPU.data() + (size_t)tokenId * cfg.nEmbd;
            const float invSqrt2 = 0.70710678f;
            float eps = cfg.rmsNormEps;
            std::vector<float> proj(totalPleDim);
            std::vector<uint32_t> layers(cfg.nLayer);
            std::iota(layers.begin(), layers.end(), 0u);
            std::for_each(std::execution::par, layers.begin(), layers.end(), [&](uint32_t li) {
                float* projSlice = proj.data() + (size_t)li * pleDim;
                for (uint32_t r = 0; r < pleDim; r++) {
                    const float* w = pleModelProjCPU.data() +
                        (size_t)(li * pleDim + r) * cfg.nEmbd;
                    float acc = 0.0f;
                    for (uint32_t k = 0; k < cfg.nEmbd; k++) acc += w[k] * embSrc[k];
                    projSlice[r] = acc;
                }
                float ss = 0.0f;
                for (uint32_t r = 0; r < pleDim; r++) ss += projSlice[r] * projSlice[r];
                float rms = 1.0f / sqrtf(ss / (float)pleDim + eps);
                for (uint32_t r = 0; r < pleDim; r++) {
                    float w = (r < pleProjNormCPU.size()) ? pleProjNormCPU[r] : 1.0f;
                    float projNormed = projSlice[r] * rms * w;
                    perLayerInput[li * pleDim + r] =
                        (projNormed + perLayerInput[li * pleDim + r]) * invSqrt2;
                }
            });
        }

        // One concatenated upload replaces 35 per-layer queue writes.
        gpu->writeBuffer(pleInputBuf, perLayerInput.data(), totalPleDim * 4);
    }
}

void ModelRunner::updateDecodeParams(uint32_t pos, uint32_t cacheLen) {
    auto* p = reinterpret_cast<int32_t*>(ropeParamData.data());
    p[3] = pos;
    p[5] = cacheLen * cfg.nKvHeads * cfg.headDim;
    if (decodeUsesFusedRopeParams) {
        gpu->writeBuffer(fusedRopeParamsBuf, ropeParamData.data(), 32);
    }

    if (q35RopeQParamsBuf.handle) {
        uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        uint32_t q[8] = {cfg.nHead, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], pos, ropeHalf};
        uint32_t k[8] = {cfg.nKvHeads, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], pos, ropeHalf};
        uint32_t kv[8] = {cfg.nKvHeads * cfg.headDim, cacheLen * cfg.nKvHeads * cfg.headDim, 0, 0, 0, 0, 0, 0};
        gpu->writeBuffer(q35RopeQParamsBuf, q, 32);
        gpu->writeBuffer(q35RopeKParamsBuf, k, 32);
        gpu->writeBuffer(q35KvWriteParamsBuf, kv, 32);
    }

    uint32_t T_total = cacheLen + 1;
    uint32_t n_chunks = (T_total + gqaChunkSize - 1) / gqaChunkSize;
    auto* cp = reinterpret_cast<uint32_t*>(chunkedAttnParamData.data());
    cp[2] = T_total;
    cp[3] = 0;
    cp[4] = n_chunks;
    gpu->writeBuffer(chunkedAttnParamsBuf, chunkedAttnParamData.data(), 32);

    // SWA: clamp T_total to sliding window size
    if (chunkedAttnParamsBufSWA.handle && cfg.slidingWindow > 0) {
        uint32_t T_swa = std::min(T_total, cfg.slidingWindow);
        uint32_t n_chunks_swa = (T_swa + gqaChunkSize - 1) / gqaChunkSize;
        uint8_t swaData[32];
        memcpy(swaData, chunkedAttnParamData.data(), 32);
        auto* sp = reinterpret_cast<uint32_t*>(swaData);
        sp[2] = T_swa;
        sp[3] = T_total - T_swa;
        sp[4] = n_chunks_swa;
        gpu->writeBuffer(chunkedAttnParamsBufSWA, swaData, 32);
    }

    // Variable head dims (Gemma 4): refresh each layer's own rope + attention
    // params with the layer's head dim, rope half-dim, KV stride and scale.
    if (hasVariableHeadDim) {
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            uint32_t hdL = cfg.perLayer[li].headDim;
            uint32_t qDimL = cfg.nHead * hdL;
            uint32_t kvDimL = cfg.nKvHeads * hdL;
            uint32_t ropeHalfL = layerRotaryDim(li) / 2;
            // rope params: [nHead, qDim, kvDim, pos, ropeHalf, kvWriteOff, eps, 0]
            uint8_t rd[32];
            memcpy(rd, ropeParamData.data(), 32);
            auto* rp = reinterpret_cast<int32_t*>(rd);
            rp[0] = (int)cfg.nHead; rp[1] = (int)qDimL; rp[2] = (int)kvDimL;
            rp[3] = (int)pos; rp[4] = (int)ropeHalfL;
            rp[5] = (int)(cacheLen * kvDimL);
            gpu->writeBuffer(perLayerRopeParamBufs[li], rd, 32);

            // attn params: [kvStride, n_rep, T, kvStart, n_chunks, scale, -inf, maxChunks]
            bool isSWA = !cfg.layerAttnTypes.empty() && li < cfg.layerAttnTypes.size() &&
                         cfg.layerAttnTypes[li] == AttnLayerType::SlidingWindow;
            uint32_t Tl = (isSWA && cfg.slidingWindow > 0)
                        ? std::min(T_total, cfg.slidingWindow) : T_total;
            uint32_t nchL = (Tl + gqaChunkSize - 1) / gqaChunkSize;
            uint8_t ad[32];
            auto* ap = reinterpret_cast<uint32_t*>(ad);
            ap[0] = kvDimL;
            ap[1] = cfg.nHead / cfg.nKvHeads;
            ap[2] = Tl; ap[3] = isSWA ? T_total - Tl : 0u; ap[4] = nchL;
            float scaleL = cfg.arch == "gemma4" ? 1.0f
                                                  : 1.0f / sqrtf((float)hdL);
            float neg_inf = -1e9f;
            memcpy(&ap[5], &scaleL, 4);
            memcpy(&ap[6], &neg_inf, 4);
            ap[7] = reinterpret_cast<const uint32_t*>(chunkedAttnParamData.data())[7];
            gpu->writeBuffer(perLayerAttnParamBufs[li], ad, 32);
        }
    }
}

void ModelRunner::prepareDecodeParams(uint32_t pos, uint32_t cacheLen, int slot) {
    auto& ps = pool[slot];

    uint8_t localRope[32], localAttn[32];
    memcpy(localRope, ropeParamData.data(), 32);
    auto* p = reinterpret_cast<int32_t*>(localRope);
    p[3] = pos;
    p[5] = cacheLen * cfg.nKvHeads * cfg.headDim;
    gpu->writeBuffer(ps.ropeParamsBuf, localRope, 32);

    uint32_t T_total = cacheLen + 1;
    uint32_t n_chunks = (T_total + gqaChunkSize - 1) / gqaChunkSize;
    memcpy(localAttn, chunkedAttnParamData.data(), 32);
    auto* cp = reinterpret_cast<uint32_t*>(localAttn);
    cp[2] = T_total;
    cp[3] = 0;
    cp[4] = n_chunks;
    gpu->writeBuffer(ps.attnParamsBuf, localAttn, 32);

    if (ps.q35RopeQParamsBuf.handle) {
        uint32_t ropeHalf = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        uint32_t q[8] = {cfg.nHead, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], pos, ropeHalf};
        uint32_t k[8] = {cfg.nKvHeads, cfg.headDim, (uint32_t)cfg.ropeSections[0], (uint32_t)cfg.ropeSections[1], (uint32_t)cfg.ropeSections[2], (uint32_t)cfg.ropeSections[3], pos, ropeHalf};
        uint32_t kv[8] = {cfg.nKvHeads * cfg.headDim,
                          cacheLen * cfg.nKvHeads * cfg.headDim,
                          0, 0, 0, 0, 0, 0};
        gpu->writeBuffer(ps.q35RopeQParamsBuf, q, 32);
        gpu->writeBuffer(ps.q35RopeKParamsBuf, k, 32);
        gpu->writeBuffer(ps.q35KvWriteParamsBuf, kv, 32);
    }

    // SWA per-slot param buffers
    if (ps.attnParamsBufSWA.handle && cfg.slidingWindow > 0) {
        uint32_t T_swa = std::min(T_total, cfg.slidingWindow);
        uint32_t n_chunks_swa = (T_swa + gqaChunkSize - 1) / gqaChunkSize;
        uint8_t swaAttn[32];
        memcpy(swaAttn, chunkedAttnParamData.data(), 32);
        auto* sp = reinterpret_cast<uint32_t*>(swaAttn);
        sp[2] = T_swa;
        sp[3] = T_total - T_swa;
        sp[4] = n_chunks_swa;
        gpu->writeBuffer(ps.attnParamsBufSWA, swaAttn, 32);
    }

    if (hasVariableHeadDim) {
        constexpr uint32_t kParamStride = 256;
        std::vector<uint8_t> ropePacked(cfg.nLayer * kParamStride, 0);
        std::vector<uint8_t> attnPacked(cfg.nLayer * kParamStride, 0);
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            uint32_t hdL = cfg.perLayer[li].headDim;
            uint32_t kvDimL = cfg.nKvHeads * hdL;
            uint8_t rd[32];
            memcpy(rd, ropeParamData.data(), 32);
            auto* rp = reinterpret_cast<int32_t*>(rd);
            rp[0] = (int)cfg.nHead;
            rp[1] = (int)(cfg.nHead * hdL);
            rp[2] = (int)kvDimL;
            rp[3] = (int)pos;
            rp[4] = (int)(layerRotaryDim(li) / 2);
            rp[5] = (int)(cacheLen * kvDimL);
            memcpy(ropePacked.data() + li * kParamStride, rd, 32);

            bool isSWA = li < cfg.layerAttnTypes.size() &&
                         cfg.layerAttnTypes[li] == AttnLayerType::SlidingWindow;
            uint32_t Tl = isSWA && cfg.slidingWindow > 0
                ? std::min(T_total, cfg.slidingWindow) : T_total;
            uint32_t ad[8] = {kvDimL, cfg.nHead / cfg.nKvHeads, Tl,
                              isSWA ? T_total - Tl : 0u, (Tl + gqaChunkSize - 1) / gqaChunkSize,
                              0, 0, reinterpret_cast<const uint32_t*>(chunkedAttnParamData.data())[7]};
            float scaleL = cfg.arch == "gemma4" ? 1.0f : 1.0f / sqrtf((float)hdL);
            float negInf = -1e9f;
            memcpy(&ad[5], &scaleL, 4);
            memcpy(&ad[6], &negInf, 4);
            memcpy(attnPacked.data() + li * kParamStride, ad, 32);
        }
        gpu->writeBuffer(ps.layerRopeParamsPacked, ropePacked.data(), ropePacked.size());
        gpu->writeBuffer(ps.layerAttnParamsPacked, attnPacked.data(), attnPacked.size());
    }

}

std::vector<float> ModelRunner::decode(int32_t tokenId, uint32_t posOffset) {
    bool gatherEmbeddingOnGpu = embeddingCPU.empty();
    if (gatherEmbeddingOnGpu)
        seedDecodeTokenInputs(tokenId);
    else
        uploadEmbedding(tokenId);

    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    const uint32_t stopDispatchPos = std::getenv("BP_STOP_POS")
        ? (uint32_t)std::max(0, std::atoi(std::getenv("BP_STOP_POS"))) : 0u;
    if (const char* stopDispatch = std::getenv("BP_STOP_AFTER_DISPATCH");
        stopDispatch && posOffset == stopDispatchPos) {
        size_t end = allDecodeDispatches.size();
        for (size_t j = 0; j < allDecodeDispatches.size(); ++j) {
            if (allDecodeDispatches[j].name == stopDispatch) { end = j + 1; break; }
        }
        if (end < allDecodeDispatches.size()) {
            const char* which = std::getenv("BP_STOP_BUFFER");
            std::string bufferName = which ? which : "x";
            GPUBuffer buffer = xBuf;
            uint32_t count = cfg.nEmbd;
            uint32_t diagnosticIM = cfg.intermediateSize;
            if (stopDispatch[0] == 'L') {
                const uint32_t li = (uint32_t)std::max(0, std::atoi(stopDispatch + 1));
                if (li < cfg.perLayer.size()) diagnosticIM = cfg.perLayer[li].intermediateSize;
            }
            if (bufferName == "qkv") { buffer = qkvBuf; count = qkvOut; }
            else if (bufferName == "qrot") { buffer = qRotBuf; count = qDim; }
            else if (bufferName == "attn") { buffer = attnOutBuf; count = qDim; }
            else if (bufferName == "proj") { buffer = projOutBuf; count = cfg.nEmbd; }
            else if (bufferName == "norm") { buffer = normOutBuf; count = cfg.nEmbd; }
            else if (bufferName == "gateup") { buffer = gateUpBuf; count = 2 * diagnosticIM; }
            else if (bufferName == "silu") { buffer = siluMulDebugBuf; count = diagnosticIM; }
            else if (bufferName == "ple") { buffer = pleBuf; count = cfg.pleSize; }
            else if (bufferName == "pleinput") { buffer = pleInputBuf; count = cfg.pleSize * cfg.nLayer; }
            else if (bufferName.rfind("kvv", 0) == 0) {
                uint32_t li = (uint32_t)std::max(0, std::atoi(bufferName.c_str() + 3));
                if (li < kvCache.size()) {
                    buffer = kvCache[li].V;
                    count = cfg.perLayer[li].kvDim;
                }
            }
            else if (bufferName == "pleout") { buffer = pleOutBuf; count = cfg.nEmbd; }
            else if (bufferName == "logits") { buffer = logitsBuf; count = cfg.nVocab; }
            std::vector<Dispatch> partial(allDecodeDispatches.begin(), allDecodeDispatches.begin() + end);
            auto bytes = gpu->submitAndReadback(partial, buffer, count * 4, passPerDispatch);
            if (bufferName.rfind("kvv", 0) == 0) {
                auto raw = gpu->readBuffer(buffer, count * 2);
                const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data());
                auto h2f = [](uint16_t x) {
                    uint32_t s=(x>>15)&1,e=(x>>10)&31,m=x&1023,out;
                    if(e==0) out=(s<<31)|(m<<13);
                    else if(e==31) out=(s<<31)|0x7f800000|(m<<13);
                    else out=(s<<31)|((e+112)<<23)|(m<<13);
                    float f; memcpy(&f,&out,4); return f;
                };
                fprintf(stderr, "[dispatch-dump] after=%s buffer=%s count=%u first8=",
                        stopDispatch, bufferName.c_str(), count);
                for (uint32_t j=0;j<std::min(8u,count);j++)
                    fprintf(stderr,"%s%.9g",j?",":"",h2f(h[j]));
                fprintf(stderr,"\n"); std::exit(0);
            }
            const float* values = reinterpret_cast<const float*>(bytes.data());
            double ss = 0.0; uint32_t nonfinite = 0;
            uint32_t firstNonfinite = count;
            for (uint32_t j = 0; j < count; ++j) {
                if (!std::isfinite(values[j])) {
                    if (firstNonfinite == count) firstNonfinite = j;
                    ++nonfinite;
                }
                else ss += (double)values[j] * values[j];
            }
            fprintf(stderr, "[dispatch-dump] after=%s buffer=%s count=%u nonfinite=%u first_nonfinite=%s norm=%.9g first8=",
                    stopDispatch, bufferName.c_str(), count, nonfinite,
                    firstNonfinite == count ? "none" : std::to_string(firstNonfinite).c_str(), sqrt(ss));
            for (uint32_t j = 0; j < std::min(8u, count); ++j)
                fprintf(stderr, "%s%.9g", j ? "," : "", values[j]);
            fprintf(stderr, "\n"); fflush(stderr);
            if (bufferName == "silu" && firstNonfinite < count) {
                auto gateBytes = gpu->readBuffer(gateUpBuf, 2 * count * 4);
                const float* gateUp = reinterpret_cast<const float*>(gateBytes.data());
                fprintf(stderr, "[dispatch-dump] silu_bad_index=%u gate=%.9g up=%.9g\n",
                        firstNonfinite, gateUp[firstNonfinite], gateUp[count + firstNonfinite]);
            }
            std::exit(0);
        }
    }

    const char* stopEnv = std::getenv("BP_STOP_AFTER_LAYER");
    uint32_t stopPos = std::getenv("BP_STOP_POS")
        ? (uint32_t)std::max(0, std::atoi(std::getenv("BP_STOP_POS"))) : 0u;
    if (stopEnv && tokenId != 0 && posOffset == stopPos) {
        int stopLayer = std::max(0, std::atoi(stopEnv));
        std::string prefix = "L" + std::to_string(stopLayer + 1) + "/";
        size_t end = allDecodeDispatches.size();
        for (size_t j = 0; j < allDecodeDispatches.size(); j++) {
            if (allDecodeDispatches[j].name.rfind(prefix, 0) == 0) {
                end = j; break;
            }
        }
        std::vector<Dispatch> partial(allDecodeDispatches.begin(),
                                      allDecodeDispatches.begin() + end);
        auto bytes = gpu->submitAndReadback(partial, xBuf, cfg.nEmbd * 4,
                                             passPerDispatch);
        const float* x = reinterpret_cast<const float*>(bytes.data());
        double ss = 0.0;
        for (uint32_t j = 0; j < cfg.nEmbd; j++) ss += (double)x[j] * x[j];
        fprintf(stderr, "[layer-dump] layer=%d token=%d pos=%u norm=%.9g first8=",
                stopLayer, tokenId, posOffset, sqrt(ss));
        for (uint32_t j = 0; j < std::min(8u, cfg.nEmbd); j++)
            fprintf(stderr, "%s%.9g", j ? "," : "", x[j]);
        fprintf(stderr, "\n"); fflush(stderr);
        std::exit(0);
    }

    std::vector<uint8_t> result;
    const auto& decodePlan = gatherEmbeddingOnGpu ? autoDecodeDispatches
                                                   : allDecodeDispatches;
    if (profiler && profiler->enabled()) {
        result = gpu->submitAndReadbackProfiled(
            decodePlan, logitsBuf, cfg.nVocab * 4, *profiler);
    } else {
        result = gpu->submitAndReadback(
            decodePlan, logitsBuf, cfg.nVocab * 4, passPerDispatch);
    }

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    if (std::getenv("BP_DUMP_BUFFER_STATS")) {
        dumpFloatBufferStats(*gpu, "xBuf", xBuf, cfg.nEmbd);
        dumpFloatBufferStats(*gpu, "normOutBuf", normOutBuf, cfg.nEmbd);
        dumpFloatBufferStats(*gpu, "logitsBuf", logitsBuf, cfg.nVocab);
    }

    if (std::getenv("BP_DUMP_KV")) {
        // Dump the first element of the first few token slots of layer 0's K
        // cache, to verify serial prefill writes distinct per-token K/V.
        uint32_t kvStride = cfg.nKvHeads * cfg.headDim;  // elements per token slot
        uint32_t nSlots = std::min<uint32_t>(kvCache[0].len, 8);
        auto bytes = gpu->readBuffer(kvCache[0].K,
                                     (uint64_t)nSlots * kvStride * 2ull);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(bytes.data());
        auto h2f = [](uint16_t x) -> float {
            uint32_t s = (x >> 15) & 1, e = (x >> 10) & 0x1F, m = x & 0x3FF, out;
            if (e == 0) out = (s << 31) | ((m) << 13);
            else if (e == 31) out = (s << 31) | 0x7F800000 | (m << 13);
            else out = (s << 31) | ((e + 112) << 23) | (m << 13);
            float f; memcpy(&f, &out, 4); return f;
        };
        fprintf(stderr, "[kv] layer0 K pos=%u len=%u stride=%u:", posOffset,
                kvCache[0].len, kvStride);
        for (uint32_t t = 0; t < nSlots; t++)
            fprintf(stderr, " t%u[%.3f %.3f %.3f]", t,
                    h2f(h[t * kvStride + 0]), h2f(h[t * kvStride + 1]),
                    h2f(h[t * kvStride + 2]));
        fprintf(stderr, "\n");
    }

    if (std::getenv("BP_DUMP_ATTN")) {
        dumpFloatBufferStats(*gpu, "attnOutBuf", attnOutBuf, cfg.headDim * cfg.nHead);
    }

    if (std::getenv("BP_DUMP_NORMW") && posOffset == 0) {
        auto& l0 = layerWeights[0];
        if (l0.inputNorm.handle) dumpFloatBufferStats(*gpu, "inputNorm.w", l0.inputNorm, cfg.nEmbd);
        if (l0.qNorm.handle)     dumpFloatBufferStats(*gpu, "qNorm.w", l0.qNorm, cfg.headDim);
        if (l0.kNorm.handle)     dumpFloatBufferStats(*gpu, "kNorm.w", l0.kNorm, cfg.headDim);
        if (l0.postNorm.handle)  dumpFloatBufferStats(*gpu, "postNorm.w", l0.postNorm, cfg.nEmbd);
    }

    std::vector<float> logits(cfg.nVocab);
    memcpy(logits.data(), result.data(), cfg.nVocab * 4);

    return logits;
}

int32_t ModelRunner::decodeArgmax(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);

    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    auto result = gpu->submitAndReadback(
        allDecodeDispatches, argmaxResultBuf, 4, passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    int32_t token;
    memcpy(&token, result.data(), 4);
    return token;
}

int32_t ModelRunner::decodeArgmaxPooled(int32_t tokenId, uint32_t posOffset) {
    // Unlike throughput benchmarking, generation has a true token dependency:
    // compute the token-specific embedding/PLE first, then reuse exactly one
    // pre-recorded slot and wait for its four-byte argmax result.
    auto t0 = std::chrono::steady_clock::now();
    uploadEmbedding(tokenId);
    auto t1 = std::chrono::steady_clock::now();
    seedDecodeTokenInputs(tokenId);
    submitDecode(posOffset, 0);
    auto t2 = std::chrono::steady_clock::now();
    int32_t result = readArgmax(0);
    auto t3 = std::chrono::steady_clock::now();
    if (std::getenv("BP_PROFILE_CPU")) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr, "[decode-cpu] embedding+PLE=%.2fms prepare+submit=%.2fms wait=%.2fms\n",
                ms(t0, t1), ms(t1, t2), ms(t2, t3));
    }
    return result;
}

int32_t ModelRunner::prefillPooledKnown(const int32_t* tokenIds, uint32_t count,
                                        uint32_t posOffset) {
    if (count == 0 || pool.empty()) return -1;
    auto& ps = pool[0];
    for (uint32_t t = 0; t < count; t++) {
        int32_t token = tokenIds[t];
        if (token < 0 || (uint32_t)token >= cfg.nVocab) token = 0;
        if (cfg.pleSize > 0 && !pleGpuPreprocess) uploadEmbedding(token);
        gpu->writeBuffer(ps.tokenInBuf, &token, 4);

        uint32_t cacheLen = kvCache[0].len;
        prepareDecodeParams(posOffset + t, cacheLen, 0);
        if (t + 1 < count) {
            if (ps.knownPrefillCbIdx >= (int)ps.knownPrefillCBPool.size())
                refillKnownPrefillCBPool(0);
            WGPUCommandBuffer cb = ps.knownPrefillCBPool[ps.knownPrefillCbIdx++];
            wgpuQueueSubmit(gpu->queue, 1, &cb);
            wgpuCommandBufferRelease(cb);
        } else {
            if (ps.cbIdx >= (int)ps.cbPool.size()) refillCBPool(0);
            for (int g = 0; g < nGroups; g++) {
                WGPUCommandBuffer cb = ps.cbPool[ps.cbIdx++];
                wgpuQueueSubmit(gpu->queue, 1, &cb);
                wgpuCommandBufferRelease(cb);
            }
        }
        for (uint32_t i = 0; i < cfg.nLayer; i++) kvCache[i].len++;
    }

    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void*) {};
    auto future = wgpuBufferMapAsync(ps.stagingBuf, 1, 0, 4, mcb);
    return gpu->completeAsyncMapI32(ps.stagingBuf, future);
}

void ModelRunner::submitDecode(uint32_t posOffset, int slot) {
    auto& ps = pool[slot];
    uint32_t cacheLen = kvCache[0].len;
    prepareDecodeParams(posOffset, cacheLen, slot);

    // Refill pool if exhausted
    if (ps.cbIdx >= (int)ps.cbPool.size())
        refillCBPool(slot);

    using hrc = std::chrono::high_resolution_clock;
    auto t0 = hrc::now();

    // Submit nGroups pre-recorded CBs for this token
    for (int g = 0; g < nGroups; g++) {
        WGPUCommandBuffer cb = ps.cbPool[ps.cbIdx++];
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
    }

    auto t1 = hrc::now();
    gpu->timing.submit_ns += (t1 - t0).count();

    // Start async map (non-blocking until WaitAny is called)
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void*) {};
    ps.pendingFuture = wgpuBufferMapAsync(ps.stagingBuf, 1, 0, 4, mcb);

    auto t2 = hrc::now();
    gpu->timing.map_start_ns += (t2 - t1).count();
    gpu->timing.count++;

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;
}

int32_t ModelRunner::readArgmax(int slot) {
    auto& ps = pool[slot];
    return gpu->completeAsyncMapI32(ps.stagingBuf, ps.pendingFuture);
}

void ModelRunner::seedDecodeTokenInputs(int32_t tokenId) {
    if (argmaxResultBuf.handle) {
        gpu->writeBuffer(argmaxResultBuf, &tokenId, 4);
    }
    for (auto& ps : pool) {
        if (ps.tokenInBuf.handle)
            gpu->writeBuffer(ps.tokenInBuf, &tokenId, 4);
    }
}

void ModelRunner::prefillStep(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);
    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    gpu->submitOnly(allDecodeDispatches, !passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;
}

std::vector<float> ModelRunner::prefillFinish(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);
    uint32_t cacheLen = kvCache[0].len;
    updateDecodeParams(posOffset, cacheLen);

    // Submit and read back logits (blocking — only for the last prefill token)
    auto result = gpu->submitAndReadback(
        allDecodeDispatches, logitsBuf, cfg.nVocab * 4, passPerDispatch);

    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;

    std::vector<float> logits(cfg.nVocab);
    memcpy(logits.data(), result.data(), cfg.nVocab * 4);
    return logits;
}

int32_t ModelRunner::prefillQwen35Batched(
        const int32_t* tokenIds,uint32_t T,uint32_t posOffset){
    if(!qwen35Pf.ready||T==0)return -1;
    const bool traceQpf=std::getenv("BP_PROFILE_QWEN_PREFILL")!=nullptr;
    if(traceQpf){fprintf(stderr,"[qwen-prefill] enter T=%u\n",T);fflush(stderr);}
    int32_t result=-1;uint32_t done=0;
    while(done<T){
        uint32_t M=std::min(qwen35Pf.capacity,T-done),E=cfg.nEmbd;
        std::vector<int32_t> toks(M);for(uint32_t i=0;i<M;i++){int32_t v=tokenIds[done+i];toks[i]=(v>=0&&(uint32_t)v<cfg.nVocab)?v:0;}
        gpu->writeBuffer(qwen35Pf.tokens,toks.data(),M*4);
        if(traceQpf){fprintf(stderr,"[qwen-prefill] tokens uploaded M=%u\n",M);fflush(stderr);}
        std::vector<Dispatch> ds;std::vector<WGPUBindGroup>bgs;
        uint64_t paramCursor=0;
        std::vector<uint8_t> paramHost(qwen35Pf.paramArena.size,0);
        auto mkp=[&](const std::string&n,std::initializer_list<uint32_t>v,bool uniform=false){
            (void)n;(void)uniform;size_t cnt=v.size();uint64_t bytes=cnt<=4?16:((cnt*4+15)/16)*16;
            if(paramCursor+256>qwen35Pf.paramArena.size){fprintf(stderr,"Qwen prefill parameter arena exhausted\n");std::abort();}
            std::vector<uint32_t>d(bytes/4,0);size_t i=0;for(auto x:v)d[i++]=x;
            GPUBuffer b=qwen35Pf.paramArena;b.offset=paramCursor;b.size=bytes;
            memcpy(paramHost.data()+paramCursor,d.data(),bytes);paramCursor+=256;return b;};
        auto add=[&](const CompiledPipeline&pl,std::initializer_list<std::pair<uint32_t,GPUBuffer>>bind,uint32_t x,uint32_t y,uint32_t z,const std::string&n){
            auto bg=makeBG(pl,std::vector<std::pair<uint32_t,GPUBuffer>>(bind));bgs.push_back(bg);ds.push_back({pl.pipeline,bg,x,y,z,n});
        };
        uint32_t eb;memcpy(&eb,&cfg.rmsNormEps,4);uint32_t cacheLen=kvCache[0].len;
        auto&gather=getKernel("q6k_gather_batched");auto ep=mkp("qpf_ep",{M,E,kqLmRowStride});
        add(gather,{{0,lmHeadKQ},{1,qwen35Pf.tokens},{2,qwen35Pf.x},{3,ep}},(M*E+255)/256,1,1,"qpf_embed");
        if(traceQpf){fprintf(stderr,"[qwen-prefill] embed encoded\n");fflush(stderr);}
        auto&rms=getKernel("rms_norm_batched");auto&q8=getKernel("q8_matmul_batched_dp4a");
        auto&addnorm=getKernel("add_rms_norm_batched");auto&normadd=getKernel("gemma_norm_add_batched");
        auto&addip=getKernel("add_inplace_batched");
        auto&silu=getKernel("silu_mul_batched");
        auto&bagKernel=getKernel("qwen35_alpha_beta_gate_batched");
        auto&convKernel=getKernel("qwen35_conv_scan_silu");
        auto&splitSsmKernel=getKernel("qwen35_split_qkv_l2_batched");
        const bool valueMajorPrefill=
            (gpu->adapterName.find("NVIDIA")!=std::string::npos&&cfg.ssmTimeStepRank==32u)||
            (gpu->adapterName.find("AMD")!=std::string::npos&&
             (cfg.ssmTimeStepRank==16u||cfg.ssmTimeStepRank==32u));
        const auto&deltaKernel=valueMajorPrefill
            ? gpu->getOrCreatePipeline("delta_net_scan_x2_value_major",
                deltaNetValueMajorSource(WGSL_DELTA_NET_SCAN_X2),8)
            : getKernel("delta_net_scan_x2");
        auto&normGateKernel=getKernel("qwen35_norm_gated_batched");
        auto mm=[&](GPUBuffer x,GPUBuffer w,GPUBuffer s,GPUBuffer bias,GPUBuffer y,uint32_t K,uint32_t N,const std::string&n){auto p=mkp(n+"_p",{K,N,M});add(q8,{{0,x},{1,w},{2,s},{3,bias},{4,y},{5,p}},(M+3)/4,(N+31)/32,1,n);};
        auto kpl=[&](GGUFType t)->const CompiledPipeline&{return t==GGUF_TYPE_Q4_K?getKernel("q4k_matmul"):t==GGUF_TYPE_Q5_K?getKernel("q5k_matmul"):getKernel("q6k_matmul");};
        auto mmk=[&](GPUBuffer x,GPUBuffer w,GGUFType t,uint32_t nb,uint32_t rs,GPUBuffer bias,GPUBuffer y,uint32_t K,uint32_t N,const std::string&n){
            if(t==GGUF_TYPE_Q4_K&&M>=8&&gpu->adapterName.find("AMD")==std::string::npos){auto p=mkp(n+"_p",{K,N,M,nb,rs});auto&kp=getKernel("q4k_matmul_batched8");add(kp,{{0,x},{1,w},{2,bias},{3,y},{4,p}},(M+7)/8,(N+7)/8,1,n);}
            else if((t==GGUF_TYPE_Q4_K||t==GGUF_TYPE_Q5_K||t==GGUF_TYPE_Q6_K)&&M>=4&&gpu->adapterName.find("AMD")==std::string::npos){auto p=mkp(n+"_p",{K,N,M,nb,rs});const char*kn=t==GGUF_TYPE_Q4_K?"q4k_matmul_batched4":t==GGUF_TYPE_Q5_K?"q5k_matmul_batched4":"q6k_matmul_batched4";auto&kp=getKernel(kn);add(kp,{{0,x},{1,w},{2,bias},{3,y},{4,p}},(M+3)/4,(N+7)/8,1,n);}
            else{auto p=mkp(n+"_p",{K,N,nb,rs,0});auto&kp=kpl(t);add(kp,{{0,x},{1,w},{2,bias},{3,y},{4,p}},M,(N+7)/8,1,n);}};
        for(uint32_t li=0;li<cfg.nLayer;li++){
            if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u build begin\n",li);fflush(stderr);}
            auto&lw=layerWeights[li];auto&pl=cfg.perLayer[li];std::string L="qpf_L"+std::to_string(li)+"/";
            auto rp=mkp(L+"rms_p",{E,E,eb});add(rms,{{0,qwen35Pf.x},{1,qwen35Pf.norm},{2,lw.inputNorm},{3,qwen35Pf.rstd},{4,rp}},M,1,1,L+"rms");
            if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u rms built\n",li);fflush(stderr);}
            if(!cfg.isAttentionLayer(li)){
                uint32_t C=cfg.ssmInnerSize+2u*cfg.ssmGroupCount*cfg.ssmStateSize;
                uint32_t R=cfg.ssmTimeStepRank,DV=cfg.ssmInnerSize/R;
                mmk(qwen35Pf.norm,lw.qkvKQ,lw.qkvKQType,lw.qkvKQNBlocks,lw.qkvKQRowStride,zeroBiasQKV,qwen35Pf.qkv,E,C,L+"qkv");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u qkv built\n",li);fflush(stderr);}
                mmk(qwen35Pf.norm,lw.attnGateKQ,lw.attnGateKQType,lw.attnGateKQNBlocks,lw.attnGateKQRowStride,zeroBiasQKV,qwen35Pf.z,E,cfg.ssmInnerSize,L+"z");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u z built\n",li);fflush(stderr);}
                mm(qwen35Pf.norm,lw.ssmBetaAlphaW,lw.ssmBetaAlphaS,zeroBiasQKV,qwen35Pf.qj,E,2u*R,L+"ba");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u ba built\n",li);fflush(stderr);}
                auto bap=mkp(L+"bag_p",{M,R});
                add(bagKernel,{{0,qwen35Pf.qj},{1,lw.ssmDtBias},{2,lw.ssmA},{3,qwen35Pf.beta},{4,qwen35Pf.gate},{5,bap}},(M*R+63)/64,1,1,L+"bag");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u bag built\n",li);fflush(stderr);}
                auto cp=mkp(L+"conv_p",{C,cfg.ssmConvKernel,M});
                add(convKernel,{{0,ssmConvState[li]},{1,qwen35Pf.qkv},{2,lw.ssmConv1dW},{3,zeroBiasQKV},{4,qwen35Pf.conv},{5,cp}},(C+255)/256,1,1,L+"conv");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u conv built\n",li);fflush(stderr);}
                auto spm=mkp(L+"ssm_split_p",{cfg.ssmGroupCount,R,cfg.ssmStateSize,DV,eb,M});
                add(splitSsmKernel,{{0,qwen35Pf.conv},{1,qwen35Pf.sq},{2,qwen35Pf.sk},{3,qwen35Pf.sv},{4,spm}},3,std::max(cfg.ssmGroupCount,R),M,L+"ssm_split");
                auto dp=mkp(L+"delta_p",{R,cfg.ssmGroupCount,cfg.ssmStateSize,DV,M});
                add(deltaKernel,{{0,qwen35Pf.sq},{1,qwen35Pf.sk},{2,qwen35Pf.sv},{3,qwen35Pf.beta},{4,qwen35Pf.gate},{5,ssmHState[li]},{6,qwen35Pf.sy},{7,dp}},R,(DV+1)/2,1,L+"delta");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u delta built\n",li);fflush(stderr);}
                auto ngp=mkp(L+"ng_p",{R,DV,eb,M});
                add(normGateKernel,{{0,qwen35Pf.sy},{1,lw.ssmNorm},{2,qwen35Pf.z},{3,qwen35Pf.snorm},{4,ngp}},R,M,1,L+"normgate");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u normgate built\n",li);fflush(stderr);}
                mmk(qwen35Pf.snorm,lw.ssmOutKQ,lw.ssmOutKQType,lw.ssmOutKQNBlocks,lw.ssmOutKQRowStride,zeroBiasE,qwen35Pf.proj,cfg.ssmInnerSize,E,L+"out");
                if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u out built\n",li);fflush(stderr);}
            }else{
                uint32_t hd=pl.headDim,qd=pl.qDim,kd=pl.kvDim;
                mmk(qwen35Pf.norm,lw.qjKQ,lw.qjKQType,lw.qjKQNBlocks,lw.qjKQRowStride,zeroBiasV,qwen35Pf.qj,E,2u*qd,L+"q");
                mmk(qwen35Pf.norm,lw.kSepKQ,lw.kSepKQType,lw.kSepKQNBlocks,lw.kSepKQRowStride,zeroBiasV,qwen35Pf.ak,E,kd,L+"k");
                mmk(qwen35Pf.norm,lw.vSepKQ,lw.vSepKQType,lw.vSepKQNBlocks,lw.vSepKQRowStride,zeroBiasV,qwen35Pf.av,E,kd,L+"v");
                auto spp=mkp(L+"split_p",{M,cfg.nHead,hd});auto&sp=getKernel("qwen35_split_qg_batched");
                add(sp,{{0,qwen35Pf.qj},{1,qwen35Pf.aq},{2,qwen35Pf.ag},{3,spp}},(M*qd+255)/256,1,1,L+"split");
                auto&hn=getKernel("head_rmsnorm_batched");auto qnp=mkp(L+"qn_p",{M,cfg.nHead,hd,eb});
                add(hn,{{0,qwen35Pf.aq},{1,lw.qNorm},{2,qnp}},cfg.nHead,M,1,L+"qnorm");
                auto knp=mkp(L+"kn_p",{M,cfg.nKvHeads,hd,eb});add(hn,{{0,qwen35Pf.ak},{1,lw.kNorm},{2,knp}},cfg.nKvHeads,M,1,L+"knorm");
                auto ropep=mkp(L+"rope_p",{M,cfg.nHead,cfg.nKvHeads,hd,posOffset+done,cacheLen,(uint32_t)cfg.ropeSections[0],(uint32_t)cfg.ropeSections[1],(uint32_t)cfg.ropeSections[2],(uint32_t)cfg.ropeSections[3],rotaryDim/2});
                auto&rope=getKernel("qwen35_rope_kv_batched");add(rope,{{0,qwen35Pf.aq},{1,qwen35Pf.ak},{2,qwen35Pf.av},{3,qwen35Pf.qrot},{4,kvCache[li].K},{5,kvCache[li].V},{6,ropeCosBuf},{7,ropeSinBuf},{8,ropep}},std::max(cfg.nHead,cfg.nKvHeads),M,1,L+"rope");
                float sc=1.0f/sqrtf((float)hd),ni=-1e9f;uint32_t sb,nb;memcpy(&sb,&sc,4);memcpy(&nb,&ni,4);
                auto ap=mkp(L+"attn_p",{kd,cfg.nHead/cfg.nKvHeads,cacheLen+M,cacheLen,M,sb,nb,0},true);
                const bool mmaAttn=gpu->backendType!=WGPUBackendType_D3D12&&gpu->supportsSubgroupMatrix;
                auto&att=getKernelHD(mmaAttn?"flash_attn_vulkan":"causal_attn",hd);
                add(att,{{0,qwen35Pf.qrot},{1,kvCache[li].K},{2,kvCache[li].V},{3,qwen35Pf.attn},{4,ap}},cfg.nHead,(M+(mmaAttn?15u:3u))/(mmaAttn?16u:4u),1,L+"attn");
                auto gp=mkp(L+"gate_p",{M*qd});auto&go=getKernel("gated_output_batched");add(go,{{0,qwen35Pf.attn},{1,qwen35Pf.ag},{2,qwen35Pf.aout},{3,gp}},(M*qd+255)/256,1,1,L+"gate");
                mmk(qwen35Pf.aout,lw.oKQ,lw.oKQType,lw.oKQNBlocks,lw.oKQRowStride,zeroBiasE,qwen35Pf.proj,qd,E,L+"oproj");
            }
            auto anp=mkp(L+"addnorm_p",{E,E,eb});add(addnorm,{{0,qwen35Pf.x},{1,qwen35Pf.proj},{2,qwen35Pf.norm},{3,lw.postNorm},{4,qwen35Pf.rstd},{5,anp}},M,1,1,L+"postnorm");
            if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u postnorm built\n",li);fflush(stderr);}
            uint32_t im=pl.intermediateSize;mmk(qwen35Pf.norm,lw.guKQ,lw.guKQType,lw.guKQNBlocks,lw.guKQRowStride,zeroBiasGU,qwen35Pf.gateup,E,2u*im,L+"gateup");
            if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u gateup built\n",li);fflush(stderr);}
            auto sap=mkp(L+"silu_p",{M,im});add(silu,{{0,qwen35Pf.gateup},{1,qwen35Pf.act},{2,sap}},(M*im+255)/256,1,1,L+"silu");
            mmk(qwen35Pf.act,lw.dnKQ,lw.dnKQType,lw.dnKQNBlocks,lw.dnKQRowStride,zeroBiasE,qwen35Pf.proj,im,E,L+"down");
            if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u down built\n",li);fflush(stderr);}
            if(lw.postFfwNorm.handle){auto nap=mkp(L+"normadd_p",{E,E,eb});add(normadd,{{0,qwen35Pf.x},{1,qwen35Pf.proj},{2,lw.postFfwNorm},{3,qwen35Pf.rstd},{4,nap}},M,1,1,L+"ffn_add");}
            else{auto ap=mkp(L+"add_p",{M*E});add(addip,{{0,qwen35Pf.x},{1,qwen35Pf.proj},{2,ap}},(M*E+255)/256,1,1,L+"ffn_add");}
            if(traceQpf){fprintf(stderr,"[qwen-prefill] layer=%u build end dispatches=%zu\n",li,ds.size());fflush(stderr);}
            if(std::getenv("BP_PROFILE_QWEN_PREFILL")){
                gpu->writeBuffer(qwen35Pf.paramArena,paramHost.data(),paramCursor);
                for(const auto& d:ds){auto t0=std::chrono::steady_clock::now();
                    fprintf(stderr,"[qwen-prefill] layer=%u dispatch=%s begin\n",li,d.name.c_str());fflush(stderr);
                    std::vector<Dispatch> one{d};(void)gpu->submitAndReadback(one,qwen35Pf.x,4,passPerDispatch);
                    auto t1=std::chrono::steady_clock::now();fprintf(stderr,"[qwen-prefill] layer=%u dispatch=%s gpu=%.2fms\n",li,d.name.c_str(),std::chrono::duration<double,std::milli>(t1-t0).count());fflush(stderr);}
                ds.clear();
                for(auto old:bgs)if(old)wgpuBindGroupRelease(old);bgs.clear();
            }
        }
        bool last=done+M==T;
        if(last){
            auto fp=mkp("qpf_final_p",{E,E,eb});add(rms,{{0,qwen35Pf.x},{1,qwen35Pf.norm},{2,finalNormW},{3,qwen35Pf.rstd},{4,fp}},M,1,1,"qpf_final");
            GPUBuffer lastn=qwen35Pf.norm;lastn.offset=(uint64_t)(M-1)*E*4;lastn.size=E*4;
            auto lp=mkp("qpf_lm_p",{E,cfg.nVocab,kqLmNBlocks,kqLmRowStride,0});auto&lmp=getKernel("q6k_matmul_wide");add(lmp,{{0,lastn},{1,lmHeadKQ},{2,zeroBiasV},{3,logitsBuf},{4,lp}},1,(cfg.nVocab+15)/16,1,"qpf_lm");
            if(softcapPipeline&&softcapBG)ds.push_back({softcapPipeline,softcapBG,softcapDispatchX,1,1,"softcap"});
            ds.push_back(allDecodeDispatches[argmaxDispatchIndex]);ds.push_back(allDecodeDispatches[argmaxReduceDispatchIndex]);
            gpu->writeBuffer(qwen35Pf.paramArena,paramHost.data(),paramCursor);
            auto bytes=gpu->submitAndReadback(ds,argmaxResultBuf,4,true);memcpy(&result,bytes.data(),4);
        }else{
            gpu->writeBuffer(qwen35Pf.paramArena,paramHost.data(),paramCursor);
            // Scratch buffers, parameter-arena slots, and bind groups are
            // reused by the next chunk. Wait once per chunk (rather than once
            // per layer) so none of them are overwritten while still in use.
            (void)gpu->submitAndReadback(ds,qwen35Pf.x,4,true);
        }
        for(uint32_t li=0;li<cfg.nLayer;li++)kvCache[li].len+=M;for(auto bg:bgs)if(bg)wgpuBindGroupRelease(bg);done+=M;
    }return result;
}

int32_t ModelRunner::prefillGemmaBatched(
        const int32_t* tokenIds, uint32_t T, uint32_t posOffset) {
    if (!gemmaPf.ready || T == 0) return -1;
    using PrefillClock = std::chrono::steady_clock;
    const bool profileCpu = std::getenv("BP_PROFILE_CPU") != nullptr;
    double buildMs = 0.0, submitMs = 0.0, cleanupMs = 0.0;
    int32_t resultToken = -1;
    uint32_t done = 0;
    while (done < T) {
        const uint32_t M = std::min(gemmaPf.capacity, T - done);
        std::vector<int32_t> tokens(M);
        for (uint32_t r = 0; r < M; r++) {
            int32_t t = tokenIds[done + r];
            tokens[r] = (t >= 0 && (uint32_t)t < cfg.nVocab) ? t : 0;
        }
        gpu->writeBuffer(gemmaPf.tokens, tokens.data(), M * 4);
        auto buildStart = PrefillClock::now();

        std::vector<Dispatch> ds;
        std::vector<WGPUBindGroup> bgs;
        std::vector<GPUBuffer> params;
        auto mkP = [&](const std::string& name, std::initializer_list<uint32_t> v,
                       bool uniform = false) {
            uint32_t d[8] = {};
            size_t n = 0; for (uint32_t x : v) d[n++] = x;
            uint64_t bytes = n > 4 ? 32 : 16;
            auto b = gpu->createBuffer(name, bytes,
                uniform ? (BUF_UNIFORM | BUF_COPY_DST) : (BUF_STORAGE | BUF_COPY_DST));
            gpu->writeBuffer(b, d, bytes); params.push_back(b); return b;
        };
        auto add = [&](const CompiledPipeline& pl,
                       std::initializer_list<std::pair<uint32_t, GPUBuffer>> binds,
                       uint32_t gx, uint32_t gy, const std::string& name) {
            auto bg = makeBG(pl, std::vector<std::pair<uint32_t, GPUBuffer>>(binds));
            bgs.push_back(bg); ds.push_back({pl.pipeline, bg, gx, gy, 1, name});
        };
        uint32_t eb; memcpy(&eb, &cfg.rmsNormEps, 4);
        uint32_t scaleBits;

        auto& q4Gather = getKernel("q4_gather_batched");
        float embScale = cfg.embeddingScale > 0 ? cfg.embeddingScale : 1.0f;
        if (weightsAreNativeQ4) {
            memcpy(&scaleBits, &embScale, 4);
            auto embP = mkP("gpf_emb_p", {M, cfg.nEmbd, cfg.nVocab, scaleBits});
            add(q4Gather, {{0,lmHeadQ8W},{1,lmHeadQ8S},{2,gemmaPf.tokens},
                           {3,gemmaPf.x},{4,embP}},
                (M * cfg.nEmbd + 255) / 256, 1, "gpf_embed");
        } else {
            std::vector<float> embeddings((size_t)M * cfg.nEmbd);
            for (uint32_t r = 0; r < M; r++) {
                const float* src = embeddingCPU.data() + (size_t)tokens[r] * cfg.nEmbd;
                float* dst = embeddings.data() + (size_t)r * cfg.nEmbd;
                for (uint32_t j = 0; j < cfg.nEmbd; j++) dst[j] = src[j] * embScale;
            }
            gpu->writeBuffer(gemmaPf.x, embeddings.data(), embeddings.size() * sizeof(float));
        }

        auto& q4mm = getKernel("matmul_q4_batched");
        auto& q8mm = getKernel(useDP4A ? "q8_matmul_batched_dp4a" : "q8_matmul_d3d12");
        auto mm = [&](GPUBuffer x, GPUBuffer w, GPUBuffer s, GPUBuffer y,
                      uint32_t K, uint32_t N, const std::string& name) {
            auto p = mkP(name + "_p", {M,N,K});
            if (weightsAreNativeQ4) {
                add(q4mm, {{0,x},{1,w},{2,s},{3,y},{4,p}},
                    (N+31)/32,(M+3)/4,name);
            } else {
                auto q8p = mkP(name + "_q8p", {K,N,M});
                add(q8mm, {{0,x},{1,w},{2,s},{3,zeroBiasQKV},{4,y},{5,q8p}},
                    (M+(useDP4A?3u:7u))/(useDP4A?4u:8u),(N+31)/32,name);
            }
        };
        if (cfg.pleSize > 0 && pleGpuPreprocess) {
            uint32_t totalPle = cfg.pleSize * cfg.nLayer;
            float ps = sqrtf((float)cfg.pleSize); memcpy(&scaleBits, &ps, 4);
            auto pg = mkP("gpf_ple_gather_p", {M,totalPle,cfg.nVocab,scaleBits});
            add(q4Gather, {{0,pleTokenEmbW},{1,pleTokenEmbS},{2,gemmaPf.tokens},
                           {3,gemmaPf.pleSignal},{4,pg}},
                (M*totalPle+255)/256,1,"gpf_ple_gather");
            auto pm = mkP("gpf_ple_model_p", {M,totalPle,cfg.nEmbd});
            mm(gemmaPf.x,pleModelProjW,pleModelProjS,gemmaPf.pleRaw,
               cfg.nEmbd,totalPle,"gpf_ple_model");
            auto pc = mkP("gpf_ple_combine_p", {M,cfg.pleSize,cfg.nLayer,eb});
            auto& combine = getKernel("ple_combine_batched");
            add(combine, {{0,gemmaPf.pleRaw},{1,pleProjNormW},
                          {2,gemmaPf.pleSignal},{3,pc}},
                M,cfg.nLayer,"gpf_ple_combine");
        } else if (cfg.pleSize > 0 && !pleEmbCPU.empty()) {
            const uint32_t pleDim=cfg.pleSize,totalPle=pleDim*cfg.nLayer;
            const float pleScale=sqrtf((float)pleDim),invSqrt2=0.70710678f;
            std::vector<float> signal((size_t)M*totalPle),proj((size_t)M*totalPle);
            std::vector<uint32_t> jobs(M*cfg.nLayer);std::iota(jobs.begin(),jobs.end(),0u);
            std::for_each(std::execution::par,jobs.begin(),jobs.end(),[&](uint32_t job){
                uint32_t r=job/cfg.nLayer,li=job%cfg.nLayer;float* out=signal.data()+(size_t)r*totalPle+li*pleDim;
                const float* tok=pleEmbCPU.data()+(size_t)tokens[r]*totalPle+li*pleDim;
                const float* emb=embeddingCPU.data()+(size_t)tokens[r]*cfg.nEmbd;
                float* pr=proj.data()+(size_t)r*totalPle+li*pleDim;
                for(uint32_t d=0;d<pleDim;d++){const float* w=pleModelProjCPU.data()+(size_t)(li*pleDim+d)*cfg.nEmbd;float acc=0;for(uint32_t k=0;k<cfg.nEmbd;k++)acc+=w[k]*emb[k];pr[d]=acc;}
                float ss=0;for(uint32_t d=0;d<pleDim;d++)ss+=pr[d]*pr[d];float rms=1.0f/sqrtf(ss/(float)pleDim+cfg.rmsNormEps);
                for(uint32_t d=0;d<pleDim;d++)out[d]=(pr[d]*rms*(d<pleProjNormCPU.size()?pleProjNormCPU[d]:1.0f)+tok[d]*pleScale)*invSqrt2;
            });
            gpu->writeBuffer(gemmaPf.pleSignal,signal.data(),signal.size()*sizeof(float));
        }

        uint32_t cacheLen = kvCache[0].len;
        auto& rms = getKernel("rms_norm_batched");
        auto& sandwich = getKernel("gemma_sandwich_attn_batched");
        auto& normAdd = getKernel("gemma_norm_add_batched");
        auto& geluMul = getKernel("gelu_mul_batched");
        auto& pleMul = getKernel("ple_gelu_mul_batched");
        auto& scaleBuf = getKernel("scale_by_buffer");
        for (uint32_t li = 0; li < cfg.nLayer; li++) {
            auto& lw = layerWeights[li]; auto& pl = cfg.perLayer[li];
            const uint32_t hd=pl.headDim,qdim=pl.qDim,kvdim=pl.kvDim,im=pl.intermediateSize;
            auto rp = mkP("gpf_rms_"+std::to_string(li),{cfg.nEmbd,cfg.nEmbd,eb});
            add(rms,{{0,gemmaPf.x},{1,gemmaPf.norm},{2,lw.inputNorm},
                     {3,gemmaPf.rstd},{4,rp}},M,1,"gpf_rms");

            bool qonly = lw.qOnly;
            uint32_t qkvN = qonly ? qdim : qdim + 2u*kvdim;
            auto qp=mkP("gpf_qkv_"+std::to_string(li),{M,qkvN,cfg.nEmbd});
            mm(gemmaPf.norm,qonly?lw.qOnlyW:lw.qkvW,qonly?lw.qOnlyS:lw.qkvS,
               gemmaPf.qkv,cfg.nEmbd,qkvN,"gpf_qkv");

            bool swa = li<cfg.layerAttnTypes.size() &&
                       cfg.layerAttnTypes[li]==AttnLayerType::SlidingWindow;
            uint32_t cacheLayer = pl.kvSourceLayer>=0 ? (uint32_t)pl.kvSourceLayer : li;
            uint32_t flags=(qonly?1u:0u)|2u;
            auto ropeP=mkP("gpf_rope_"+std::to_string(li),
                {cfg.nHead,qdim,kvdim,posOffset+done,layerRotaryDim(li)/2,
                 cacheLen,cfg.nKvHeads,flags});
            auto& rope=getKernelHD("gemma_rope_batched",hd);
            auto& rc=(swa&&ropeCosBufSWA.handle)?ropeCosBufSWA:ropeCosBuf;
            auto& rs=(swa&&ropeSinBufSWA.handle)?ropeSinBufSWA:ropeSinBuf;
            add(rope,{{0,gemmaPf.qkv},{1,gemmaPf.qrot},{2,kvCache[cacheLayer].K},
                      {3,kvCache[cacheLayer].V},{4,rc},{5,rs},{6,lw.qNorm},
                      {7,lw.kNorm},{8,ropeP}},
                cfg.nHead+(qonly?0u:cfg.nKvHeads),M,"gpf_rope");

            uint32_t total=cacheLen+M;
            uint32_t kvStart=(swa&&cfg.slidingWindow>0&&total>cfg.slidingWindow)
                ? total-cfg.slidingWindow:0u;
            float ascale=1.0f,ni=-1e9f;uint32_t asb,nib;
            memcpy(&asb,&ascale,4);memcpy(&nib,&ni,4);
            auto ap=mkP("gpf_attn_"+std::to_string(li),
                {kvdim,cfg.nHead/cfg.nKvHeads,total,cacheLen,M,asb,nib,kvStart},true);
            const bool mmaAttn=gpu->backendType!=WGPUBackendType_D3D12&&gpu->supportsSubgroupMatrix;
            auto& attn=getKernelHD(mmaAttn?"flash_attn_vulkan":"causal_attn",hd);
            add(attn,{{0,gemmaPf.qrot},{1,kvCache[cacheLayer].K},
                      {2,kvCache[cacheLayer].V},{3,gemmaPf.attn},{4,ap}},
                cfg.nHead,(M+(mmaAttn?15u:3u))/(mmaAttn?16u:4u),"gpf_attn");

            auto op=mkP("gpf_o_"+std::to_string(li),{M,cfg.nEmbd,qdim});
            mm(gemmaPf.attn,lw.oW,lw.oS,gemmaPf.proj,qdim,cfg.nEmbd,"gpf_oproj");
            auto sp=mkP("gpf_sand_"+std::to_string(li),{cfg.nEmbd,cfg.nEmbd,eb});
            add(sandwich,{{0,gemmaPf.x},{1,gemmaPf.proj},{2,lw.postNorm},
                          {3,lw.ffnNorm},{4,gemmaPf.norm},{5,gemmaPf.rstd},{6,sp}},
                M,1,"gpf_sandwich");

            auto gp=mkP("gpf_gu_"+std::to_string(li),{M,2u*im,cfg.nEmbd});
            mm(gemmaPf.norm,lw.guW,lw.guS,gemmaPf.gateup,cfg.nEmbd,2u*im,"gpf_gateup");
            auto gap=mkP("gpf_gelu_"+std::to_string(li),{M,im});
            add(geluMul,{{0,gemmaPf.gateup},{1,gemmaPf.act},{2,gap}},
                (M*im+255)/256,1,"gpf_gelu");
            auto dp=mkP("gpf_down_"+std::to_string(li),{M,cfg.nEmbd,im});
            mm(gemmaPf.act,lw.dnW,lw.dnS,gemmaPf.proj,im,cfg.nEmbd,"gpf_down");
            auto np=mkP("gpf_ffn_add_"+std::to_string(li),{cfg.nEmbd,cfg.nEmbd,eb});
            add(normAdd,{{0,gemmaPf.x},{1,gemmaPf.proj},{2,lw.postFfwNorm},
                         {3,gemmaPf.rstd},{4,np}},M,1,"gpf_ffn_add");

            if(cfg.pleSize>0&&(pleGpuPreprocess||!pleEmbCPU.empty())&&lw.pleInpGateW.handle){
                auto p1=mkP("gpf_pg_"+std::to_string(li),{M,cfg.pleSize,cfg.nEmbd});
                mm(gemmaPf.x,lw.pleInpGateW,lw.pleInpGateS,gemmaPf.pleGate,
                   cfg.nEmbd,cfg.pleSize,"gpf_ple_gate");
                auto p2=mkP("gpf_pm_"+std::to_string(li),{M,cfg.pleSize,li,cfg.nLayer});
                add(pleMul,{{0,gemmaPf.pleGate},{1,gemmaPf.pleSignal},{2,p2}},
                    (M*cfg.pleSize+255)/256,1,"gpf_ple_mul");
                auto p3=mkP("gpf_pp_"+std::to_string(li),{M,cfg.nEmbd,cfg.pleSize});
                mm(gemmaPf.pleGate,lw.pleProjW,lw.pleProjS,gemmaPf.pleOut,
                   cfg.pleSize,cfg.nEmbd,"gpf_ple_proj");
                auto p4=mkP("gpf_pa_"+std::to_string(li),{cfg.nEmbd,cfg.nEmbd,eb});
                add(normAdd,{{0,gemmaPf.x},{1,gemmaPf.pleOut},{2,lw.plePostNorm},
                             {3,gemmaPf.rstd},{4,p4}},M,1,"gpf_ple_add");
            }
            if(lw.outScale.handle){
                auto scp=mkP("gpf_scale_"+std::to_string(li),{M*cfg.nEmbd});
                add(scaleBuf,{{0,gemmaPf.x},{1,lw.outScale},{2,scp}},
                    (M*cfg.nEmbd+255)/256,1,"gpf_scale");
            }
        }

        bool lastChunk = done + M == T;
        if (lastChunk) {
            auto fp=mkP("gpf_final",{cfg.nEmbd,cfg.nEmbd,eb});
            add(rms,{{0,gemmaPf.x},{1,gemmaPf.norm},{2,finalNormW},
                     {3,gemmaPf.rstd},{4,fp}},M,1,"gpf_final_rms");
            GPUBuffer lastNorm=gemmaPf.norm;
            lastNorm.offset=(uint64_t)(M-1)*cfg.nEmbd*4; lastNorm.size=cfg.nEmbd*4;
            if (weightsAreNativeQ4) {
                auto lp=mkP("gpf_lm",{0,cfg.nVocab,cfg.nEmbd,0});
                auto& lm=getKernel("matmul_q4_decode");
                add(lm,{{0,lastNorm},{1,lmHeadQ8W},{2,lmHeadQ8S},
                        {3,logitsBuf},{4,lp}},
                    (cfg.nVocab+31)/32,1,"gpf_lm");
            } else {
                auto lp=mkP("gpf_lm_q8",{cfg.nEmbd,cfg.nVocab,1});
                add(q8mm,{{0,lastNorm},{1,lmHeadQ8W},{2,lmHeadQ8S},
                          {3,zeroBiasV},{4,logitsBuf},{5,lp}},
                    1,(cfg.nVocab+31)/32,"gpf_lm");
            }
            if(softcapPipeline&&softcapBG){
                ds.push_back({softcapPipeline,softcapBG,softcapDispatchX,1,1,"logit_softcap"});
            }
            ds.push_back(allDecodeDispatches[argmaxDispatchIndex]);
            ds.push_back(allDecodeDispatches[argmaxReduceDispatchIndex]);
            auto buildEnd = PrefillClock::now();
            auto bytes=gpu->submitAndReadback(ds,argmaxResultBuf,4,passPerDispatch);
            auto submitEnd = PrefillClock::now();
            buildMs += std::chrono::duration<double, std::milli>(buildEnd-buildStart).count();
            submitMs += std::chrono::duration<double, std::milli>(submitEnd-buildEnd).count();
            memcpy(&resultToken,bytes.data(),4);
        } else {
            auto buildEnd = PrefillClock::now();
            gpu->submitOnly(ds,!passPerDispatch);
            auto submitEnd = PrefillClock::now();
            buildMs += std::chrono::duration<double, std::milli>(buildEnd-buildStart).count();
            submitMs += std::chrono::duration<double, std::milli>(submitEnd-buildEnd).count();
        }
        for(uint32_t li=0;li<cfg.nLayer;li++)kvCache[li].len+=M;
        auto cleanupStart = PrefillClock::now();
        for(auto bg:bgs)if(bg)wgpuBindGroupRelease(bg);
        for(auto p:params)if(p.handle)wgpuBufferRelease(p.handle);
        cleanupMs += std::chrono::duration<double, std::milli>(PrefillClock::now()-cleanupStart).count();
        done+=M;
    }
    if (profileCpu) {
        fprintf(stderr, "[gemma-prefill-cpu] T=%u build=%.2fms submit/wait=%.2fms cleanup=%.2fms\n",
                T, buildMs, submitMs, cleanupMs);
    }
    return resultToken;
}

int32_t ModelRunner::prefillBatched(
        const int32_t* tokenIds, uint32_t T, uint32_t posOffset) {
    if (qwen35Pf.ready && !std::getenv("BP_QWEN35_SERIAL_PREFILL"))
        return prefillQwen35Batched(tokenIds,T,posOffset);
    if (gemmaPf.ready && !pleTokenEmbAsymmetric &&
        !std::getenv("BP_GEMMA_SERIAL_PREFILL"))
        return prefillGemmaBatched(tokenIds, T, posOffset);
    if ((pleGpuPreprocess || cfg.arch == "qwen35") &&
        !std::getenv("BP_SYNC_PREFILL"))
        return prefillPooledKnown(tokenIds, T, posOffset);
    // For small T or when prefill resources are not available (K-quant), use serial path
    if (T <= 16 || !pfCache.pX.handle) {
        fprintf(stderr, "  [prefillBatched] serial path T=%u\n", T); fflush(stderr);
        if (T == 1) {
            auto logits = decode(tokenIds[0], posOffset);
            return argmax(logits);
        }
        // Use synchronous decode for each token (safer than fire-and-forget prefillStep)
        std::vector<float> logits;
        for (uint32_t t = 0; t < T; t++)
            logits = decode(tokenIds[t], posOffset + t);
        return argmax(logits);
    }

    // ── True batched prefill using pre-allocated resources ───────────
    using hrc = std::chrono::high_resolution_clock;
    auto t_start = hrc::now();

    uint32_t qDimL  = cfg.nHead * cfg.headDim;
    uint32_t kvDimL = cfg.nKvHeads * cfg.headDim;
    uint32_t qkvOutL = qDimL + 2 * kvDimL;
    uint32_t Q8_TILE = 8, TILE_M = 8;
    bool usePrequant = !useMMA && useDP4A;

    // Upload embeddings into pre-allocated pX
    std::vector<float> embData(T * cfg.nEmbd);
    for (uint32_t t = 0; t < T; t++) {
        int32_t tok = (tokenIds[t] >= 0 && (uint32_t)tokenIds[t] < cfg.nVocab)
                      ? tokenIds[t] : 0;
        memcpy(embData.data() + t * cfg.nEmbd,
               embeddingCPU.data() + tok * cfg.nEmbd, cfg.nEmbd * 4);
    }
    if (cfg.embeddingScale > 0.0f) {
        for (size_t i = 0; i < embData.size(); i++)
            embData[i] *= cfg.embeddingScale;
    }
    gpu->writeBuffer(pfCache.pX, embData.data(), T * cfg.nEmbd * 4);

    // Write T-dependent params into pre-allocated param buffers
    {
        uint32_t v[4];
        v[0] = T; v[1] = qkvOutL; v[2] = cfg.nEmbd; v[3] = 0;
        gpu->writeBuffer(pfCache.pQkvP, v, 16);
        v[0] = T; v[1] = cfg.nEmbd; v[2] = qDimL;
        gpu->writeBuffer(pfCache.pOpP, v, 16);
        v[0] = T; v[1] = 2 * cfg.intermediateSize; v[2] = cfg.nEmbd;
        gpu->writeBuffer(pfCache.pGuP, v, 16);
        v[0] = cfg.intermediateSize; v[1] = cfg.nEmbd; v[2] = cfg.intermediateSize; v[3] = T;
        gpu->writeBuffer(pfCache.pDnP, v, 16);

        if (usePrequant) {
            v[0] = T; v[1] = cfg.nEmbd; v[2] = 0; v[3] = 0;
            gpu->writeBuffer(pfCache.pNormQP, v, 16);
            v[0] = T; v[1] = qDimL; v[2] = 0; v[3] = 0;
            gpu->writeBuffer(pfCache.pAttnQP, v, 16);
            v[0] = T; v[1] = cfg.intermediateSize; v[2] = 0; v[3] = 0;
            gpu->writeBuffer(pfCache.pGUQP, v, 16);
        }
    }

    uint32_t cacheLen = kvCache[0].len;

    // Write per-layer RoPE + attention params
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        uint32_t ropeP[8] = {};
        ropeP[0] = cfg.nHead;  ropeP[1] = qDimL;  ropeP[2] = kvDimL;
        ropeP[3] = posOffset;  ropeP[4] = (rotaryDim > 0) ? rotaryDim / 2 : cfg.headDim / 2;
        ropeP[5] = cacheLen;
        float eps_f = cfg.rmsNormEps; memcpy(&ropeP[6], &eps_f, 4);
        ropeP[7] = cfg.nKvHeads;
        gpu->writeBuffer(pfCache.ropeParams[li], ropeP, 32);

        uint32_t T_total = cacheLen + T;
        uint32_t ap[8] = {};
        ap[0] = cfg.nKvHeads * cfg.headDim;
        ap[1] = cfg.nHead / cfg.nKvHeads;
        ap[2] = T_total;  ap[3] = cacheLen;
        ap[4] = T;  // T_prefill
        float sc = 1.0f / sqrtf((float)cfg.headDim);
        float ni = -1e9f;
        memcpy(&ap[5], &sc, 4);  memcpy(&ap[6], &ni, 4);
        gpu->writeBuffer(pfCache.attnParams[li], ap, 32);
    }

    auto t_params = hrc::now();

    // Write grid sizes into pre-allocated indirect dispatch buffer
    const uint32_t MAT_M = tuning.prefillMatM;
    const uint32_t MAT_N = tuning.prefillMatN;
    const uint32_t MAT_WIDE_M = tuning.prefillWideMatM;
    const uint32_t MAT_WIDE_N = tuning.prefillWideMatN;
    const uint32_t DS_M  = tuning.prefillDnM;
    const uint32_t DS_N  = tuning.prefillDnN;
    const uint32_t ATTN_BQ = tuning.prefillAttnBlockQ;
    const uint32_t QN_NORM = (cfg.nEmbd + 255u) / 256u;
    const uint32_t QN_ATTN = (qDimL + 255u) / 256u;
    const uint32_t QN_GU   = (cfg.intermediateSize + 255u) / 256u;

    uint32_t perLayerDispatches = usePrequant ? 12u : 8u;
    uint32_t nDispatches = cfg.nLayer * perLayerDispatches + 1;
    std::vector<uint32_t> grids(nDispatches * 3);
    for (uint32_t li = 0; li < cfg.nLayer; li++) {
        uint32_t base = li * perLayerDispatches * 3;
        uint32_t idx = base;
        auto emit = [&](uint32_t gx, uint32_t gy, uint32_t gz) {
            grids[idx + 0] = gx;
            grids[idx + 1] = gy;
            grids[idx + 2] = gz;
            idx += 3;
        };

        emit(T, 1, 1);
        if (usePrequant) emit(QN_NORM, T, 1);
           emit((qkvOutL + MAT_WIDE_N - 1) / MAT_WIDE_N,
               (T + MAT_WIDE_M - 1) / MAT_WIDE_M, 1);
        emit(cfg.nHead + cfg.nKvHeads, T, 1);
        emit(cfg.nHead, (T + ATTN_BQ - 1) / ATTN_BQ, 1);
        if (usePrequant) emit(QN_ATTN, T, 1);
        emit((cfg.nEmbd + MAT_N - 1) / MAT_N, (T + MAT_M - 1) / MAT_M, 1);
        emit(T, 1, 1);
           if (usePrequant) emit(QN_NORM, T, 1);
           emit((2 * cfg.intermediateSize + MAT_WIDE_N - 1) / MAT_WIDE_N,
               (T + MAT_WIDE_M - 1) / MAT_WIDE_M, 1);
        if (usePrequant) emit(QN_GU, T, 1);
        emit((cfg.nEmbd + DS_N - 1) / DS_N, (T + DS_M - 1) / DS_M, 1);
    }
    uint32_t fb = cfg.nLayer * perLayerDispatches * 3;
    grids[fb + 0] = T; grids[fb + 1] = 1; grids[fb + 2] = 1;

    gpu->writeBuffer(pfCache.indirectBuf, grids.data(), nDispatches * 12);

    // Submit all prefill dispatches in one shot
    auto t_dispatches = hrc::now();

    // Write LM head params before submission (needed in the same command buffer)
    {
        uint32_t v[4] = {cfg.nEmbd, cfg.nVocab, 1, 0};
        gpu->writeBuffer(pfCache.pLmP, v, 16);
    }

    std::vector<uint8_t> result;

    if (profiler && profiler->enabled()) {
        // Profiler path: use direct dispatch (needs per-dispatch compute passes)
        std::vector<Dispatch> allPrefill;
        allPrefill.reserve(nDispatches);
        for (uint32_t i = 0; i < nDispatches; i++) {
            auto& e = pfCache.indirectTable[i];
            allPrefill.push_back({e.pipeline, e.bindGroup,
                grids[i*3], grids[i*3+1], grids[i*3+2], e.name});
        }
        gpu->submitOnlyProfiled(allPrefill, *profiler);

        // Copy last token's norm output → normOutBuf for LM head
        {
            WGPUCommandEncoderDescriptor enD{};
            auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                pfCache.pNorm.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
                normOutBuf.handle, 0, cfg.nEmbd * 4);
            WGPUCommandBufferDescriptor cbD{};
            auto cb = wgpuCommandEncoderFinish(enc, &cbD);
            wgpuQueueSubmit(gpu->queue, 1, &cb);
            wgpuCommandEncoderRelease(enc);
            wgpuCommandBufferRelease(cb);
        }

        // LM head + softcap + argmax
        std::vector<Dispatch> lmArgmax;
        if (lmHeadIsQ8) {
            lmArgmax.push_back({getKernel("q8_matmul").pipeline, pfCache.lmBG,
                1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1, "pf_lm"});
        }
        if (softcapPipeline) {
            lmArgmax.push_back({softcapPipeline, softcapBG,
                softcapDispatchX, 1, 1, "pf_softcap"});
        }
        lmArgmax.push_back({getKernel("argmax").pipeline, pfCache.argmaxBG,
            1, 1, 1, "pf_argmax"});
        result = gpu->submitAndReadbackProfiled(lmArgmax, argmaxResultBuf, 4,
                                                 *profiler);
    } else {
        // Fast path: indirect dispatch — pre-recorded pipeline/BG, only grid size buffer changes
        auto rb = gpu->getOrCreateReadbackBuf(4);

        auto t_enc0 = hrc::now();

        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);

        // 1. All prefill dispatches in a single compute pass using indirect dispatch
        {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            for (auto& e : pfCache.indirectTable) {
                wgpuComputePassEncoderSetPipeline(pass, e.pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, e.bindGroup, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroupsIndirect(
                    pass, pfCache.indirectBuf.handle, e.indirectOffset);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 2. Copy last token's norm output → normOutBuf
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pfCache.pNorm.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
            normOutBuf.handle, 0, cfg.nEmbd * 4);

        // 3. LM head + argmax in another compute pass
        {
            WGPUComputePassDescriptor cpD{};
            auto pass = wgpuCommandEncoderBeginComputePass(enc, &cpD);
            if (lmHeadIsQ8) {
                wgpuComputePassEncoderSetPipeline(pass, getKernel("q8_matmul").pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, pfCache.lmBG, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, 1, (cfg.nVocab + Q8_TILE - 1) / Q8_TILE, 1);
            }
            if (softcapPipeline) {
                wgpuComputePassEncoderSetPipeline(pass, softcapPipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, softcapBG, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, softcapDispatchX, 1, 1);
            }
            wgpuComputePassEncoderSetPipeline(pass, getKernel("argmax").pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, pfCache.argmaxBG, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 4. Copy argmax result for readback
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            argmaxResultBuf.handle, 0, rb, 0, 4);

        // 5. Copy last hidden state → xBuf for decode
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pfCache.pX.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
            xBuf.handle, 0, cfg.nEmbd * 4);

        auto t_enc1 = hrc::now();  // encoding done

        // Single submit
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);

        auto t_finish = hrc::now();  // finish (Dawn barrier computation) done

        wgpuQueueSubmit(gpu->queue, 1, &cb);

        auto t_submitted = hrc::now();  // submit done

        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);

        // Synchronous map of 4 bytes
        struct { bool done; uint32_t status; } ms{false, 0};
        WGPUBufferMapCallbackInfo mcb{};
        mcb.mode = WGPUCallbackMode_WaitAnyOnly;
        mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
            auto* p = static_cast<decltype(&ms)>(u);
            p->done = true; p->status = s;
        };
        mcb.userdata1 = &ms;
        auto mf = wgpuBufferMapAsync(rb, 1 /*READ*/, 0, 4, mcb);
        WGPUFutureWaitInfo mw{mf, 0};
        wgpuInstanceWaitAny(gpu->instance, 1, &mw, UINT64_MAX);

        auto t_gpudone = hrc::now();  // GPU finished + readback mapped

        result.resize(4);
        if (ms.status == 1) {
            auto ptr = wgpuBufferGetConstMappedRange(rb, 0, 4);
            memcpy(result.data(), ptr, 4);
            wgpuBufferUnmap(rb);
        }

        // Detailed CPU timing breakdown
        auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        fprintf(stderr, "    cpu: encode=%.0fus finish=%.0fus submit=%.0fus gpu_wait=%.0fus\n",
               us(t_enc0, t_enc1), us(t_enc1, t_finish),
               us(t_finish, t_submitted), us(t_submitted, t_gpudone));
    }
    auto t_submit = hrc::now();

    // Print prefill timing breakdown
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "  [prefill T=%u] params=%.1fms build=%.1fms gpu+readback=%.1fms total=%.1fms (%zu dispatches)\n",
           T, ms(t_start, t_params), ms(t_params, t_dispatches),
           ms(t_dispatches, t_submit),
           ms(t_start, t_submit), pfCache.indirectTable.size());

    // Update KV cache lengths
    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len += T;

    // Copy last hidden state to xBuf for subsequent decode (profiler path only;
    // non-profiled path already includes this in the consolidated encoder)
    if (profiler && profiler->enabled()) {
        WGPUCommandEncoderDescriptor enD{};
        auto enc = wgpuDeviceCreateCommandEncoder(gpu->device, &enD);
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            pfCache.pX.handle, (uint64_t)(T - 1) * cfg.nEmbd * 4,
            xBuf.handle, 0, cfg.nEmbd * 4);
        WGPUCommandBufferDescriptor cbD{};
        auto cb = wgpuCommandEncoderFinish(enc, &cbD);
        wgpuQueueSubmit(gpu->queue, 1, &cb);
        wgpuCommandEncoderRelease(enc);
        wgpuCommandBufferRelease(cb);
    }

    int32_t tokId;
    memcpy(&tokId, result.data(), 4);
    return tokId;
}
int32_t ModelRunner::argmax(const std::vector<float>& logits) {
    return (int32_t)std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end()));
}

void ModelRunner::resetKVCache() {
    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len = 0;

    // Recurrent models carry state outside the attention KV cache. Leaving
    // these buffers intact makes warmup/autotuning and previous prompts alter
    // the next prompt's very first token.
    uint64_t maxBytes = 0;
    for (const auto& b : ssmConvState) if (b.handle) maxBytes = std::max(maxBytes, b.size);
    for (const auto& b : ssmHState)    if (b.handle) maxBytes = std::max(maxBytes, b.size);
    if (maxBytes > 0) {
        std::vector<uint8_t> zeros((size_t)maxBytes, 0);
        for (const auto& b : ssmConvState)
            if (b.handle) gpu->writeBuffer(b, zeros.data(), b.size);
        for (const auto& b : ssmHState)
            if (b.handle) gpu->writeBuffer(b, zeros.data(), b.size);
    }
}

void ModelRunner::enableProfiling() {
    profiler = new GPUProfiler();
    if (!profiler->init(gpu->device, gpu->instance, gpu->queue)) {
        fprintf(stderr, "WARNING: GPU timestamp queries not available\n");
        delete profiler;
        profiler = nullptr;
        return;
    }

    // Acquire CPU<->GPU clock calibration
    auto cal = acquireClockCalibration(gpu->device, gpu->backendType);
    if (cal.valid) {
        calibration = new ClockCalibration(cal);
        fprintf(stderr, "  Clock calibration: GPU=%llu ns, CPU=%llu ns, deviation=%llu ns\n",
               (unsigned long long)cal.gpuTimestampNs,
               (unsigned long long)cal.cpuTimestampNs,
               (unsigned long long)cal.maxDeviationNs);
    } else {
        fprintf(stderr, "  Clock calibration: not available (GPU-only timing)\n");
    }
}


void ModelRunner::printProfileReport(int nDecodeTokens, int nPrefillTokens,
                                     double prefillMs, double decodeMs,
                                     const std::string& profileOutputPath) {
    if (!profiler || !profiler->enabled() || profiler->nextIndex == 0)
        return;

    // Map the profiler's readback buffer to get timestamp values
    uint32_t count = profiler->nextIndex;
    uint64_t readSize = count * 8;

    struct { bool done; uint32_t status; } ms{false, 0};
    WGPUBufferMapCallbackInfo mcb{};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
        auto* p = static_cast<decltype(&ms)>(u);
        p->done = true; p->status = s;
    };
    mcb.userdata1 = &ms;
    auto mf = wgpuBufferMapAsync(profiler->readbackBuf, 1, 0, readSize, mcb);
    WGPUFutureWaitInfo mw{mf, 0};
    wgpuInstanceWaitAny(gpu->instance, 1, &mw, UINT64_MAX);

    if (ms.status != 1) {
        fprintf(stderr, "Failed to map profiler readback buffer\n");
        return;
    }

    auto ptr = (const uint64_t*)wgpuBufferGetConstMappedRange(
        profiler->readbackBuf, 0, readSize);

    // Compute GPU frequency (nanoseconds per tick)
    // Vulkan timestamps are in nanoseconds on most drivers
    // but on some may use the device's timestamp period.
    // Dawn normalizes to nanoseconds for us.

    // Aggregate by kernel name (strip layer prefix)
    struct AggEntry {
        double totalUs = 0;
        uint32_t count = 0;
    };
    std::unordered_map<std::string, AggEntry> agg;
    double totalGpuUs = 0;

    for (auto& e : profiler->entries) {
        uint64_t begin = ptr[e.beginIdx];
        uint64_t end   = ptr[e.endIdx];
        // Skip invalid timestamps (uninitialized or wrapped)
        if (end <= begin || begin == 0) continue;
        double durNs = (double)(end - begin);
        double durUs = durNs / 1000.0;

        // Strip layer prefix: "L5/q8_qkv" -> "q8_qkv"
        std::string kernel = e.name;
        auto slash = kernel.find('/');
        if (slash != std::string::npos)
            kernel = kernel.substr(slash + 1);

        agg[kernel].totalUs += durUs;
        agg[kernel].count++;
        totalGpuUs += durUs;
    }

    // Sort by total time descending
    std::vector<std::pair<std::string, AggEntry>> sorted_agg(agg.begin(), agg.end());
    std::sort(sorted_agg.begin(), sorted_agg.end(),
              [](auto& a, auto& b) { return a.second.totalUs > b.second.totalUs; });

    // Print report
    fprintf(stderr, "\n--- GPU Profile (hardware timestamps) ---\n");
    fprintf(stderr, "%-20s %10s %6s %10s %6s\n",
           "Kernel", "Total(ms)", "Count", "Avg(us)", "%%");
    fprintf(stderr, "%-20s %10s %6s %10s %6s\n",
           "--------------------", "----------", "------", "----------", "------");
    for (auto& [name, e] : sorted_agg) {
        double totalMs = e.totalUs / 1000.0;
        double avgUs = e.totalUs / e.count;
        double pct = totalGpuUs > 0 ? e.totalUs / totalGpuUs * 100.0 : 0;
        fprintf(stderr, "%-20s %10.2f %6u %10.1f %5.1f%%\n",
               name.c_str(), totalMs, e.count, avgUs, pct);
    }
    fprintf(stderr, "%-20s %10.2f\n", "TOTAL", totalGpuUs / 1000.0);

    // Generate HTML timeline report
    std::string htmlPath = profileOutputPath;
    if (htmlPath.empty()) {
        // Default: place profile.html next to the GGUF file
        auto dir = std::filesystem::path(ggufPath).parent_path();
        htmlPath = (dir / "profile.html").string();
    }
    generateProfileHTML(*gpu, *profiler, calibration, ptr,
                        nDecodeTokens, nPrefillTokens,
                        prefillMs, decodeMs, htmlPath);

    wgpuBufferUnmap(profiler->readbackBuf);

    profiler->destroy();
    delete profiler;
    profiler = nullptr;
    if (calibration) { delete calibration; calibration = nullptr; }
}

// ─── MTP (Multi-Token Prediction) ─────────────────────────────────────────

int32_t ModelRunner::mtpDraft(int32_t lastToken, uint32_t pos,
                               std::vector<int32_t>& draftTokens) {
    if (mtpCfg.type == MTPType::None) return 0;
    if (!mtpWeights.preProjW.handle) return 0;

    draftTokens.clear();
    uint32_t numDraft = mtpCfg.numDraftTokens;

    // Buffers for MTP hidden state (reuse existing intermediate buffers where possible)
    // MTP hidden dim may differ from backbone hidden dim
    uint32_t mtpE = mtpCfg.hiddenSize > 0 ? mtpCfg.hiddenSize : cfg.nEmbd;

    for (uint32_t d = 0; d < numDraft; d++) {
        int32_t inputToken = (d == 0) ? lastToken : draftTokens.back();

        // 1. Embed the input token (reuse backbone embedding)
        if (inputToken < 0 || (uint32_t)inputToken >= cfg.nVocab) break;
        const float* emb = embeddingCPU.data() + inputToken * cfg.nEmbd;

        // 2. For Gemma 4: concat(embed * scale, backbone_hidden) → pre-project
        //    For Qwen 3.6: enorm(embed), hnorm(hidden) → concat → project
        //
        // The backbone's last hidden state is in xBuf after the most recent decode.
        // Read it back from GPU for the concat.
        //
        // NOTE: For full GPU implementation, we'd keep everything on GPU.
        // This CPU-side implementation is a correctness reference; GPU kernels
        // for the MTP head would be needed for production perf.

        // 3. Run MTP decoder layers (Q-only attention for Gemma 4, full for Qwen)
        // 4. Post-project back to backbone dim
        // 5. RMSNorm + LM head → argmax

        // For now: MTP weights must be loaded and pipeline built.
        // The full GPU pipeline for MTP requires:
        //   - Pre-projection matmul (2*E → mtpE)
        //   - Per-layer: RMSNorm, Q-proj, attention (shared KV), FFN
        //   - Post-projection matmul (mtpE → E)
        //   - Final norm + LM head matmul + argmax
        //
        // Each of these can reuse existing Q8 matmul / norm / attention kernels.
        // The Q-only attention (Gemma 4) reads K/V from the backbone's kvCache
        // but only computes Q from the MTP hidden state.

        // Placeholder: without MTP weights loaded, we can't draft
        break;
    }

    return (int32_t)draftTokens.size();
}

int32_t ModelRunner::mtpVerifyAndAccept(const std::vector<int32_t>& draftTokens,
                                         uint32_t pos, uint32_t& acceptedCount) {
    if (draftTokens.empty()) { acceptedCount = 0; return -1; }

    // Verification: run the backbone on all draft tokens in parallel
    // using the batched prefill path. This gives us logits for each position.
    //
    // Build input: [draft_0, draft_1, ..., draft_N-1]
    // Run prefillBatched to get the backbone's prediction at each position.
    // Compare each backbone prediction with the corresponding draft token.

    uint32_t T = (uint32_t)draftTokens.size();
    acceptedCount = 0;

    // Use prefillBatched if available, otherwise serial
    if (T == 1) {
        auto logits = decode(draftTokens[0], pos);
        int32_t backbone_pred = argmax(logits);
        if (backbone_pred == draftTokens[0]) {
            acceptedCount = 1;
            // Need one more decode to get the next token
            auto next_logits = decode(draftTokens[0], pos + 1);
            return argmax(next_logits);
        }
        return backbone_pred;
    }

    // For T > 1: ideally use batched prefill for parallel verification.
    // The backbone processes [draft_0, ..., draft_{T-1}] and returns
    // logits at each position. We accept greedily until mismatch.
    //
    // Since prefillBatched returns only the last token's argmax,
    // we need the serial path for proper verification:
    for (uint32_t i = 0; i < T; i++) {
        auto logits = decode(draftTokens[i], pos + i);
        int32_t backbone_pred = argmax(logits);

        if (i + 1 < T && backbone_pred != draftTokens[i + 1]) {
            // Mismatch at position i+1: accept tokens 0..i, return backbone's prediction
            acceptedCount = i + 1;
            return backbone_pred;
        }

        if (i + 1 == T) {
            // All drafts verified, return the bonus token
            acceptedCount = T;
            return backbone_pred;
        }
    }

    acceptedCount = 0;
    return -1;
}
