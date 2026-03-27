#pragma once
/**
 * wgsl_template.h — Dtype-polymorphic WGSL shader generation.
 *
 * Generates f32 or fp16 kernel variants from template WGSL source.
 * fp16 data is stored as array<u32> and accessed via unpack2x16float /
 * pack2x16float to work around the D3D12 typed buffer limitation (§1.1).
 *
 * All shader generation happens ONCE per variant (on first pipeline
 * cache miss).  Use GetPipelineT() which only calls the generator
 * lambda on cache miss:
 *
 *   auto& pl = ex.GetPipelineT("binary_f16", 4, []() {
 *       return instantiateTemplate(WGSL_BINARY_ELEMENTWISE_T, TensorDtype::Float16);
 *   });
 *
 * For kernels where different dtypes need different ALGORITHMS (not just
 * storage types), use a custom generator lambda:
 *
 *   auto& pl = ex.GetPipelineT("gqa_decode_f16", 5, []() {
 *       return generateGqaDecodeF16();  // entirely different shader
 *   });
 *
 * Markers in templates:
 *   ${T}        — storage type: "f32" or "u32"
 *   ${T_READ}   — helper function for reading one element
 *   ${T_WRITE}  — helper function for writing one element (with RMW for fp16)
 *   ${T_WRITE2} — helper function for writing a pair of elements (race-free)
 *   ${T_DTYPE}  — "f32" or "f16" (for metadata/comments)
 *   ${T_BYTES}  — "4" or "2" (bytes per element)
 */

#include <string>
#include <cstring>

enum class TensorDtype;  // forward from graph_executor.h

// Dtype suffix for pipeline cache keys
inline const char* dtypeSuffix(TensorDtype dtype) {
    // TensorDtype::Float16 = 1 in our enum
    switch (dtype) {
        case TensorDtype(1): return "_f16";  // Float16
        default: return "_f32";
    }
}

// Replace all occurrences of `from` with `to` in `str`
inline void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
}

/// Instantiate a WGSL template for the given dtype.
/// For f32: direct array<f32> access.
/// For fp16: array<u32> with unpack2x16float/pack2x16float.
inline std::string instantiateTemplate(const char* wgslTemplate, TensorDtype dtype) {
    std::string wgsl(wgslTemplate);

    // Check if template uses any markers
    if (wgsl.find("${T}") == std::string::npos) {
        return wgsl;  // Not a template, return as-is
    }

    bool isFp16 = (dtype == TensorDtype(1));  // Float16

    if (!isFp16) {
        // f32 path: simple direct access
        replaceAll(wgsl, "${T}", "f32");
        replaceAll(wgsl, "${T_DTYPE}", "f32");
        replaceAll(wgsl, "${T_BYTES}", "4");

        // Helper functions for f32
        replaceAll(wgsl, "${T_READ}", R"(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}
)");
        replaceAll(wgsl, "${T_WRITE}", R"(
fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}
)");
        replaceAll(wgsl, "${T_WRITE2}", R"(
fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}
)");
    } else {
        // fp16 path: packed u32 access via unpack2x16float / pack2x16float
        replaceAll(wgsl, "${T}", "u32");
        replaceAll(wgsl, "${T_DTYPE}", "f16");
        replaceAll(wgsl, "${T_BYTES}", "2");

        // Helper functions for fp16
        replaceAll(wgsl, "${T_READ}", R"(
fn t_read(buf: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
    let packed = (*buf)[idx / 2u];
    let pair = unpack2x16float(packed);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
)");
        replaceAll(wgsl, "${T_WRITE}", R"(
fn t_write(buf: ptr<storage, array<u32>, read_write>, idx: u32, val: f32) {
    let u32_idx = idx / 2u;
    let existing = (*buf)[u32_idx];
    let pair = unpack2x16float(existing);
    if ((idx & 1u) == 0u) {
        (*buf)[u32_idx] = pack2x16float(vec2<f32>(val, pair.y));
    } else {
        (*buf)[u32_idx] = pack2x16float(vec2<f32>(pair.x, val));
    }
}
)");
        replaceAll(wgsl, "${T_WRITE2}", R"(
fn t_write2(buf: ptr<storage, array<u32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx / 2u] = pack2x16float(vec2<f32>(v0, v1));
}
)");
    }

    return wgsl;
}

/// Instantiate template with separate read/write dtype support.
/// Useful for kernels with mixed I/O (e.g., f32 input → fp16 output).
inline std::string instantiateTemplateMixed(
    const char* wgslTemplate,
    TensorDtype inputDtype,
    TensorDtype outputDtype)
{
    std::string wgsl(wgslTemplate);

    bool inFp16 = (inputDtype == TensorDtype(1));
    bool outFp16 = (outputDtype == TensorDtype(1));

    // Input storage type
    replaceAll(wgsl, "${T_IN}", inFp16 ? "u32" : "f32");
    // Output storage type
    replaceAll(wgsl, "${T_OUT}", outFp16 ? "u32" : "f32");

    // Input read helper
    if (!inFp16) {
        replaceAll(wgsl, "${T_IN_READ}", R"(
fn t_in_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}
)");
    } else {
        replaceAll(wgsl, "${T_IN_READ}", R"(
fn t_in_read(buf: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
    let packed = (*buf)[idx / 2u];
    let pair = unpack2x16float(packed);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
)");
    }

    // Output write helper
    if (!outFp16) {
        replaceAll(wgsl, "${T_OUT_WRITE}", R"(
fn t_out_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}
)");
        replaceAll(wgsl, "${T_OUT_WRITE2}", R"(
fn t_out_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}
)");
    } else {
        replaceAll(wgsl, "${T_OUT_WRITE}", R"(
fn t_out_write(buf: ptr<storage, array<u32>, read_write>, idx: u32, val: f32) {
    let u32_idx = idx / 2u;
    let existing = (*buf)[u32_idx];
    let pair = unpack2x16float(existing);
    if ((idx & 1u) == 0u) {
        (*buf)[u32_idx] = pack2x16float(vec2<f32>(val, pair.y));
    } else {
        (*buf)[u32_idx] = pack2x16float(vec2<f32>(pair.x, val));
    }
}
)");
        replaceAll(wgsl, "${T_OUT_WRITE2}", R"(
fn t_out_write2(buf: ptr<storage, array<u32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx / 2u] = pack2x16float(vec2<f32>(v0, v1));
}
)");
    }

    return wgsl;
}
