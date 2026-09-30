#include "native_quant.h"
#include "quants_iq_tables.h"
#include "native_quant_source.h"
#include <stdexcept>

namespace {
struct Layout { uint32_t elements, bytes; const void* codebook; size_t codebookBytes; };
Layout layout(GGUFType type) {
    switch (type) {
        case GGUF_TYPE_Q2_K: return {256, 84, nullptr, 0};
        case GGUF_TYPE_Q3_K: return {256, 110, nullptr, 0};
        case GGUF_TYPE_Q4_K: return {256, 144, nullptr, 0};
        case GGUF_TYPE_Q5_K: return {256, 176, nullptr, 0};
        case GGUF_TYPE_Q6_K: return {256, 210, nullptr, 0};
        case GGUF_TYPE_Q8_0: return {32, 34, nullptr, 0};
        case GGUF_TYPE_IQ2_XXS: return {256, 66, iq2xxs_grid, sizeof(iq2xxs_grid)};
        case GGUF_TYPE_IQ2_XS: return {256, 74, iq2xs_grid, sizeof(iq2xs_grid)};
        case GGUF_TYPE_IQ3_XXS: return {256, 98, iq3xxs_grid, sizeof(iq3xxs_grid)};
        case GGUF_TYPE_IQ1_S: return {256, 50, iq1s_grid, sizeof(iq1s_grid)};
        case GGUF_TYPE_IQ4_NL: return {32, 18, nullptr, 0};
        case GGUF_TYPE_IQ3_S: return {256, 110, iq3s_grid, sizeof(iq3s_grid)};
        case GGUF_TYPE_IQ2_S: return {256, 82, iq2s_grid, sizeof(iq2s_grid)};
        case GGUF_TYPE_IQ4_XS: return {256, 136, nullptr, 0};
        default: return {};
    }
}
}

bool supportsNativeQuant(GGUFType type) { return layout(type).bytes != 0; }

KQuantPacked pack_native_quant(const void* raw, uint32_t rows, uint32_t cols, GGUFType type) {
    const auto spec = layout(type);
    if (!spec.bytes || !rows || !cols || cols % spec.elements)
        throw std::runtime_error("Unsupported native quantization shape/type");
    KQuantPacked out;
    out.N = rows; out.K = cols; out.nBlocks = cols / spec.elements;
    const size_t rowBytes = size_t(out.nBlocks) * spec.bytes;
    out.rowStrideWords = uint32_t((rowBytes + 3) / 4);
    const size_t weightWords = size_t(rows) * out.rowStrideWords;
    out.data.resize(weightWords + spec.codebookBytes / 4, 0);
    for (uint32_t row = 0; row < rows; ++row)
        memcpy(out.data.data() + size_t(row) * out.rowStrideWords,
               static_cast<const uint8_t*>(raw) + size_t(row) * rowBytes, rowBytes);
    if (spec.codebookBytes)
        memcpy(out.data.data() + weightWords, spec.codebook, spec.codebookBytes);
    return out;
}

std::string nativeQuantShader(GGUFType type, bool gather, bool prefill) {
    const auto spec = layout(type);
    if (!spec.bytes) throw std::runtime_error("Unsupported native quantization shader");
    std::string source = NATIVE_QUANT_SOURCE;
    if (prefill) {
        const auto main = source.find("\nvar<workgroup> sums:");
        if (gather || main == std::string::npos)
            throw std::runtime_error("Invalid native prefill shader assembly");
        source.erase(main);
        source += NATIVE_QUANT_PREFILL_SOURCE;
    }
    for (const auto& entry : {std::pair<std::string, std::string>{"__TYPE__", std::to_string(type)},
                             {"__BLOCK_BYTES__", std::to_string(spec.bytes)},
                             {"__BLOCK_ELEMENTS__", std::to_string(spec.elements)},
                             {"__GATHER__", gather ? "true" : "false"}}) {
        size_t offset = 0;
        while ((offset = source.find(entry.first, offset)) != std::string::npos) {
            source.replace(offset, entry.first.size(), entry.second);
            offset += entry.second.size();
        }
    }
    return source;
}
