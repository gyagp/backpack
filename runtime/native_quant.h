#pragma once
#include "gguf_loader.h"

// Lossless GPU storage and scalar decoding for the mixed quantization types
// used by Unsloth. Codebooks follow the row-padded weight data in one buffer.
bool supportsNativeQuant(GGUFType type);
KQuantPacked pack_native_quant(const void* raw, uint32_t rows, uint32_t cols, GGUFType type);
std::string nativeQuantShader(GGUFType type, bool gather = false, bool prefill = false,
                              uint32_t prefillRows = 16);

// Staged prefill parameters: K, packed N, blocks/row, packed stride,
// source column offset, output stride, M, staged columns, output offset.
std::string nativeQuantDecodeSliceShader(GGUFType type);
std::string nativeQuantDensePrefillShader();
