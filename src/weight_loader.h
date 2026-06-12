#pragma once

#include <map>
#include <stdexcept>
#include <string>

#include "gguf_parser.h"
#include "gpu_context.h"
#include "mmap_file.h"
#include "tensor.h"

inline DType gguf_dtype_to_dtype(GGUFDType gtype) {
    switch (gtype) {
        case GGUFDType::F32:    return DType::f32;
        case GGUFDType::F16:    return DType::f16;
        case GGUFDType::Q8_0:   return DType::q8_0;
        case GGUFDType::Q4_K_M: return DType::q4_k_m;
    }
    throw std::runtime_error("Unsupported GGUF dtype: " + std::to_string(static_cast<uint32_t>(gtype)));
}

inline std::map<std::string, Tensor> load_weights(
    const GGUFFile& gguf,
    const MmapFile& mmap,
    const GpuContext& ctx)
{
    std::map<std::string, Tensor> weights;
    const uint8_t* base = mmap.data() + gguf.data_offset;

    for (const auto& info : gguf.tensors) {
        DType dtype = gguf_dtype_to_dtype(info.dtype);

        std::vector<uint32_t> shape;
        shape.reserve(info.dimensions.size());
        for (auto d : info.dimensions)
            shape.push_back(static_cast<uint32_t>(d));

        uint32_t element_count = 1;
        for (auto s : shape)
            element_count *= s;

        uint64_t byte_size = compute_buffer_size(element_count, dtype);
        const uint8_t* tensor_data = base + info.offset;

        weights.emplace(
            info.name,
            Tensor::from_data(ctx.device, ctx.queue, std::move(shape), dtype, tensor_data, byte_size));
    }

    return weights;
}
