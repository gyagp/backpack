#pragma once

#include <dawn/webgpu_cpp.h>
#include <cstdint>

#include "buffer_utils.h"
#include "shader_utils.h"

inline uint32_t gpu_argmax(const wgpu::Device& device,
                           const wgpu::Queue& queue,
                           PipelineCache& pipeline_cache,
                           const wgpu::Buffer& logits,
                           uint32_t vocab_size) {
    static wgpu::ComputePipeline pipeline;
    if (!pipeline) {
        pipeline = pipeline_cache.get("src/shaders/argmax.wgsl");
    }

    auto result_buf = create_buffer(device, sizeof(uint32_t),
                                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                    nullptr);

    auto params_buf = create_buffer(device, sizeof(uint32_t),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                    &vocab_size);

    wgpu::BindGroupEntry entries[3];
    entries[0] = {}; entries[0].binding = 0; entries[0].buffer = logits;
    entries[0].offset = 0; entries[0].size = vocab_size * sizeof(float);
    entries[1] = {}; entries[1].binding = 1; entries[1].buffer = result_buf;
    entries[1].offset = 0; entries[1].size = sizeof(uint32_t);
    entries[2] = {}; entries[2].binding = 2; entries[2].buffer = params_buf;
    entries[2].offset = 0; entries[2].size = sizeof(uint32_t);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bg = device.CreateBindGroup(&bg_desc);

    wgpu::CommandEncoder enc = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(1);
    pass.End();
    wgpu::CommandBuffer cmd = enc.Finish();
    queue.Submit(1, &cmd);

    auto result_bytes = read_buffer(device, queue, result_buf, sizeof(uint32_t));
    uint32_t token_id;
    std::memcpy(&token_id, result_bytes.data(), sizeof(uint32_t));
    return token_id;
}
