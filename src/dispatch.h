#pragma once

#include <dawn/webgpu_cpp.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

inline void dispatch_elementwise_enc(const wgpu::Device& device,
                                     wgpu::CommandEncoder& encoder,
                                     const wgpu::ComputePipeline& pipeline,
                                     std::initializer_list<wgpu::Buffer> bindings,
                                     uint32_t element_count,
                                     uint32_t workgroup_size = 64) {
    std::vector<wgpu::BindGroupEntry> entries;
    entries.reserve(bindings.size());
    uint32_t index = 0;
    for (const auto& buf : bindings) {
        wgpu::BindGroupEntry entry{};
        entry.binding = index++;
        entry.buffer = buf;
        entry.offset = 0;
        entry.size = buf.GetSize();
        entries.push_back(entry);
    }

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = entries.size();
    bg_desc.entries = entries.data();
    wgpu::BindGroup bind_group = device.CreateBindGroup(&bg_desc);

    uint32_t workgroup_count = (element_count + workgroup_size - 1) / workgroup_size;

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(workgroup_count);
    pass.End();
}

inline void dispatch_elementwise(const wgpu::Device& device,
                                 const wgpu::Queue& queue,
                                 const wgpu::ComputePipeline& pipeline,
                                 std::initializer_list<wgpu::Buffer> bindings,
                                 uint32_t element_count,
                                 uint32_t workgroup_size = 64) {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    dispatch_elementwise_enc(device, encoder, pipeline, bindings, element_count, workgroup_size);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
}
