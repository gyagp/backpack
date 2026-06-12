#pragma once

#include <dawn/webgpu_cpp.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

inline std::string load_wgsl(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "Failed to open shader: %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

inline wgpu::ShaderModule create_shader_module(const wgpu::Device& device,
                                               const std::string& wgsl_source) {
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = wgsl_source.c_str();

    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl;

    return device.CreateShaderModule(&desc);
}

inline wgpu::ComputePipeline create_compute_pipeline(const wgpu::Device& device,
                                                     const wgpu::ShaderModule& module,
                                                     const char* entry_point = "main") {
    wgpu::ComputePipelineDescriptor desc{};
    desc.compute.module = module;
    desc.compute.entryPoint = entry_point;
    return device.CreateComputePipeline(&desc);
}

inline wgpu::ComputePipeline load_compute_pipeline(const wgpu::Device& device,
                                                   const std::string& wgsl_path,
                                                   const char* entry_point = "main") {
    std::string source = load_wgsl(wgsl_path);
    wgpu::ShaderModule module = create_shader_module(device, source);
    return create_compute_pipeline(device, module, entry_point);
}

class PipelineCache {
public:
    explicit PipelineCache(const wgpu::Device& device) : device_(device) {}

    wgpu::ComputePipeline get(const std::string& wgsl_path,
                              const char* entry_point = "main") {
        std::string key = wgsl_path + ":" + entry_point;
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        auto pipeline = load_compute_pipeline(device_, wgsl_path, entry_point);
        cache_[key] = pipeline;
        return pipeline;
    }

private:
    wgpu::Device device_;
    std::unordered_map<std::string, wgpu::ComputePipeline> cache_;
};
