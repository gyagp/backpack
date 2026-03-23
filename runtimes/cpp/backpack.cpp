/**
 * backpack.cpp — Implementation of the Backpack runtime API (Layer 1).
 *
 * Device, Model, Tensor, Session — general-purpose ONNX execution.
 * All LLM-specific logic (tokenizer, KV cache, decode) lives in the app.
 */

#include "backpack.h"
#include "gpu_context.h"

#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

static WGPUBackendType toWGPU(bp::Backend b) {
    switch (b) {
        case bp::Backend::D3D12:  return WGPUBackendType_D3D12;
        case bp::Backend::Metal:  return WGPUBackendType_Metal;
        case bp::Backend::Vulkan: return WGPUBackendType_Vulkan;
        default: {
#if defined(_WIN32)
            return WGPUBackendType_D3D12;
#elif defined(__APPLE__)
            return WGPUBackendType_Metal;
#else
            return WGPUBackendType_Vulkan;
#endif
        }
    }
}

static std::string backendStr(WGPUBackendType bt) {
    switch (bt) {
        case WGPUBackendType_D3D12: return "d3d12";
        case WGPUBackendType_Metal: return "metal";
        case WGPUBackendType_Vulkan: return "vulkan";
        default: return "unknown";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Device
// ═══════════════════════════════════════════════════════════════════════════

struct bp::Device::Impl {
    GPUContext gpu;
};

bp::Device bp::Device::Create(Backend backend) {
    Device d;
    d.impl_ = std::make_unique<Impl>();
    if (!d.impl_->gpu.init(toWGPU(backend))) {
        fprintf(stderr, "bp::Device::Create: GPU init failed\n");
        d.impl_.reset();
    }
    return d;
}

std::string bp::Device::GetName() const {
    return impl_ ? impl_->gpu.adapterName : "";
}

std::string bp::Device::GetBackendName() const {
    return impl_ ? backendStr(impl_->gpu.backendType) : "";
}

void bp::Device::Release() { impl_.reset(); }
bp::Device::~Device() = default;
bp::Device::Device(Device&& o) noexcept = default;
bp::Device& bp::Device::operator=(Device&& o) noexcept = default;

void* bp::Device::GetGPUContext() const {
    return impl_ ? &impl_->gpu : nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Model (stub — full ONNX graph execution is future work)
// ═══════════════════════════════════════════════════════════════════════════

struct bp::Model::Impl {
    Device* device = nullptr;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

bp::Model bp::Model::Load(Device& device, const std::string& path) {
    Model m;
    m.impl_ = std::make_unique<Impl>();
    m.impl_->device = &device;
    // TODO: general ONNX graph loading + compilation
    return m;
}

int bp::Model::GetInputCount() const {
    return impl_ ? (int)impl_->inputNames.size() : 0;
}
int bp::Model::GetOutputCount() const {
    return impl_ ? (int)impl_->outputNames.size() : 0;
}
std::string bp::Model::GetInputName(int index) const {
    if (!impl_ || index < 0 || index >= (int)impl_->inputNames.size()) return "";
    return impl_->inputNames[index];
}
std::string bp::Model::GetOutputName(int index) const {
    if (!impl_ || index < 0 || index >= (int)impl_->outputNames.size()) return "";
    return impl_->outputNames[index];
}

void bp::Model::Release() { impl_.reset(); }
bp::Model::~Model() = default;
bp::Model::Model(Model&& o) noexcept = default;
bp::Model& bp::Model::operator=(Model&& o) noexcept = default;

// ═══════════════════════════════════════════════════════════════════════════
// Tensor (stub)
// ═══════════════════════════════════════════════════════════════════════════

struct bp::Tensor::Impl {
    bp::DataType dtype;
    std::vector<int64_t> shape;
    GPUBuffer gpuBuf;
    Device* device = nullptr;
};

bp::Tensor bp::Tensor::Create(Device& device, DataType dtype,
                                const std::vector<int64_t>& shape) {
    Tensor t;
    t.impl_ = std::make_unique<Impl>();
    t.impl_->dtype = dtype;
    t.impl_->shape = shape;
    t.impl_->device = &device;
    return t;
}

void bp::Tensor::SetData(const void* data, size_t bytes) {
    (void)data; (void)bytes; // TODO
}
void bp::Tensor::GetData(void* data, size_t bytes) const {
    (void)data; (void)bytes; // TODO
}
bp::DataType bp::Tensor::GetDtype() const {
    return impl_ ? impl_->dtype : DataType::Float32;
}
std::vector<int64_t> bp::Tensor::GetShape() const {
    return impl_ ? impl_->shape : std::vector<int64_t>{};
}
int64_t bp::Tensor::GetElementCount() const {
    if (!impl_) return 0;
    int64_t n = 1;
    for (auto d : impl_->shape) n *= d;
    return n;
}

void bp::Tensor::Release() { impl_.reset(); }
bp::Tensor::~Tensor() = default;
bp::Tensor::Tensor(Tensor&& o) noexcept = default;
bp::Tensor& bp::Tensor::operator=(Tensor&& o) noexcept = default;

// ═══════════════════════════════════════════════════════════════════════════
// Session (stub)
// ═══════════════════════════════════════════════════════════════════════════

struct bp::Session::Impl {
    Model* model = nullptr;
};

bp::Session bp::Session::Create(Model& model) {
    Session s;
    s.impl_ = std::make_unique<Impl>();
    s.impl_->model = &model;
    return s;
}

void bp::Session::SetInput(const std::string& name, Tensor& tensor) {
    (void)name; (void)tensor; // TODO
}
void bp::Session::SetOutput(const std::string& name, Tensor& tensor) {
    (void)name; (void)tensor; // TODO
}
void bp::Session::Run() {
    // TODO: execute ONNX graph on GPU
}

void bp::Session::Release() { impl_.reset(); }
bp::Session::~Session() = default;
bp::Session::Session(Session&& o) noexcept = default;
bp::Session& bp::Session::operator=(Session&& o) noexcept = default;
