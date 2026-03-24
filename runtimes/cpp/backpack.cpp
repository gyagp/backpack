/**
 * backpack.cpp — Implementation of the Backpack runtime API (Layer 1).
 *
 * Device, Model, Tensor, Session — general-purpose ONNX execution.
 * All LLM-specific logic (tokenizer, KV cache, decode) lives in the app.
 */

#include "backpack.h"
#include "gpu_context.h"
#include "graph_executor.h"

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
    GraphExecutor executor;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

bp::Model bp::Model::Load(Device& device, const std::string& path) {
    if (!device.IsValid()) return {};
    Model m;
    m.impl_ = std::make_unique<Impl>();
    m.impl_->device = &device;

    auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
    if (!m.impl_->executor.Load(*gpuCtx, path)) {
        fprintf(stderr, "bp::Model::Load: failed to load %s\n", path.c_str());
        m.impl_.reset();
        return m;
    }

    // Populate input/output names from parsed graph
    auto& graph = m.impl_->executor.GetGraph();
    for (auto& inp : graph.inputs)
        m.impl_->inputNames.push_back(inp.name);
    for (auto& out : graph.outputs)
        m.impl_->outputNames.push_back(out.name);

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
    TensorDtype geDtype; // graph executor dtype
};

static TensorDtype toGeDtype(bp::DataType dt) {
    switch (dt) {
        case bp::DataType::Float32: return TensorDtype::Float32;
        case bp::DataType::Float16: return TensorDtype::Float16;
        case bp::DataType::Int32: return TensorDtype::Int32;
        case bp::DataType::Int64: return TensorDtype::Int64;
        case bp::DataType::UInt8: return TensorDtype::UInt8;
        case bp::DataType::Int8: return TensorDtype::Int8;
        case bp::DataType::Bool: return TensorDtype::Bool;
    }
    return TensorDtype::Float32;
}

static size_t dtypeSize(bp::DataType dt) {
    switch (dt) {
        case bp::DataType::Float32: case bp::DataType::Int32: return 4;
        case bp::DataType::Float16: return 2;
        case bp::DataType::Int64: return 8;
        case bp::DataType::UInt8: case bp::DataType::Int8: case bp::DataType::Bool: return 1;
    }
    return 4;
}

bp::Tensor bp::Tensor::Create(Device& device, DataType dtype,
                                const std::vector<int64_t>& shape) {
    if (!device.IsValid()) return {};
    Tensor t;
    t.impl_ = std::make_unique<Impl>();
    t.impl_->dtype = dtype;
    t.impl_->shape = shape;
    t.impl_->device = &device;
    t.impl_->geDtype = toGeDtype(dtype);

    int64_t nel = 1;
    for (auto d : shape) nel *= d;
    size_t bytes = (size_t)nel * dtypeSize(dtype);
    if (bytes == 0) bytes = 4;

    auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
    t.impl_->gpuBuf = gpuCtx->createBuffer("tensor", bytes);
    return t;
}

void bp::Tensor::SetData(const void* data, size_t bytes) {
    if (!impl_ || !impl_->device) return;
    auto* gpuCtx = static_cast<GPUContext*>(impl_->device->GetGPUContext());
    gpuCtx->writeBuffer(impl_->gpuBuf, data, bytes);
}

void bp::Tensor::GetData(void* data, size_t bytes) const {
    if (!impl_ || !impl_->device) return;
    auto* gpuCtx = static_cast<GPUContext*>(impl_->device->GetGPUContext());
    auto readback = gpuCtx->readBuffer(impl_->gpuBuf, bytes);
    memcpy(data, readback.data(), std::min(bytes, readback.size()));
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
    std::vector<GpuTensor> ownedInputs;
    std::vector<GpuTensor> ownedOutputs;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<Tensor*> outputTensors;  // back-pointers for buffer propagation
};

bp::Session bp::Session::Create(Model& model) {
    Session s;
    s.impl_ = std::make_unique<Impl>();
    s.impl_->model = &model;
    return s;
}

void bp::Session::SetInput(const std::string& name, Tensor& tensor) {
    if (!impl_ || !tensor.GetImpl()) return;
    auto* ti = tensor.GetImpl();
    GpuTensor gt;
    gt.buffer = ti->gpuBuf;
    gt.shape = ti->shape;
    gt.dtype = ti->geDtype;
    impl_->ownedInputs.push_back(gt);
    // Don't store pointer yet — will resolve after all SetInput calls
    impl_->inputNames.push_back(name);
}

void bp::Session::SetOutput(const std::string& name, Tensor& tensor) {
    if (!impl_ || !tensor.GetImpl()) return;
    auto* ti = tensor.GetImpl();
    GpuTensor gt;
    gt.buffer = ti->gpuBuf;
    gt.shape = ti->shape;
    gt.dtype = ti->geDtype;
    impl_->ownedOutputs.push_back(gt);
    impl_->outputNames.push_back(name);
    impl_->outputTensors.push_back(&tensor);
}

void bp::Session::Run() {
    if (!impl_ || !impl_->model || !impl_->model->GetImpl()) return;

    // Build input/output maps from owned vectors (pointers stable after all push_backs)
    std::unordered_map<std::string, GpuTensor*> inputs;
    for (size_t i = 0; i < impl_->inputNames.size(); i++)
        inputs[impl_->inputNames[i]] = &impl_->ownedInputs[i];

    std::unordered_map<std::string, GpuTensor*> outputs;
    for (size_t i = 0; i < impl_->outputNames.size(); i++)
        outputs[impl_->outputNames[i]] = &impl_->ownedOutputs[i];

    impl_->model->GetImpl()->executor.Execute(inputs, outputs);

    // Propagate output buffer handles back to the Tensor objects
    // Execute may have replaced the buffer handle with the actual computed result
    for (size_t i = 0; i < impl_->outputTensors.size(); i++) {
        if (impl_->outputTensors[i]) {
            auto* ti = impl_->outputTensors[i]->GetImpl();
            if (ti) {
                ti->gpuBuf = impl_->ownedOutputs[i].buffer;
                ti->shape = impl_->ownedOutputs[i].shape;
            }
        }
    }
}

void bp::Session::Release() { impl_.reset(); }
bp::Session::~Session() = default;
bp::Session::Session(Session&& o) noexcept = default;
bp::Session& bp::Session::operator=(Session&& o) noexcept = default;
