#pragma once
/**
 * backpack.h — Public C++ API for the Backpack inference runtime.
 *
 * General-purpose ONNX execution engine on WebGPU.
 * Model-agnostic: the runtime knows inputs, outputs, and tensors —
 * not "LLM" or "image generation". Those are application concerns.
 *
 * Naming follows Dawn/WebGPU C++ style: bp::Object::Verb (PascalCase).
 *
 * Quick start:
 *   auto device  = bp::Device::Create();
 *   auto model   = bp::Model::Load(device, "model.onnx");
 *   auto session = bp::Session::Create(model);
 *   session.SetInput("x", inputTensor);
 *   session.Run();
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// DLL export for shared library builds.
#if defined(_WIN32)
#  if defined(BACKPACK_BUILDING)
#    define BP_EXPORT __declspec(dllexport)
#  else
#    define BP_EXPORT __declspec(dllimport)
#  endif
#else
#  define BP_EXPORT __attribute__((visibility("default")))
#endif

namespace bp {

// ─── Enums ──────────────────────────────────────────────────────────────────

enum class Backend { Vulkan, D3D12, Metal, Default };

enum class DataType { Float32, Float16, Int32, Int64, UInt8, Int8, Bool };

// ─── Device ─────────────────────────────────────────────────────────────────
/// GPU device. One per physical GPU. Shared by all models.

class BP_EXPORT Device {
public:
    struct Impl;

    static Device Create(Backend backend = Backend::Default);

    std::string GetName() const;
    std::string GetBackendName() const;
    void Release();

    Device() = default;
    ~Device();
    Device(Device&& o) noexcept;
    Device& operator=(Device&& o) noexcept;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Impl* GetImpl() const { return impl_.get(); }
    bool IsValid() const { return impl_ != nullptr; }

    /// Access the underlying GPUContext (for apps needing internal access).
    void* GetGPUContext() const;

private:
    std::unique_ptr<Impl> impl_;
};

// ─── Model ──────────────────────────────────────────────────────────────────
/// A compiled ONNX graph: weights uploaded, shaders compiled.
/// Immutable after creation.

class BP_EXPORT Model {
public:
    struct Impl;

    static Model Load(Device& device, const std::string& path);

    int GetInputCount() const;
    int GetOutputCount() const;
    std::string GetInputName(int index) const;
    std::string GetOutputName(int index) const;

    void Release();

    Model() = default;
    ~Model();
    Model(Model&& o) noexcept;
    Model& operator=(Model&& o) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Impl* GetImpl() const { return impl_.get(); }
    bool IsValid() const { return impl_ != nullptr; }

private:
    std::unique_ptr<Impl> impl_;
};

// ─── Tensor ─────────────────────────────────────────────────────────────────
/// A shaped, typed buffer (GPU or CPU).
/// Flows between sessions: output of one model becomes input of another.

class BP_EXPORT Tensor {
public:
    struct Impl;

    static Tensor Create(Device& device, DataType dtype,
                          const std::vector<int64_t>& shape);

    void SetData(const void* data, size_t bytes);
    void GetData(void* data, size_t bytes) const;

    /// In-place element-wise scale: buf[i] *= s. GPU-only, no CPU sync.
    void Scale(float s);
    /// In-place negate: buf[i] = -buf[i]. Shorthand for Scale(-1).
    void Negate();

    DataType GetDtype() const;
    std::vector<int64_t> GetShape() const;
    int64_t GetElementCount() const;

    void Release();

    Tensor() = default;
    ~Tensor();
    Tensor(Tensor&& o) noexcept;
    Tensor& operator=(Tensor&& o) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Impl* GetImpl() const { return impl_.get(); }
    bool IsValid() const { return impl_ != nullptr; }

private:
    std::unique_ptr<Impl> impl_;
};

// ─── Session ────────────────────────────────────────────────────────────────
/// Bind inputs, run a model, read outputs. Stateless per run.

class BP_EXPORT Session {
public:
    struct Impl;

    static Session Create(Model& model);

    void SetInput(const std::string& name, Tensor& tensor);
    void SetOutput(const std::string& name, Tensor& tensor);
    void Run();

    void Release();

    Session() = default;
    ~Session();
    Session(Session&& o) noexcept;
    Session& operator=(Session&& o) noexcept;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace bp
