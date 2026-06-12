#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <dawn/webgpu_cpp.h>

#include "dispatch.h"
#include "gpu_context.h"
#include "onnx_loader.h"
#include "shader_utils.h"

namespace onnx_runtime {

struct GpuTensor {
    wgpu::Buffer buffer;
    std::vector<uint32_t> shape;
    uint32_t element_count() const {
        uint32_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class OnnxRuntime {
public:
    OnnxRuntime(const GpuContext& ctx, const std::string& shader_dir)
        : device_(ctx.device), queue_(ctx.queue), cache_(ctx.device), shader_dir_(shader_dir) {}

    std::unordered_map<std::string, std::vector<float>> run(
        const onnx::GraphProto& graph,
        const std::unordered_map<std::string, std::vector<float>>& inputs) {

        tensors_.clear();

        for (auto& init : graph.initializers) {
            upload_initializer(init);
        }

        for (auto& [name, data] : inputs) {
            upload_input(name, data, graph);
        }

        for (auto& node : graph.nodes) {
            dispatch_node(node);
        }

        std::unordered_map<std::string, std::vector<float>> results;
        for (auto& out : graph.outputs) {
            results[out.name] = readback(out.name);
        }
        return results;
    }

private:
    wgpu::Device device_;
    wgpu::Queue queue_;
    PipelineCache cache_;
    std::string shader_dir_;
    std::unordered_map<std::string, GpuTensor> tensors_;

    GpuTensor& get_tensor(const std::string& name) {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("OnnxRuntime: tensor not found: " + name);
        return it->second;
    }

    void upload_initializer(const onnx::TensorProto& tp) {
        std::vector<uint32_t> shape;
        for (auto d : tp.dims) shape.push_back(static_cast<uint32_t>(d));

        uint32_t count = 1;
        for (auto d : shape) count *= d;

        const void* data = nullptr;
        size_t byte_size = 0;

        if (tp.raw_data && tp.raw_data_size > 0) {
            data = tp.raw_data;
            byte_size = tp.raw_data_size;
        } else if (!tp.float_data.empty()) {
            data = tp.float_data.data();
            byte_size = tp.float_data.size() * sizeof(float);
        }

        auto buf = create_buffer(byte_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        if (data) queue_.WriteBuffer(buf, 0, data, byte_size);

        tensors_[tp.name] = GpuTensor{buf, shape};
    }

    void upload_input(const std::string& name, const std::vector<float>& data,
                      const onnx::GraphProto& graph) {
        std::vector<uint32_t> shape;
        for (auto& vi : graph.inputs) {
            if (vi.name == name) {
                for (auto d : vi.shape.dims) shape.push_back(static_cast<uint32_t>(d));
                break;
            }
        }
        if (shape.empty()) {
            uint32_t n = static_cast<uint32_t>(data.size());
            shape = {n};
        }

        size_t byte_size = data.size() * sizeof(float);
        auto buf = create_buffer(byte_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(buf, 0, data.data(), byte_size);
        tensors_[name] = GpuTensor{buf, shape};
    }

    wgpu::Buffer create_buffer(uint64_t size, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor desc{};
        desc.size = size;
        desc.usage = usage;
        return device_.CreateBuffer(&desc);
    }

    wgpu::Buffer create_uniform(const void* data, uint64_t size) {
        wgpu::BufferDescriptor desc{};
        desc.size = size;
        desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        auto buf = device_.CreateBuffer(&desc);
        queue_.WriteBuffer(buf, 0, data, size);
        return buf;
    }

    void dispatch_node(const onnx::NodeProto& node) {
        const auto& op = node.op_type;
        if (op == "MatMul")              dispatch_matmul(node);
        else if (op == "Add")            dispatch_add(node);
        else if (op == "Relu")           dispatch_relu(node);
        else if (op == "Softmax")        dispatch_softmax(node);
        else if (op == "LayerNormalization") dispatch_layernorm(node);
        else if (op == "Gather")         dispatch_gather(node);
        else if (op == "Mul")            dispatch_mul(node);
        else if (op == "Div")            dispatch_div(node);
        else if (op == "Tanh")           dispatch_tanh(node);
        else if (op == "Gelu")           dispatch_gelu(node);
        else if (op == "Reshape")        dispatch_reshape(node);
        else if (op == "Transpose")      dispatch_transpose(node);
        else if (op == "Cast")           dispatch_cast(node);
        else if (op == "Unsqueeze")      dispatch_unsqueeze(node);
        else if (op == "Squeeze")        dispatch_squeeze(node);
        else if (op == "Concat")         dispatch_concat(node);
        else if (op == "Split")          dispatch_split(node);
        else if (op == "Conv")           dispatch_conv(node);
        else if (op == "RotaryEmbedding") dispatch_rotary_embedding(node);
        else if (op == "Sigmoid")        dispatch_sigmoid(node);
        else if (op == "Clip")           dispatch_clip(node);
        else if (op == "Sub")            dispatch_sub(node);
        else if (op == "Sqrt")           dispatch_sqrt(node);
        else if (op == "Pow")            dispatch_pow(node);
        else if (op == "ReduceMean")     dispatch_reducemean(node);
        else if (op == "Where")          dispatch_where(node);
        else if (op == "Equal")          dispatch_equal(node);
        else if (op == "Expand")         dispatch_expand(node);
        else if (op == "Flatten")        dispatch_flatten(node);
        else if (op == "Gemm")           dispatch_gemm(node);
        else if (op == "Shape")          dispatch_shape(node);
        else if (op == "Slice")          dispatch_slice(node);
        else if (op == "Neg")            dispatch_neg(node);
        else if (op == "Abs")            dispatch_abs(node);
        else if (op == "Log")            dispatch_log(node);
        else if (op == "Exp")            dispatch_exp(node);
        else if (op == "Erf")            dispatch_erf(node);
        else if (op == "Identity")       dispatch_identity(node);
        else if (op == "SkipLayerNormalization") dispatch_layernorm(node);
        else throw std::runtime_error("OnnxRuntime: unsupported op: " + op);
    }

    void dispatch_matmul(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);

        uint32_t M = A.shape.size() >= 2 ? A.shape[A.shape.size() - 2] : 1;
        uint32_t K = A.shape.back();
        uint32_t N = B.shape.back();

        uint32_t out_count = M * N;
        auto out_buf = create_buffer(out_count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        struct { uint32_t M, N, K; } params = {M, N, K};
        auto param_buf = create_uniform(&params, sizeof(params));

        auto pipeline = cache_.get(shader_dir_ + "/matmul_f32.wgsl");

        std::vector<wgpu::BindGroupEntry> entries(4);
        wgpu::Buffer bufs[] = {A.buffer, B.buffer, out_buf, param_buf};
        for (int i = 0; i < 4; i++) {
            entries[i].binding = i;
            entries[i].buffer = bufs[i];
            entries[i].offset = 0;
            entries[i].size = bufs[i].GetSize();
        }

        wgpu::BindGroupDescriptor bg{};
        bg.layout = pipeline.GetBindGroupLayout(0);
        bg.entryCount = 4;
        bg.entries = entries.data();
        auto bind_group = device_.CreateBindGroup(&bg);

        uint32_t wg_x = (M + 15) / 16;
        uint32_t wg_y = (N + 15) / 16;

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(wg_x, wg_y);
        pass.End();
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        std::vector<uint32_t> out_shape;
        if (A.shape.size() >= 2) {
            out_shape = A.shape;
            out_shape.back() = N;
            out_shape[out_shape.size() - 2] = M;
        } else {
            out_shape = {M, N};
        }
        tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
    }

    void dispatch_add(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);
        uint32_t count = A.element_count();

        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        auto pipeline = cache_.get(shader_dir_ + "/add.wgsl");
        dispatch_elementwise(device_, queue_, pipeline, {A.buffer, B.buffer, out_buf}, count, 64);

        tensors_[node.outputs[0]] = GpuTensor{out_buf, A.shape};
    }

    void dispatch_relu(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();

        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        auto pipeline = cache_.get(shader_dir_ + "/relu.wgsl");
        dispatch_elementwise(device_, queue_, pipeline, {X.buffer, out_buf}, count, 64);

        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_softmax(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t num_rows = 1;
        uint32_t row_len = X.shape.back();
        for (size_t i = 0; i + 1 < X.shape.size(); i++) num_rows *= X.shape[i];

        auto out_buf = create_buffer(X.element_count() * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        uint32_t params[2] = {row_len, num_rows};
        auto param_buf = create_uniform(params, sizeof(params));

        auto pipeline = cache_.get(shader_dir_ + "/softmax.wgsl");

        std::vector<wgpu::BindGroupEntry> entries(3);
        wgpu::Buffer bufs[] = {X.buffer, out_buf, param_buf};
        for (int i = 0; i < 3; i++) {
            entries[i].binding = i;
            entries[i].buffer = bufs[i];
            entries[i].offset = 0;
            entries[i].size = bufs[i].GetSize();
        }

        wgpu::BindGroupDescriptor bg{};
        bg.layout = pipeline.GetBindGroupLayout(0);
        bg.entryCount = 3;
        bg.entries = entries.data();
        auto bind_group = device_.CreateBindGroup(&bg);

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(num_rows);
        pass.End();
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_layernorm(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        auto& scale = get_tensor(node.inputs[1]);
        auto& bias = get_tensor(node.inputs[2]);

        uint32_t row_len = X.shape.back();
        uint32_t num_rows = X.element_count() / row_len;

        float epsilon = 1e-5f;
        for (auto& attr : node.attributes) {
            if (attr.name == "epsilon") epsilon = attr.f;
        }

        auto out_buf = create_buffer(X.element_count() * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        struct { uint32_t row_length; float epsilon; } params = {row_len, epsilon};
        auto param_buf = create_uniform(&params, sizeof(params));

        auto pipeline = cache_.get(shader_dir_ + "/layernorm.wgsl");

        std::vector<wgpu::BindGroupEntry> entries(5);
        wgpu::Buffer bufs[] = {X.buffer, scale.buffer, bias.buffer, out_buf, param_buf};
        for (int i = 0; i < 5; i++) {
            entries[i].binding = i;
            entries[i].buffer = bufs[i];
            entries[i].offset = 0;
            entries[i].size = bufs[i].GetSize();
        }

        wgpu::BindGroupDescriptor bg{};
        bg.layout = pipeline.GetBindGroupLayout(0);
        bg.entryCount = 5;
        bg.entries = entries.data();
        auto bind_group = device_.CreateBindGroup(&bg);

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(num_rows);
        pass.End();
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_gather(const onnx::NodeProto& node) {
        auto& data = get_tensor(node.inputs[0]);
        auto& indices = get_tensor(node.inputs[1]);

        int64_t axis = 0;
        for (auto& attr : node.attributes) {
            if (attr.name == "axis") axis = attr.i;
        }

        uint32_t axis_dim = data.shape[static_cast<size_t>(axis)];
        uint32_t inner_dim = 1;
        for (size_t i = static_cast<size_t>(axis) + 1; i < data.shape.size(); i++)
            inner_dim *= data.shape[i];

        uint32_t index_count = indices.element_count();
        uint32_t out_count = index_count * inner_dim;

        auto out_buf = create_buffer(out_count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        struct { uint32_t axis_dim, inner_dim, index_count; } params = {axis_dim, inner_dim, index_count};
        auto param_buf = create_uniform(&params, sizeof(params));

        auto pipeline = cache_.get(shader_dir_ + "/gather.wgsl");

        std::vector<wgpu::BindGroupEntry> entries(4);
        wgpu::Buffer bufs[] = {data.buffer, indices.buffer, out_buf, param_buf};
        for (int i = 0; i < 4; i++) {
            entries[i].binding = i;
            entries[i].buffer = bufs[i];
            entries[i].offset = 0;
            entries[i].size = bufs[i].GetSize();
        }

        wgpu::BindGroupDescriptor bg{};
        bg.layout = pipeline.GetBindGroupLayout(0);
        bg.entryCount = 4;
        bg.entries = entries.data();
        auto bind_group = device_.CreateBindGroup(&bg);

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups((out_count + 63) / 64);
        pass.End();
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        std::vector<uint32_t> out_shape;
        for (size_t i = 0; i < static_cast<size_t>(axis); i++)
            out_shape.push_back(data.shape[i]);
        for (auto d : indices.shape)
            out_shape.push_back(d);
        for (size_t i = static_cast<size_t>(axis) + 1; i < data.shape.size(); i++)
            out_shape.push_back(data.shape[i]);

        tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
    }

    // --- Elementwise binary ops ---

    void dispatch_mul(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);
        uint32_t count = A.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto pipeline = cache_.get(shader_dir_ + "/mul.wgsl");
        dispatch_elementwise(device_, queue_, pipeline, {A.buffer, B.buffer, out_buf}, count, 64);
        tensors_[node.outputs[0]] = GpuTensor{out_buf, A.shape};
    }

    void dispatch_div(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);
        uint32_t count = A.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto pipeline = cache_.get(shader_dir_ + "/div.wgsl");
        dispatch_elementwise(device_, queue_, pipeline, {A.buffer, B.buffer, out_buf}, count, 64);
        tensors_[node.outputs[0]] = GpuTensor{out_buf, A.shape};
    }

    void dispatch_sub(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);
        uint32_t count = A.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        // sub = a + (-b), reuse add pattern but need sub shader — inline via scale trick
        // Actually just copy A and subtract: write directly
        // Use a simple GPU copy + negate approach, or create sub shader inline
        // For simplicity, do CPU fallback for now — create sub.wgsl later if perf matters
        auto a_data = readback_raw(node.inputs[0]);
        auto b_data = readback_raw(node.inputs[1]);
        for (size_t i = 0; i < a_data.size(); i++) a_data[i] -= b_data[i];
        queue_.WriteBuffer(out_buf, 0, a_data.data(), a_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, A.shape};
    }

    // --- Elementwise unary ops ---

    void dispatch_tanh(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto pipeline = cache_.get(shader_dir_ + "/tanh.wgsl");
        dispatch_elementwise(device_, queue_, pipeline, {X.buffer, out_buf}, count, 64);
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_gelu(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto pipeline = cache_.get(shader_dir_ + "/gelu.wgsl");
        dispatch_elementwise(device_, queue_, pipeline, {X.buffer, out_buf}, count, 64);
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_sigmoid(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_neg(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = -data[i];
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_abs(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = std::abs(data[i]);
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_log(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = std::log(data[i]);
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_exp(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = std::exp(data[i]);
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_erf(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = std::erf(data[i]);
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_sqrt(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++) data[i] = std::sqrt(data[i]);
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_pow(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        auto& Y = get_tensor(node.inputs[1]);
        uint32_t count = X.element_count();
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto x_data = readback_raw(node.inputs[0]);
        auto y_data = readback_raw(node.inputs[1]);
        bool scalar_exp = (Y.element_count() == 1);
        for (size_t i = 0; i < x_data.size(); i++)
            x_data[i] = std::pow(x_data[i], scalar_exp ? y_data[0] : y_data[i]);
        queue_.WriteBuffer(out_buf, 0, x_data.data(), x_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_clip(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();
        float min_val = -std::numeric_limits<float>::infinity();
        float max_val = std::numeric_limits<float>::infinity();
        if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
            auto mn = readback_raw(node.inputs[1]);
            if (!mn.empty()) min_val = mn[0];
        }
        if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
            auto mx = readback_raw(node.inputs[2]);
            if (!mx.empty()) max_val = mx[0];
        }
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto data = readback_raw(node.inputs[0]);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = (std::max)(min_val, (std::min)(max_val, data[i]));
        queue_.WriteBuffer(out_buf, 0, data.data(), data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    void dispatch_identity(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        tensors_[node.outputs[0]] = GpuTensor{X.buffer, X.shape};
    }

    // --- Shape manipulation ops (metadata-only, no GPU compute) ---

    void dispatch_reshape(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        auto shape_data = readback_raw_int64(node.inputs[1]);
        uint32_t total = X.element_count();

        std::vector<uint32_t> new_shape;
        int neg_idx = -1;
        uint32_t known = 1;
        for (size_t i = 0; i < shape_data.size(); i++) {
            if (shape_data[i] == -1) {
                neg_idx = static_cast<int>(i);
                new_shape.push_back(0);
            } else if (shape_data[i] == 0) {
                uint32_t d = (i < X.shape.size()) ? X.shape[i] : 1;
                new_shape.push_back(d);
                known *= d;
            } else {
                new_shape.push_back(static_cast<uint32_t>(shape_data[i]));
                known *= new_shape.back();
            }
        }
        if (neg_idx >= 0) {
            new_shape[neg_idx] = total / known;
        }

        tensors_[node.outputs[0]] = GpuTensor{X.buffer, new_shape};
    }

    void dispatch_unsqueeze(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        std::vector<int64_t> axes;
        if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
            axes = readback_raw_int64(node.inputs[1]);
        } else {
            for (auto& attr : node.attributes)
                if (attr.name == "axes") axes = attr.ints;
        }
        auto new_shape = X.shape;
        std::vector<int64_t> sorted_axes = axes;
        std::sort(sorted_axes.begin(), sorted_axes.end());
        for (auto a : sorted_axes) {
            int64_t pos = a < 0 ? static_cast<int64_t>(new_shape.size() + 1) + a : a;
            new_shape.insert(new_shape.begin() + pos, 1);
        }
        tensors_[node.outputs[0]] = GpuTensor{X.buffer, new_shape};
    }

    void dispatch_squeeze(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        std::vector<int64_t> axes;
        if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
            axes = readback_raw_int64(node.inputs[1]);
        } else {
            for (auto& attr : node.attributes)
                if (attr.name == "axes") axes = attr.ints;
        }
        auto new_shape = X.shape;
        if (axes.empty()) {
            new_shape.erase(std::remove(new_shape.begin(), new_shape.end(), 1u), new_shape.end());
        } else {
            std::vector<int64_t> sorted_axes = axes;
            for (auto& a : sorted_axes) if (a < 0) a += static_cast<int64_t>(X.shape.size());
            std::sort(sorted_axes.rbegin(), sorted_axes.rend());
            for (auto a : sorted_axes) new_shape.erase(new_shape.begin() + a);
        }
        if (new_shape.empty()) new_shape.push_back(1);
        tensors_[node.outputs[0]] = GpuTensor{X.buffer, new_shape};
    }

    void dispatch_flatten(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        int64_t axis = 1;
        for (auto& attr : node.attributes)
            if (attr.name == "axis") axis = attr.i;
        if (axis < 0) axis += static_cast<int64_t>(X.shape.size());
        uint32_t d0 = 1, d1 = 1;
        for (int64_t i = 0; i < axis; i++) d0 *= X.shape[i];
        for (size_t i = static_cast<size_t>(axis); i < X.shape.size(); i++) d1 *= X.shape[i];
        tensors_[node.outputs[0]] = GpuTensor{X.buffer, {d0, d1}};
    }

    void dispatch_cast(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        tensors_[node.outputs[0]] = GpuTensor{X.buffer, X.shape};
    }

    void dispatch_expand(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        auto shape_data = readback_raw_int64(node.inputs[1]);
        std::vector<uint32_t> target;
        for (auto d : shape_data) target.push_back(static_cast<uint32_t>(d));

        uint32_t out_count = 1;
        for (auto d : target) out_count *= d;

        if (out_count == X.element_count()) {
            tensors_[node.outputs[0]] = GpuTensor{X.buffer, target};
            return;
        }

        auto out_buf = create_buffer(out_count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        auto x_data = readback_raw(node.inputs[0]);
        std::vector<float> out_data(out_count);

        size_t x_ndim = X.shape.size();
        size_t out_ndim = target.size();
        size_t pad = out_ndim - x_ndim;

        for (uint32_t i = 0; i < out_count; i++) {
            uint32_t remaining = i;
            uint32_t x_idx = 0;
            uint32_t x_stride = 1;
            for (size_t d = 0; d < x_ndim; d++) x_stride *= X.shape[d];
            for (size_t d = 0; d < out_ndim; d++) {
                uint32_t out_stride = 1;
                for (size_t k = d + 1; k < out_ndim; k++) out_stride *= target[k];
                uint32_t coord = remaining / out_stride;
                remaining %= out_stride;
                if (d >= pad) {
                    size_t xd = d - pad;
                    x_stride /= X.shape[xd];
                    x_idx += (X.shape[xd] == 1 ? 0 : coord) * x_stride;
                }
            }
            out_data[i] = x_data[x_idx];
        }
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, target};
    }

    // --- Transpose ---

    void dispatch_transpose(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        size_t ndim = X.shape.size();

        std::vector<int64_t> perm;
        for (auto& attr : node.attributes)
            if (attr.name == "perm") perm = attr.ints;
        if (perm.empty()) {
            for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; i--) perm.push_back(i);
        }

        std::vector<uint32_t> out_shape(ndim);
        for (size_t i = 0; i < ndim; i++)
            out_shape[i] = X.shape[static_cast<size_t>(perm[i])];

        std::vector<uint32_t> in_strides(ndim, 1);
        for (int i = static_cast<int>(ndim) - 2; i >= 0; i--)
            in_strides[i] = in_strides[i + 1] * X.shape[i + 1];

        std::vector<uint32_t> perm_strides(ndim);
        for (size_t i = 0; i < ndim; i++)
            perm_strides[i] = in_strides[static_cast<size_t>(perm[i])];

        uint32_t total = X.element_count();

        if (ndim <= 4) {
            uint32_t params[12] = {};
            params[0] = total;
            params[1] = static_cast<uint32_t>(ndim);
            for (size_t i = 0; i < ndim; i++) params[4 + i] = out_shape[i];
            for (size_t i = ndim; i < 4; i++) params[4 + i] = 1;
            for (size_t i = 0; i < ndim; i++) params[8 + i] = perm_strides[i];
            for (size_t i = ndim; i < 4; i++) params[8 + i] = 1;

            auto out_buf = create_buffer(total * sizeof(float),
                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
            auto param_buf = create_uniform(params, sizeof(params));
            auto pipeline = cache_.get(shader_dir_ + "/transpose.wgsl");

            std::vector<wgpu::BindGroupEntry> entries(3);
            wgpu::Buffer bufs[] = {X.buffer, out_buf, param_buf};
            for (int i = 0; i < 3; i++) {
                entries[i].binding = i;
                entries[i].buffer = bufs[i];
                entries[i].offset = 0;
                entries[i].size = bufs[i].GetSize();
            }

            wgpu::BindGroupDescriptor bg{};
            bg.layout = pipeline.GetBindGroupLayout(0);
            bg.entryCount = 3;
            bg.entries = entries.data();
            auto bind_group = device_.CreateBindGroup(&bg);

            wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
            auto pass = encoder.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, bind_group);
            pass.DispatchWorkgroups((total + 63) / 64);
            pass.End();
            auto cmd = encoder.Finish();
            queue_.Submit(1, &cmd);

            tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
        } else {
            auto data = readback_raw(node.inputs[0]);
            std::vector<float> out_data(total);
            std::vector<uint32_t> out_strides(ndim, 1);
            for (int i = static_cast<int>(ndim) - 2; i >= 0; i--)
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];

            for (uint32_t idx = 0; idx < total; idx++) {
                uint32_t rem = idx;
                uint32_t in_idx = 0;
                for (size_t d = 0; d < ndim; d++) {
                    uint32_t coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    in_idx += coord * perm_strides[d];
                }
                out_data[idx] = data[in_idx];
            }
            auto out_buf = create_buffer(total * sizeof(float),
                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
            queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
            tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
        }
    }

    // --- Concat ---

    void dispatch_concat(const onnx::NodeProto& node) {
        int64_t axis = 0;
        for (auto& attr : node.attributes)
            if (attr.name == "axis") axis = attr.i;

        auto& first = get_tensor(node.inputs[0]);
        size_t ndim = first.shape.size();
        if (axis < 0) axis += static_cast<int64_t>(ndim);

        std::vector<std::vector<float>> all_data;
        std::vector<const GpuTensor*> tensors;
        uint32_t concat_dim = 0;
        for (auto& inp : node.inputs) {
            auto& t = get_tensor(inp);
            tensors.push_back(&t);
            concat_dim += t.shape[static_cast<size_t>(axis)];
            all_data.push_back(readback_raw(inp));
        }

        std::vector<uint32_t> out_shape = first.shape;
        out_shape[static_cast<size_t>(axis)] = concat_dim;

        uint32_t total = 1;
        for (auto d : out_shape) total *= d;

        uint32_t outer = 1, inner = 1;
        for (int64_t i = 0; i < axis; i++) outer *= out_shape[i];
        for (size_t i = static_cast<size_t>(axis) + 1; i < ndim; i++) inner *= out_shape[i];

        std::vector<float> out_data(total);
        uint32_t offset = 0;
        for (uint32_t o = 0; o < outer; o++) {
            for (size_t t = 0; t < tensors.size(); t++) {
                uint32_t t_axis = tensors[t]->shape[static_cast<size_t>(axis)];
                uint32_t chunk = t_axis * inner;
                uint32_t t_row_stride = t_axis * inner;
                std::memcpy(&out_data[offset], &all_data[t][o * t_row_stride], chunk * sizeof(float));
                offset += chunk;
            }
        }

        auto out_buf = create_buffer(total * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
    }

    // --- Split ---

    void dispatch_split(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        int64_t axis = 0;
        for (auto& attr : node.attributes)
            if (attr.name == "axis") axis = attr.i;
        size_t ndim = X.shape.size();
        if (axis < 0) axis += static_cast<int64_t>(ndim);

        std::vector<int64_t> split_sizes;
        if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
            split_sizes = readback_raw_int64(node.inputs[1]);
        } else {
            for (auto& attr : node.attributes)
                if (attr.name == "split") split_sizes = attr.ints;
        }
        if (split_sizes.empty()) {
            uint32_t n_out = static_cast<uint32_t>(node.outputs.size());
            uint32_t dim = X.shape[static_cast<size_t>(axis)];
            for (uint32_t i = 0; i < n_out; i++)
                split_sizes.push_back(dim / n_out);
        }

        auto data = readback_raw(node.inputs[0]);
        uint32_t outer = 1, inner = 1;
        for (int64_t i = 0; i < axis; i++) outer *= X.shape[i];
        for (size_t i = static_cast<size_t>(axis) + 1; i < ndim; i++) inner *= X.shape[i];

        uint32_t axis_offset = 0;
        uint32_t full_axis = X.shape[static_cast<size_t>(axis)];
        for (size_t s = 0; s < split_sizes.size() && s < node.outputs.size(); s++) {
            uint32_t sz = static_cast<uint32_t>(split_sizes[s]);
            auto out_shape = X.shape;
            out_shape[static_cast<size_t>(axis)] = sz;
            uint32_t out_count = 1;
            for (auto d : out_shape) out_count *= d;

            std::vector<float> chunk(out_count);
            for (uint32_t o = 0; o < outer; o++) {
                std::memcpy(&chunk[o * sz * inner],
                           &data[o * full_axis * inner + axis_offset * inner],
                           sz * inner * sizeof(float));
            }
            auto out_buf = create_buffer(out_count * sizeof(float),
                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
            queue_.WriteBuffer(out_buf, 0, chunk.data(), chunk.size() * sizeof(float));
            tensors_[node.outputs[s]] = GpuTensor{out_buf, out_shape};
            axis_offset += sz;
        }
    }

    // --- Conv (1D/2D, CPU fallback for correctness) ---

    void dispatch_conv(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        auto& W = get_tensor(node.inputs[1]);

        auto x_data = readback_raw(node.inputs[0]);
        auto w_data = readback_raw(node.inputs[1]);

        std::vector<float> bias_data;
        if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
            bias_data = readback_raw(node.inputs[2]);
        }

        int64_t group = 1;
        std::vector<int64_t> pads, strides, dilations, kernel_shape;
        for (auto& attr : node.attributes) {
            if (attr.name == "group") group = attr.i;
            else if (attr.name == "pads") pads = attr.ints;
            else if (attr.name == "strides") strides = attr.ints;
            else if (attr.name == "dilations") dilations = attr.ints;
            else if (attr.name == "kernel_shape") kernel_shape = attr.ints;
        }

        uint32_t N = X.shape[0], C = X.shape[1];
        uint32_t OC = W.shape[0], IC_per_g = W.shape[1];
        bool is_1d = (X.shape.size() == 3);

        if (is_1d) {
            uint32_t L = X.shape[2];
            uint32_t kL = W.shape[2];
            int64_t pad_l = pads.size() >= 1 ? pads[0] : 0;
            int64_t pad_r = pads.size() >= 2 ? pads[1] : 0;
            int64_t stride = strides.size() >= 1 ? strides[0] : 1;
            int64_t dilation = dilations.size() >= 1 ? dilations[0] : 1;
            uint32_t outL = static_cast<uint32_t>((static_cast<int64_t>(L) + pad_l + pad_r -
                dilation * (static_cast<int64_t>(kL) - 1) - 1) / stride + 1);

            std::vector<float> out_data(N * OC * outL, 0.0f);
            uint32_t oc_per_g = OC / static_cast<uint32_t>(group);
            for (uint32_t n = 0; n < N; n++) {
                for (int64_t g = 0; g < group; g++) {
                    for (uint32_t oc = 0; oc < oc_per_g; oc++) {
                        uint32_t oc_abs = static_cast<uint32_t>(g) * oc_per_g + oc;
                        for (uint32_t ol = 0; ol < outL; ol++) {
                            float val = bias_data.empty() ? 0.0f : bias_data[oc_abs];
                            for (uint32_t ic = 0; ic < IC_per_g; ic++) {
                                uint32_t ic_abs = static_cast<uint32_t>(g) * IC_per_g + ic;
                                for (uint32_t kl = 0; kl < kL; kl++) {
                                    int64_t il = static_cast<int64_t>(ol) * stride - pad_l +
                                                 static_cast<int64_t>(kl) * dilation;
                                    if (il >= 0 && il < static_cast<int64_t>(L)) {
                                        val += x_data[n * C * L + ic_abs * L + static_cast<uint32_t>(il)] *
                                               w_data[oc_abs * IC_per_g * kL + ic * kL + kl];
                                    }
                                }
                            }
                            out_data[n * OC * outL + oc_abs * outL + ol] = val;
                        }
                    }
                }
            }
            auto out_buf = create_buffer(out_data.size() * sizeof(float),
                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
            queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
            tensors_[node.outputs[0]] = GpuTensor{out_buf, {N, OC, outL}};
        } else {
            uint32_t H = X.shape[2], W_in = X.shape[3];
            uint32_t kH = W.shape[2], kW = W.shape[3];
            int64_t pad_t = pads.size() >= 1 ? pads[0] : 0;
            int64_t pad_l = pads.size() >= 2 ? pads[1] : 0;
            int64_t pad_b = pads.size() >= 3 ? pads[2] : 0;
            int64_t pad_r = pads.size() >= 4 ? pads[3] : 0;
            int64_t stride_h = strides.size() >= 1 ? strides[0] : 1;
            int64_t stride_w = strides.size() >= 2 ? strides[1] : 1;
            int64_t dil_h = dilations.size() >= 1 ? dilations[0] : 1;
            int64_t dil_w = dilations.size() >= 2 ? dilations[1] : 1;
            uint32_t outH = static_cast<uint32_t>((static_cast<int64_t>(H) + pad_t + pad_b -
                dil_h * (static_cast<int64_t>(kH) - 1) - 1) / stride_h + 1);
            uint32_t outW = static_cast<uint32_t>((static_cast<int64_t>(W_in) + pad_l + pad_r -
                dil_w * (static_cast<int64_t>(kW) - 1) - 1) / stride_w + 1);

            std::vector<float> out_data(N * OC * outH * outW, 0.0f);
            uint32_t oc_per_g = OC / static_cast<uint32_t>(group);
            for (uint32_t n = 0; n < N; n++) {
                for (int64_t g = 0; g < group; g++) {
                    for (uint32_t oc = 0; oc < oc_per_g; oc++) {
                        uint32_t oc_abs = static_cast<uint32_t>(g) * oc_per_g + oc;
                        for (uint32_t oh = 0; oh < outH; oh++) {
                            for (uint32_t ow = 0; ow < outW; ow++) {
                                float val = bias_data.empty() ? 0.0f : bias_data[oc_abs];
                                for (uint32_t ic = 0; ic < IC_per_g; ic++) {
                                    uint32_t ic_abs = static_cast<uint32_t>(g) * IC_per_g + ic;
                                    for (uint32_t kh = 0; kh < kH; kh++) {
                                        for (uint32_t kw = 0; kw < kW; kw++) {
                                            int64_t ih = static_cast<int64_t>(oh) * stride_h - pad_t +
                                                         static_cast<int64_t>(kh) * dil_h;
                                            int64_t iw = static_cast<int64_t>(ow) * stride_w - pad_l +
                                                         static_cast<int64_t>(kw) * dil_w;
                                            if (ih >= 0 && ih < static_cast<int64_t>(H) &&
                                                iw >= 0 && iw < static_cast<int64_t>(W_in)) {
                                                val += x_data[n * C * H * W_in + ic_abs * H * W_in +
                                                       static_cast<uint32_t>(ih) * W_in + static_cast<uint32_t>(iw)] *
                                                       w_data[oc_abs * IC_per_g * kH * kW + ic * kH * kW + kh * kW + kw];
                                            }
                                        }
                                    }
                                }
                                out_data[n * OC * outH * outW + oc_abs * outH * outW + oh * outW + ow] = val;
                            }
                        }
                    }
                }
            }
            auto out_buf = create_buffer(out_data.size() * sizeof(float),
                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
            queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
            tensors_[node.outputs[0]] = GpuTensor{out_buf, {N, OC, outH, outW}};
        }
    }

    // --- RotaryEmbedding ---

    void dispatch_rotary_embedding(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t count = X.element_count();

        uint32_t head_dim = X.shape.back();
        uint32_t num_heads = X.shape.size() >= 2 ? X.shape[X.shape.size() - 2] : 1;
        uint32_t M = X.shape.size() >= 3 ? X.shape[X.shape.size() - 3] : 1;
        float theta_base = 10000.0f;
        uint32_t seq_pos_offset = 0;

        for (auto& attr : node.attributes) {
            if (attr.name == "theta") theta_base = attr.f;
            else if (attr.name == "position_offset") seq_pos_offset = static_cast<uint32_t>(attr.i);
        }

        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);

        uint32_t theta_bits;
        std::memcpy(&theta_bits, &theta_base, sizeof(uint32_t));
        uint32_t params[8] = {head_dim, num_heads, theta_bits, 0, seq_pos_offset, M, 0, 0};
        auto param_buf = create_uniform(params, sizeof(params));

        auto pipeline = cache_.get(shader_dir_ + "/rope.wgsl");

        std::vector<wgpu::BindGroupEntry> entries(3);
        wgpu::Buffer bufs[] = {X.buffer, out_buf, param_buf};
        for (int i = 0; i < 3; i++) {
            entries[i].binding = i;
            entries[i].buffer = bufs[i];
            entries[i].offset = 0;
            entries[i].size = bufs[i].GetSize();
        }

        wgpu::BindGroupDescriptor bg{};
        bg.layout = pipeline.GetBindGroupLayout(0);
        bg.entryCount = 3;
        bg.entries = entries.data();
        auto bind_group = device_.CreateBindGroup(&bg);

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups((count + 63) / 64);
        pass.End();
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        tensors_[node.outputs[0]] = GpuTensor{out_buf, X.shape};
    }

    // --- ReduceMean ---

    void dispatch_reducemean(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        std::vector<int64_t> axes;
        int64_t keepdims = 1;
        for (auto& attr : node.attributes) {
            if (attr.name == "axes") axes = attr.ints;
            else if (attr.name == "keepdims") keepdims = attr.i;
        }
        if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
            axes = readback_raw_int64(node.inputs[1]);
        }

        auto data = readback_raw(node.inputs[0]);
        size_t ndim = X.shape.size();
        for (auto& a : axes) if (a < 0) a += static_cast<int64_t>(ndim);

        std::vector<uint32_t> out_shape;
        for (size_t d = 0; d < ndim; d++) {
            bool reduced = false;
            for (auto a : axes) if (static_cast<size_t>(a) == d) reduced = true;
            if (reduced) {
                if (keepdims) out_shape.push_back(1);
            } else {
                out_shape.push_back(X.shape[d]);
            }
        }
        if (out_shape.empty()) out_shape.push_back(1);

        uint32_t out_count = 1;
        for (auto d : out_shape) out_count *= d;

        std::vector<uint32_t> strides(ndim, 1);
        for (int i = static_cast<int>(ndim) - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * X.shape[i + 1];

        std::vector<float> out_data(out_count, 0.0f);
        std::vector<uint32_t> counts(out_count, 0);
        uint32_t total = X.element_count();

        for (uint32_t idx = 0; idx < total; idx++) {
            uint32_t rem = idx;
            uint32_t out_idx = 0;
            uint32_t out_stride = 1;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                uint32_t coord = rem / strides[d];
                rem %= strides[d];
                bool reduced = false;
                for (auto a : axes) if (a == d) reduced = true;
                if (!reduced) {
                    out_idx += coord * out_stride;
                    out_stride *= (d > 0 || out_shape.size() > 0) ? out_shape[d < static_cast<int>(out_shape.size()) ? d : 0] : 1;
                }
            }
            // Simpler: recompute out_idx from non-reduced coords
            uint32_t oi = 0;
            uint32_t os = 1;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                uint32_t coord = (idx / strides[d]) % X.shape[d];
                bool reduced = false;
                for (auto a : axes) if (a == d) reduced = true;
                if (!reduced) {
                    oi += coord * os;
                    os *= X.shape[d];
                }
            }
            out_data[oi] += data[idx];
            counts[oi]++;
        }
        for (uint32_t i = 0; i < out_count; i++)
            if (counts[i] > 0) out_data[i] /= static_cast<float>(counts[i]);

        auto out_buf = create_buffer(out_count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
    }

    // --- Where ---

    void dispatch_where(const onnx::NodeProto& node) {
        auto& cond = get_tensor(node.inputs[0]);
        auto& X = get_tensor(node.inputs[1]);
        auto& Y = get_tensor(node.inputs[2]);
        uint32_t count = (std::max)({cond.element_count(), X.element_count(), Y.element_count()});
        auto c_data = readback_raw(node.inputs[0]);
        auto x_data = readback_raw(node.inputs[1]);
        auto y_data = readback_raw(node.inputs[2]);
        std::vector<float> out_data(count);
        for (uint32_t i = 0; i < count; i++) {
            float c = c_data[i % c_data.size()];
            out_data[i] = (c != 0.0f) ? x_data[i % x_data.size()] : y_data[i % y_data.size()];
        }
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        auto& largest = (cond.element_count() >= X.element_count() && cond.element_count() >= Y.element_count())
            ? cond : (X.element_count() >= Y.element_count() ? X : Y);
        tensors_[node.outputs[0]] = GpuTensor{out_buf, largest.shape};
    }

    // --- Equal ---

    void dispatch_equal(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);
        uint32_t count = A.element_count();
        auto a_data = readback_raw(node.inputs[0]);
        auto b_data = readback_raw(node.inputs[1]);
        std::vector<float> out_data(count);
        for (uint32_t i = 0; i < count; i++)
            out_data[i] = (a_data[i] == b_data[i % b_data.size()]) ? 1.0f : 0.0f;
        auto out_buf = create_buffer(count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, A.shape};
    }

    // --- Gemm ---

    void dispatch_gemm(const onnx::NodeProto& node) {
        auto& A = get_tensor(node.inputs[0]);
        auto& B = get_tensor(node.inputs[1]);
        float alpha = 1.0f, beta = 1.0f;
        int64_t transA = 0, transB = 0;
        for (auto& attr : node.attributes) {
            if (attr.name == "alpha") alpha = attr.f;
            else if (attr.name == "beta") beta = attr.f;
            else if (attr.name == "transA") transA = attr.i;
            else if (attr.name == "transB") transB = attr.i;
        }
        uint32_t M = transA ? A.shape[1] : A.shape[0];
        uint32_t K = transA ? A.shape[0] : A.shape[1];
        uint32_t N = transB ? B.shape[0] : B.shape[1];

        auto a_data = readback_raw(node.inputs[0]);
        auto b_data = readback_raw(node.inputs[1]);
        std::vector<float> c_data;
        if (node.inputs.size() > 2 && !node.inputs[2].empty())
            c_data = readback_raw(node.inputs[2]);

        std::vector<float> out_data(M * N, 0.0f);
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    float a = transA ? a_data[k * M + m] : a_data[m * K + k];
                    float b = transB ? b_data[n * K + k] : b_data[k * N + n];
                    sum += a * b;
                }
                float bias = 0.0f;
                if (!c_data.empty()) {
                    bias = (c_data.size() == N) ? c_data[n] : c_data[m * N + n];
                }
                out_data[m * N + n] = alpha * sum + beta * bias;
            }
        }
        auto out_buf = create_buffer(M * N * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, {M, N}};
    }

    // --- Shape (outputs shape as int64 tensor stored as float) ---

    void dispatch_shape(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        uint32_t ndim = static_cast<uint32_t>(X.shape.size());
        std::vector<float> shape_as_float(ndim);
        for (uint32_t i = 0; i < ndim; i++)
            shape_as_float[i] = static_cast<float>(X.shape[i]);
        auto out_buf = create_buffer(ndim * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, shape_as_float.data(), shape_as_float.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, {ndim}};
    }

    // --- Slice ---

    void dispatch_slice(const onnx::NodeProto& node) {
        auto& X = get_tensor(node.inputs[0]);
        auto data = readback_raw(node.inputs[0]);
        size_t ndim = X.shape.size();

        auto starts_v = readback_raw_int64(node.inputs[1]);
        auto ends_v = readback_raw_int64(node.inputs[2]);
        std::vector<int64_t> axes_v, steps_v;
        if (node.inputs.size() > 3 && !node.inputs[3].empty())
            axes_v = readback_raw_int64(node.inputs[3]);
        if (node.inputs.size() > 4 && !node.inputs[4].empty())
            steps_v = readback_raw_int64(node.inputs[4]);

        std::vector<int64_t> starts(ndim, 0), ends(ndim), steps(ndim, 1);
        for (size_t i = 0; i < ndim; i++) ends[i] = static_cast<int64_t>(X.shape[i]);

        for (size_t i = 0; i < starts_v.size(); i++) {
            int64_t ax = axes_v.empty() ? static_cast<int64_t>(i) : axes_v[i];
            if (ax < 0) ax += static_cast<int64_t>(ndim);
            int64_t dim = static_cast<int64_t>(X.shape[ax]);
            int64_t s = starts_v[i];
            int64_t e = ends_v[i];
            int64_t st = steps_v.empty() ? 1 : steps_v[i];
            if (s < 0) s += dim;
            if (e < 0) e += dim;
            s = (std::max)(int64_t(0), (std::min)(s, dim));
            e = (std::max)(int64_t(0), (std::min)(e, dim));
            starts[ax] = s;
            ends[ax] = e;
            steps[ax] = st;
        }

        std::vector<uint32_t> out_shape(ndim);
        for (size_t i = 0; i < ndim; i++) {
            int64_t span = ends[i] - starts[i];
            out_shape[i] = static_cast<uint32_t>((span + steps[i] - (steps[i] > 0 ? 1 : -1)) / steps[i]);
            if (out_shape[i] == 0) out_shape[i] = 0;
        }

        uint32_t out_count = 1;
        for (auto d : out_shape) out_count *= d;

        std::vector<uint32_t> in_strides(ndim, 1);
        for (int i = static_cast<int>(ndim) - 2; i >= 0; i--)
            in_strides[i] = in_strides[i + 1] * X.shape[i + 1];
        std::vector<uint32_t> out_strides(ndim, 1);
        for (int i = static_cast<int>(ndim) - 2; i >= 0; i--)
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];

        std::vector<float> out_data(out_count);
        for (uint32_t idx = 0; idx < out_count; idx++) {
            uint32_t in_idx = 0;
            uint32_t rem = idx;
            for (size_t d = 0; d < ndim; d++) {
                uint32_t coord = rem / out_strides[d];
                rem %= out_strides[d];
                in_idx += static_cast<uint32_t>(starts[d] + static_cast<int64_t>(coord) * steps[d]) * in_strides[d];
            }
            out_data[idx] = data[in_idx];
        }

        auto out_buf = create_buffer(out_count * sizeof(float),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        queue_.WriteBuffer(out_buf, 0, out_data.data(), out_data.size() * sizeof(float));
        tensors_[node.outputs[0]] = GpuTensor{out_buf, out_shape};
    }

    // --- Readback helpers ---

    std::vector<float> readback_raw(const std::string& name) {
        return readback(name);
    }

    std::vector<int64_t> readback_raw_int64(const std::string& name) {
        auto& t = get_tensor(name);
        uint64_t byte_size = t.buffer.GetSize();
        uint32_t count = t.element_count();

        wgpu::BufferDescriptor desc{};
        desc.size = byte_size;
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        auto staging = device_.CreateBuffer(&desc);

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(t.buffer, 0, staging, 0, byte_size);
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        bool done = false;
        staging.MapAsync(wgpu::MapMode::Read, 0, byte_size,
            wgpu::CallbackMode::AllowProcessEvents,
            [&done](wgpu::MapAsyncStatus status, wgpu::StringView) {
                if (status == wgpu::MapAsyncStatus::Success) done = true;
            });
        while (!done) device_.Tick();

        const uint8_t* raw = static_cast<const uint8_t*>(staging.GetConstMappedRange(0, byte_size));
        std::vector<int64_t> result;

        if (byte_size == count * sizeof(int64_t)) {
            result.resize(count);
            std::memcpy(result.data(), raw, byte_size);
        } else {
            const float* fp = reinterpret_cast<const float*>(raw);
            result.resize(count);
            for (uint32_t i = 0; i < count; i++)
                result[i] = static_cast<int64_t>(fp[i]);
        }

        staging.Unmap();
        return result;
    }

    std::vector<float> readback(const std::string& name) {
        auto& t = get_tensor(name);
        uint64_t size = t.element_count() * sizeof(float);

        wgpu::BufferDescriptor desc{};
        desc.size = size;
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        auto staging = device_.CreateBuffer(&desc);

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(t.buffer, 0, staging, 0, size);
        auto cmd = encoder.Finish();
        queue_.Submit(1, &cmd);

        bool done = false;
        staging.MapAsync(wgpu::MapMode::Read, 0, size,
            wgpu::CallbackMode::AllowProcessEvents,
            [&done](wgpu::MapAsyncStatus status, wgpu::StringView) {
                if (status == wgpu::MapAsyncStatus::Success) done = true;
            });

        while (!done) {
            device_.Tick();
        }

        const float* mapped = static_cast<const float*>(staging.GetConstMappedRange(0, size));
        std::vector<float> result(mapped, mapped + t.element_count());
        staging.Unmap();
        return result;
    }
};

} // namespace onnx_runtime
