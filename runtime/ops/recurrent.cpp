/**
 * ops/recurrent.cpp — ONNX recurrent primitives used by Qwen 3.5.
 *
 * The definitions intentionally follow ORT WebGPU's indexing and state
 * update semantics.  Computation is performed in fp32; fp16 graph inputs are
 * converted on the GPU so recurrent accumulation is stable and portable.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cmath>
#include <cstring>
#include <string>

static void recurrentEnsureF32(OpContext& ex, GpuTensor& tensor) {
    ex.EnsureGpu(tensor);
    if (tensor.dtype != TensorDtype::Float16) return;
    const uint32_t count = static_cast<uint32_t>(tensor.ElementCount());
    GpuTensor converted = ex.AllocTensor(tensor.shape, TensorDtype::Float32);
    uint32_t params[4] = {count, 0, 0, 0};
    auto param = ex.getParamBuffer(sizeof(params));
    ex.getGpu()->writeBuffer(param, params, sizeof(params));
    auto& pipeline = ex.GetPipeline("recurrent_cast_f16_to_f32",
                                    WGSL_CAST_F16_TO_F32, 3);
    auto group = ex.MakeBindGroup(pipeline, {
        {0, tensor.buffer}, {1, converted.buffer}, {2, param}});
    ex.QueueDispatch(pipeline.pipeline, group, (count + 255) / 256, 1, 1,
                     "recurrent_cast_f16_to_f32");
    tensor = std::move(converted);
}

static const char* kCausalConvWithState = R"WGSL(
struct Params {
    batch: u32, channels: u32, length: u32, kernel: u32,
    state_length: u32, output_size: u32, use_silu: u32, _pad: u32,
};
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> past: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read_write> present: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= p.output_size) { return; }
    let position = index % p.length;
    let bc = index / p.length;
    let batch_index = bc / p.channels;
    let channel = bc % p.channels;
    let input_base = bc * p.length;
    let state_base = bc * p.state_length;
    var sum = 0.0;
    for (var j = 0u; j < p.kernel; j++) {
        let virtual_position = position + j;
        var value = 0.0;
        if (virtual_position < p.state_length) {
            value = past[state_base + virtual_position];
        } else {
            value = input[input_base + virtual_position - p.state_length];
        }
        sum += value * weight[channel * p.kernel + j];
    }
    sum += bias[channel];
    if (p.use_silu != 0u) { sum = sum / (1.0 + exp(-sum)); }
    output[index] = sum;

    // Exactly one invocation updates the state for each batch/channel.
    if (position == 0u) {
        for (var s = 0u; s < p.state_length; s++) {
            let vp = p.length + s;
            var value = 0.0;
            if (vp < p.state_length) {
                value = past[state_base + vp];
            } else {
                value = input[input_base + vp - p.state_length];
            }
            present[state_base + s] = value;
        }
    }
}
)WGSL";

static void opCausalConvWithState(OpContext& ex, const OnnxGraphNode& node,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in.size() < 4 || out.size() < 2 || !in[0] || !in[1] || !in[2] || !in[3]) {
        fprintf(stderr, "CausalConvWithState requires input, weight, bias, and state\n");
        return;
    }
    auto& input = *in[0];
    auto& weight = *in[1];
    auto& bias = *in[2];
    auto& state = *in[3];
    if (input.shape.size() != 3 || weight.shape.size() != 3 || state.shape.size() != 3) {
        fprintf(stderr, "CausalConvWithState received invalid ranks\n");
        return;
    }
    recurrentEnsureF32(ex, input);
    recurrentEnsureF32(ex, weight);
    recurrentEnsureF32(ex, bias);
    recurrentEnsureF32(ex, state);

    const uint32_t batch = static_cast<uint32_t>(input.shape[0]);
    const uint32_t channels = static_cast<uint32_t>(input.shape[1]);
    const uint32_t length = static_cast<uint32_t>(input.shape[2]);
    const uint32_t kernel = static_cast<uint32_t>(weight.shape[2]);
    const uint32_t stateLength = kernel - 1;
    *out[0] = ex.AllocTensor(input.shape, TensorDtype::Float32);
    *out[1] = ex.AllocTensor({batch, channels, stateLength}, TensorDtype::Float32);

    const std::string activation = [&]() {
        auto it = node.attrStrings.find("activation");
        return it == node.attrStrings.end() ? std::string{} : it->second;
    }();
    uint32_t params[8] = {batch, channels, length, kernel, stateLength,
                          batch * channels * length,
                          activation == "silu" || activation == "swish", 0};
    auto param = ex.getParamBuffer(sizeof(params));
    ex.getGpu()->writeBuffer(param, params, sizeof(params));
    auto& pipeline = ex.GetPipeline("causal_conv_with_state_f32",
                                    kCausalConvWithState, 7);
    auto group = ex.MakeBindGroup(pipeline, {
        {0, input.buffer}, {1, weight.buffer}, {2, bias.buffer},
        {3, state.buffer}, {4, out[0]->buffer}, {5, out[1]->buffer},
        {6, param}});
    ex.QueueDispatch(pipeline.pipeline, group,
                     (params[5] + 255) / 256, 1, 1,
                     "CausalConvWithState");
}

REGISTER_OP(CausalConvWithState, opCausalConvWithState)

// Portable scalar implementation of ORT's gated-delta LinearAttention.  A
// workgroup owns four value columns for one recurrent head and processes the
// sequence in order.  This favors exact, cross-adapter semantics first; the
// vec4/subgroup variants can be selected later after parity is established.
static const char* kLinearAttentionGatedDelta = R"WGSL(
struct Params {
    batch: u32, heads: u32, length: u32, dk: u32,
    dv: u32, q_heads: u32, k_heads: u32, dv_tiles: u32,
    scale: f32, decay_broadcast: u32, _p0: u32, _p1: u32,
};
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> initial_state: array<f32>;
@group(0) @binding(4) var<storage, read> decay: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read_write> present_state: array<f32>;
@group(0) @binding(8) var<uniform> p: Params;

const TILE: u32 = 4u;
var<workgroup> retrieved: array<f32, 512>;
var<workgroup> pre_output: array<f32, 512>;
var<workgroup> kq: array<f32, 128>;
var<workgroup> delta: array<f32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lane = lid3.x;
    let tile = wid.x % p.dv_tiles;
    let bh = wid.x / p.dv_tiles;
    let batch_index = bh / p.heads;
    let head = bh % p.heads;
    let dv0 = tile * TILE;
    let state_base = ((batch_index * p.heads + head) * p.dk + lane) * p.dv + dv0;
    var state: array<f32, 4>;
    if (lane < p.dk) {
        for (var j = 0u; j < TILE; j++) {
            if (dv0 + j < p.dv) { state[j] = initial_state[state_base + j]; }
        }
    }

    let packed_q = p.q_heads * p.dk;
    let packed_k = p.k_heads * p.dk;
    let packed_v = p.heads * p.dv;
    for (var t = 0u; t < p.length; t++) {
        let bt = batch_index * p.length + t;
        let k_head = head * p.k_heads / p.heads;
        let q_head = head * p.q_heads / p.heads;
        var kval = 0.0;
        var qval = 0.0;
        if (lane < p.dk) {
            kval = key[bt * packed_k + k_head * p.dk + lane];
            qval = query[bt * packed_q + q_head * p.dk + lane];
            var gate = 0.0;
            if (p.decay_broadcast != 0u) {
                gate = decay[bt * p.heads + head];
            } else {
                gate = decay[bt * p.heads * p.dk + head * p.dk + lane];
            }
            let factor = exp(gate);
            for (var j = 0u; j < TILE; j++) { state[j] *= factor; }
        }
        for (var j = 0u; j < TILE; j++) {
            retrieved[j * 128u + lane] = state[j] * kval;
            pre_output[j * 128u + lane] = state[j] * qval;
        }
        kq[lane] = kval * qval;
        workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                for (var j = 0u; j < TILE; j++) {
                    retrieved[j * 128u + lane] += retrieved[j * 128u + lane + stride];
                    pre_output[j * 128u + lane] += pre_output[j * 128u + lane + stride];
                }
                kq[lane] += kq[lane + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            let b = beta[bt * p.heads + head];
            let vbase = bt * packed_v + head * p.dv + dv0;
            let obase = bt * packed_v + head * p.dv + dv0;
            for (var j = 0u; j < TILE; j++) {
                if (dv0 + j < p.dv) {
                    delta[j] = b * (value[vbase + j] - retrieved[j * 128u]);
                    output[obase + j] =
                        (pre_output[j * 128u] + delta[j] * kq[0]) * p.scale;
                } else { delta[j] = 0.0; }
            }
        }
        workgroupBarrier();
        if (lane < p.dk) {
            for (var j = 0u; j < TILE; j++) { state[j] += kval * delta[j]; }
        }
        workgroupBarrier();
    }
    if (lane < p.dk) {
        for (var j = 0u; j < TILE; j++) {
            if (dv0 + j < p.dv) { present_state[state_base + j] = state[j]; }
        }
    }
}
)WGSL";

static void opLinearAttention(OpContext& ex, const OnnxGraphNode& node,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in.size() < 6 || out.size() < 2 || !in[0] || !in[1] || !in[2] ||
        !in[3] || !in[4] || !in[5]) {
        fprintf(stderr, "LinearAttention requires Q, K, V, state, decay, and beta\n");
        return;
    }
    auto rule = node.attrStrings.find("update_rule");
    if (rule == node.attrStrings.end() || rule->second != "gated_delta") {
        fprintf(stderr, "LinearAttention update rule is not yet supported: %s\n",
                rule == node.attrStrings.end() ? "missing" : rule->second.c_str());
        return;
    }
    for (size_t i = 0; i < 6; i++) recurrentEnsureF32(ex, *in[i]);
    auto& q = *in[0]; auto& k = *in[1]; auto& v = *in[2];
    auto& state = *in[3]; auto& decay = *in[4]; auto& beta = *in[5];
    if (q.shape.size() != 3 || k.shape.size() != 3 || v.shape.size() != 3 ||
        state.shape.size() != 4) {
        fprintf(stderr, "LinearAttention received invalid ranks\n");
        return;
    }
    const uint32_t batch = static_cast<uint32_t>(q.shape[0]);
    const uint32_t length = static_cast<uint32_t>(q.shape[1]);
    const uint32_t qHeads = static_cast<uint32_t>(node.GetInt("q_num_heads", 1));
    const uint32_t heads = static_cast<uint32_t>(node.GetInt("kv_num_heads", 1));
    const uint32_t dk = static_cast<uint32_t>(q.shape[2] / qHeads);
    const uint32_t kHeads = static_cast<uint32_t>(k.shape[2] / dk);
    const uint32_t dv = static_cast<uint32_t>(v.shape[2] / heads);
    const uint32_t tiles = (dv + 3) / 4;
    const float scale = node.GetFloat("scale", 0.0f) == 0.0f
        ? 1.0f / sqrtf(static_cast<float>(dk)) : node.GetFloat("scale", 0.0f);
    *out[0] = ex.AllocTensor({batch, length, heads * dv}, TensorDtype::Float32);
    *out[1] = ex.AllocTensor({batch, heads, dk, dv}, TensorDtype::Float32);
    struct Params {
        uint32_t batch, heads, length, dk, dv, qHeads, kHeads, tiles;
        float scale;
        uint32_t decayBroadcast, pad0, pad1;
    } params{batch, heads, length, dk, dv, qHeads, kHeads, tiles, scale,
             decay.shape.back() == static_cast<int64_t>(heads), 0, 0};
    auto param = ex.getParamBuffer(sizeof(params));
    ex.getGpu()->writeBuffer(param, &params, sizeof(params));
    auto& pipeline = ex.GetPipeline("linear_attention_gated_delta_f32",
                                    kLinearAttentionGatedDelta, 9);
    auto group = ex.MakeBindGroup(pipeline, {
        {0, q.buffer}, {1, k.buffer}, {2, v.buffer}, {3, state.buffer},
        {4, decay.buffer}, {5, beta.buffer}, {6, out[0]->buffer},
        {7, out[1]->buffer}, {8, param}});
    ex.QueueDispatch(pipeline.pipeline, group, batch * heads * tiles, 1, 1,
                     "LinearAttention");
}

REGISTER_OP(LinearAttention, opLinearAttention)

static const char* kLpNormalizeLastAxis = R"WGSL(
struct Params { rows: u32, width: u32, epsilon_bits: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let row = wid.x;
    let lane = lid3.x;
    if (row >= p.rows) { return; }
    let base = row * p.width;
    var sum = 0.0;
    for (var i = lane; i < p.width; i += 256u) {
        let x = input[base + i]; sum += x * x;
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    let denom = max(sqrt(partial[0]), bitcast<f32>(p.epsilon_bits));
    for (var i = lane; i < p.width; i += 256u) {
        output[base + i] = input[base + i] / denom;
    }
}
)WGSL";

static void opLpNormalization(OpContext& ex, const OnnxGraphNode& node,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in.empty() || !in[0] || out.empty() || !out[0]) return;
    auto& input = *in[0];
    recurrentEnsureF32(ex, input);
    int64_t axis = node.GetInt("axis", -1);
    if (axis < 0) axis += static_cast<int64_t>(input.shape.size());
    if (node.GetInt("p", 2) != 2 || axis != static_cast<int64_t>(input.shape.size()) - 1) {
        fprintf(stderr, "LpNormalization supports p=2 on the last axis\n");
        return;
    }
    const uint32_t width = static_cast<uint32_t>(input.shape.back());
    const uint32_t rows = static_cast<uint32_t>(input.ElementCount() / width);
    *out[0] = ex.AllocTensor(input.shape, TensorDtype::Float32);
    float epsilon = node.GetFloat("epsilon", 1.0e-12f);
    uint32_t epsilonBits; memcpy(&epsilonBits, &epsilon, 4);
    uint32_t params[4] = {rows, width, epsilonBits, 0};
    auto param = ex.getParamBuffer(sizeof(params));
    ex.getGpu()->writeBuffer(param, params, sizeof(params));
    auto& pipeline = ex.GetPipeline("lp_normalize_last_axis_f32",
                                    kLpNormalizeLastAxis, 3);
    auto group = ex.MakeBindGroup(pipeline,
        {{0, input.buffer}, {1, out[0]->buffer}, {2, param}});
    ex.QueueDispatch(pipeline.pipeline, group, rows, 1, 1, "LpNormalization");
}

REGISTER_OP(LpNormalization, opLpNormalization)
