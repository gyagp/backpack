"""WGSL GatedDeltaNet recurrence kernel for Qwen3.5 SSM layers.

Single-dispatch kernel that performs the full SSM recurrence for T=1 decode:
  1. Conv1d with SiLU activation
  2. Split Q/K/V from conv output
  3. L2 normalize Q and K
  4. Compute decay (g) and gate (beta) from a,b projections
  5. GatedDeltaNet state recurrence (parallel across 48 heads)
  6. Gated RMSNorm with Z gate
  7. Write output

Grid: (48,) — one workgroup per value head.
Workgroup size: 128 threads — each thread handles one element of head_dim.

Buffers:
  @binding(0) QKV: array<f32>     — conv1d input (10240,) = key_dim*2 + val_dim
  @binding(1) Z: array<f32>       — Z projection output (6144,) = n_v * hv
  @binding(2) A_proj: array<f32>  — a projection output (48,)
  @binding(3) B_proj: array<f32>  — b projection output (48,)
  @binding(4) ConvState: array<f32> — conv1d state (3 * 10240)
  @binding(5) ConvWeight: array<f32> — conv1d weight (10240 * 4)
  @binding(6) SSMState: array<f32> — SSM state (48 * 128 * 128)
  @binding(7) A_log: array<f32>   — decay log (48,)
  @binding(8) DT_bias: array<f32> — dt bias (48,)
  @binding(9) NormWeight: array<f32> — norm weight (48 * 128)
  @binding(10) Output: array<f32> — output (6144,) = n_v * hv
  @binding(11) _params_: array<u32> — parameters
"""

import struct
import numpy as np
from triton.backends.webgpu.dawn_runner import BufferBinding

# Dimensions (hardcoded for Qwen3.5)
GDN_N_V_HEADS = 48
GDN_N_K_HEADS = 16
GDN_HEAD_K_DIM = 128
GDN_HEAD_V_DIM = 128
GDN_V_PER_K = 3
GDN_KEY_DIM = GDN_N_K_HEADS * GDN_HEAD_K_DIM   # 2048
GDN_VAL_DIM = GDN_N_V_HEADS * GDN_HEAD_V_DIM   # 6144
GDN_CONV_DIM = GDN_KEY_DIM * 2 + GDN_VAL_DIM    # 10240
GDN_CONV_KERNEL = 4

GDN_BINDINGS = [
    BufferBinding(name='QKV', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='Z', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='A_proj', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='B_proj', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='ConvState', binding=4, access='read_write', elem_type='f32'),
    BufferBinding(name='ConvWeight', binding=5, access='read_write', elem_type='f32'),
    BufferBinding(name='SSMState', binding=6, access='read_write', elem_type='f32'),
    BufferBinding(name='A_log', binding=7, access='read_write', elem_type='f32'),
    BufferBinding(name='DT_bias', binding=8, access='read_write', elem_type='f32'),
    BufferBinding(name='NormWeight', binding=9, access='read_write', elem_type='f32'),
    BufferBinding(name='Output', binding=10, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=11, access='read_write', elem_type='u32'),
]

GDN_WORKGROUP_SIZE = 128  # = head_k_dim = head_v_dim

def pack_gdn_params(rms_eps_bits: int):
    """Pack params: just the RMS epsilon as f32 bits."""
    return np.array([rms_eps_bits], dtype=np.uint32)


WGSL_GDN_KERNEL = """
// GatedDeltaNet recurrence kernel for Qwen3.5 SSM decode (T=1)
// Grid: (48,) — one workgroup per value head
// 128 threads per workgroup = one thread per head dimension element

// Hardcoded dimensions (Qwen3.5)
const N_V: u32 = 48u;
const N_K: u32 = 16u;
const HK: u32 = 128u;
const HV: u32 = 128u;
const V_PER_K: u32 = 3u;
const KEY_DIM: u32 = 2048u;
const VAL_DIM: u32 = 6144u;
const CONV_DIM: u32 = 10240u;
const CONV_KERNEL: u32 = 4u;
const CONV_HIST: u32 = 3u;  // CONV_KERNEL - 1

@group(0) @binding(0) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> Z: array<f32>;
@group(0) @binding(2) var<storage, read_write> A_proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> B_proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> ConvState: array<f32>;
@group(0) @binding(5) var<storage, read_write> ConvWeight: array<f32>;
@group(0) @binding(6) var<storage, read_write> SSMState: array<f32>;
@group(0) @binding(7) var<storage, read_write> A_log_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> DT_bias_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> NormWeight: array<f32>;
@group(0) @binding(10) var<storage, read_write> Output: array<f32>;
@group(0) @binding(11) var<storage, read_write> _params_: array<u32>;

var<workgroup> shared_scalar: f32;
var<workgroup> shared_vec: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let v_head: u32 = wg_id.x;    // 0..47 (value head index)
    let tid: u32 = lid.x;          // 0..127 (element within head)
    let k_head: u32 = v_head / V_PER_K;  // 0..15 (key head index)
    let v_in_k: u32 = v_head % V_PER_K;  // 0..2 (which v head within k group)

    // ---- 1. Conv1d for this head's Q/K/V channels ----
    // Each v_head covers:
    //   If this is a K-group leader (v_in_k == 0): process Q[k_head] and K[k_head]
    //   Always: process V[v_head]
    // Conv1d: output[ch] = sum(state[t] * w[ch,t]) + input[ch] * w[ch,3]

    // Q channel for this head (only k-group leader processes)
    let q_ch_base: u32 = k_head * HK;  // offset in QKV for Q
    let k_ch_base: u32 = KEY_DIM + k_head * HK;  // offset for K
    let v_ch_base: u32 = KEY_DIM * 2u + v_head * HV;  // offset for V

    // Conv1d for V channel (each head does its own V)
    let v_ch: u32 = v_ch_base + tid;
    var conv_out_v: f32 = 0.0;
    for (var t: u32 = 0u; t < CONV_HIST; t++) {
        let state_val = ConvState[t * CONV_DIM + v_ch];
        let w_val = ConvWeight[v_ch * CONV_KERNEL + t];
        conv_out_v += state_val * w_val;
    }
    let qkv_v = QKV[v_ch];
    conv_out_v += qkv_v * ConvWeight[v_ch * CONV_KERNEL + CONV_HIST];

    // Update conv state: shift history, add new input
    for (var t: u32 = 0u; t < CONV_HIST - 1u; t++) {
        ConvState[t * CONV_DIM + v_ch] = ConvState[(t + 1u) * CONV_DIM + v_ch];
    }
    ConvState[(CONV_HIST - 1u) * CONV_DIM + v_ch] = qkv_v;

    // SiLU on V conv output
    let v_val: f32 = conv_out_v / (1.0 + exp(-conv_out_v));

    // Conv1d for Q and K channels (shared across v_per_k heads)
    let q_ch: u32 = q_ch_base + tid;
    var conv_out_q: f32 = 0.0;
    for (var t: u32 = 0u; t < CONV_HIST; t++) {
        conv_out_q += ConvState[t * CONV_DIM + q_ch] * ConvWeight[q_ch * CONV_KERNEL + t];
    }
    conv_out_q += QKV[q_ch] * ConvWeight[q_ch * CONV_KERNEL + CONV_HIST];
    // Update conv state for Q
    if (v_in_k == 0u) {
        for (var t: u32 = 0u; t < CONV_HIST - 1u; t++) {
            ConvState[t * CONV_DIM + q_ch] = ConvState[(t + 1u) * CONV_DIM + q_ch];
        }
        ConvState[(CONV_HIST - 1u) * CONV_DIM + q_ch] = QKV[q_ch];
    }
    let q_silu: f32 = conv_out_q / (1.0 + exp(-conv_out_q));

    let k_ch: u32 = k_ch_base + tid;
    var conv_out_k: f32 = 0.0;
    for (var t: u32 = 0u; t < CONV_HIST; t++) {
        conv_out_k += ConvState[t * CONV_DIM + k_ch] * ConvWeight[k_ch * CONV_KERNEL + t];
    }
    conv_out_k += QKV[k_ch] * ConvWeight[k_ch * CONV_KERNEL + CONV_HIST];
    if (v_in_k == 0u) {
        for (var t: u32 = 0u; t < CONV_HIST - 1u; t++) {
            ConvState[t * CONV_DIM + k_ch] = ConvState[(t + 1u) * CONV_DIM + k_ch];
        }
        ConvState[(CONV_HIST - 1u) * CONV_DIM + k_ch] = QKV[k_ch];
    }
    let k_silu: f32 = conv_out_k / (1.0 + exp(-conv_out_k));

    // ---- 2. L2 normalize Q and K ----
    // Squared sum via shared memory reduction
    shared_vec[tid] = q_silu * q_silu;
    workgroupBarrier();
    // Simple reduction for 128 elements
    if (tid < 64u) { shared_vec[tid] += shared_vec[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { shared_vec[tid] += shared_vec[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { shared_vec[tid] += shared_vec[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { shared_vec[tid] += shared_vec[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { shared_vec[tid] += shared_vec[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { shared_vec[tid] += shared_vec[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { shared_scalar = 1.0 / (sqrt(shared_vec[0] + shared_vec[1]) + 1e-6); }
    workgroupBarrier();
    let q_norm: f32 = q_silu * shared_scalar / sqrt(f32(HK));  // L2 norm + scale

    shared_vec[tid] = k_silu * k_silu;
    workgroupBarrier();
    if (tid < 64u) { shared_vec[tid] += shared_vec[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { shared_vec[tid] += shared_vec[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { shared_vec[tid] += shared_vec[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { shared_vec[tid] += shared_vec[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { shared_vec[tid] += shared_vec[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { shared_vec[tid] += shared_vec[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { shared_scalar = 1.0 / (sqrt(shared_vec[0] + shared_vec[1]) + 1e-6); }
    workgroupBarrier();
    let k_norm: f32 = k_silu * shared_scalar;

    // ---- 3. Decay and gate ----
    // g = -exp(A_log) * softplus(a + dt_bias)
    // Only need one scalar per head — compute on thread 0, broadcast
    if (tid == 0u) {
        let A_log_val = A_log_buf[v_head];
        let dt_bias_val = DT_bias_buf[v_head];
        let a_val = A_proj[v_head];
        let b_val = B_proj[v_head];
        let g = -exp(A_log_val) * log(1.0 + exp(a_val + dt_bias_val));
        // Store decay and beta in shared memory
        shared_vec[0] = exp(g);   // decay factor
        shared_vec[1] = 1.0 / (1.0 + exp(-b_val));  // sigmoid(b) = beta
    }
    workgroupBarrier();
    let decay: f32 = shared_vec[0];
    let beta: f32 = shared_vec[1];

    // ---- 4. GatedDeltaNet recurrence ----
    // State: (HK, HV) per head = (128, 128)
    // state[h, i, j] where h=v_head, i=key_dim_idx, j=value_dim_idx
    // Thread tid handles column j (value dimension)
    let state_base: u32 = v_head * HK * HV;

    // Step 4a: Decay state
    for (var i: u32 = 0u; i < HK; i++) {
        let idx = state_base + i * HV + tid;
        SSMState[idx] = SSMState[idx] * decay;
    }

    // Step 4b: kv_mem = sum_i(state[i, tid] * k[i]) for this column tid
    var kv_mem: f32 = 0.0;
    for (var i: u32 = 0u; i < HK; i++) {
        // k_norm is stored per thread — need to broadcast k[i]
        // k[i] lives in thread i's register. Use shared memory.
        shared_vec[tid] = k_norm;
        workgroupBarrier();
        let k_i = shared_vec[i];
        kv_mem += SSMState[state_base + i * HV + tid] * k_i;
    }

    // Step 4c: delta = (v - kv_mem) * beta
    let delta: f32 = (v_val - kv_mem) * beta;

    // Step 4d: state += outer(k, delta) = k[i] * delta[tid]
    for (var i: u32 = 0u; i < HK; i++) {
        shared_vec[tid] = k_norm;
        workgroupBarrier();
        let k_i = shared_vec[i];
        let idx = state_base + i * HV + tid;
        SSMState[idx] = SSMState[idx] + k_i * delta;
    }

    // Step 4e: output = sum_i(state[i, tid] * q[i])
    var out_val: f32 = 0.0;
    for (var i: u32 = 0u; i < HK; i++) {
        shared_vec[tid] = q_norm;
        workgroupBarrier();
        let q_i = shared_vec[i];
        out_val += SSMState[state_base + i * HV + tid] * q_i;
    }

    // ---- 5. Gated RMSNorm ----
    // norm(output) * silu(z)
    let eps_bits = _params_[0];
    let eps: f32 = bitcast<f32>(eps_bits);

    // RMS of output
    shared_vec[tid] = out_val * out_val;
    workgroupBarrier();
    if (tid < 64u) { shared_vec[tid] += shared_vec[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { shared_vec[tid] += shared_vec[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { shared_vec[tid] += shared_vec[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { shared_vec[tid] += shared_vec[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { shared_vec[tid] += shared_vec[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { shared_vec[tid] += shared_vec[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) {
        shared_scalar = 1.0 / sqrt((shared_vec[0] + shared_vec[1]) / f32(HV) + eps);
    }
    workgroupBarrier();
    let rms_inv = shared_scalar;

    let norm_w = NormWeight[v_head * HV + tid];
    let normed = out_val * rms_inv * norm_w;

    // Gated: multiply by SiLU(z)
    let z_val = Z[v_head * HV + tid];
    let z_silu = z_val / (1.0 + exp(-z_val));
    let gated_out = normed * z_silu;

    // Write output
    Output[v_head * HV + tid] = gated_out;
}
"""
