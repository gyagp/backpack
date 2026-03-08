enable subgroups;

// Batched causal self-attention for prefill.
// Computes: out[T, n_head, HD] = softmax(Q·K^T / sqrt(HD), causal_mask) · V
//
// Q: [T, n_head, HD]  (query, T positions)
// K: [T_kv, n_kv, HD] (key cache, T_kv = cache_len + T)
// V: [T_kv, n_kv, HD] (value cache)
//
// GQA: Q heads are grouped, each group shares one KV head.
//   kv_head = q_head / n_rep
//
// Each workgroup handles one Q head for one query position.
// Grid: (n_head, T, 1)
// WG: 32 threads (1 warp)
//
// Causal mask: Q at position q_pos can attend to K at position k_pos
//   only if k_pos <= q_pos + cache_offset.
//
// Bindings:
//   0: Q — rotated queries [T × n_head × HD]
//   1: K_cache — key cache [T_total × n_kv × HD]
//   2: V_cache — value cache [T_total × n_kv × HD]
//   3: Out — attention output [T × n_head × HD]
//   4: _params_ — [kv_stride, n_rep, T_total, cache_offset, 0, scale_u32, neg_inf_u32, 0]

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;  // 32 threads × 4 = 128
const MAX_SEQ: u32 = 4096u;     // max sequence length

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let q_idx = wid.y;   // which query position within the prefill batch
    let lane = lid.x;

    let kv_stride = _params_[0];   // n_kv * HD
    let n_rep = _params_[1];       // n_head / n_kv
    let T_total = _params_[2];     // total KV length (cache_offset + T_prefill)
    let cache_offset = _params_[3]; // how many tokens already in cache
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;

    // Load Q for this position
    let n_head_total = kv_stride / HD * n_rep;  // total Q heads
    let q_base = q_idx * n_head_total * HD + head * HD;
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    // The causal boundary: this query at absolute position (cache_offset + q_idx)
    // can attend to positions [0, cache_offset + q_idx] inclusive.
    let q_abs_pos = cache_offset + q_idx;

    // Online softmax: streaming max and sum
    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var t = 0u; t < MAX_SEQ; t = t + 1u) {
        let causal_valid = (t < T_total) && (t <= q_abs_pos);

        // Only load K/V when position is valid (skip masked positions)
        let k_base = select(0u, t * kv_stride + kv_off, causal_valid);
        let k0 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD], causal_valid);
        let k1 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 1u], causal_valid);
        let k2 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 2u], causal_valid);
        let k3 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 3u], causal_valid);

        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot_qk = subgroupAdd(partial);
        let score = select(neg_inf, dot_qk * scale, causal_valid);

        // Online softmax update
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = t * kv_stride + kv_off;
        let v0 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD], causal_valid);
        let v1 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 1u], causal_valid);
        let v2 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 2u], causal_valid);
        let v3 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 3u], causal_valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output
    let out_base = q_idx * n_head_total * HD + head * HD;
    Out[out_base + lane * HD_PER_THREAD]      = acc0;
    Out[out_base + lane * HD_PER_THREAD + 1u] = acc1;
    Out[out_base + lane * HD_PER_THREAD + 2u] = acc2;
    Out[out_base + lane * HD_PER_THREAD + 3u] = acc3;
}
