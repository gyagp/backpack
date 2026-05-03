enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);

// Fused single-dispatch GQA decode attention with fp16 KV cache.
// Replaces the 2-dispatch chunked approach (attn_p1 + attn_p2).
//
// Each workgroup handles one Q head for T=1 decode.
// Grid: (n_head, 1, 1)
// WG: 32 threads (1 warp)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let max_seq = _params_[3];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    var q: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        q[e] = Q[q_base + lane * HD_PER_THREAD + e];
    }

    var acc: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) { acc[e] = 0.0; }
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var t = 0u; t < max_seq; t = t + 1u) {
        let valid = t < T_total;
        let k_base = select(0u, t * kv_stride + kv_off, valid);
        var dot_partial: f32 = 0.0;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let k = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + e]), valid);
            dot_partial += q[e] * k;
        }
        let dot_qk = subgroupAdd(dot_partial);
        let score = select(neg_inf, dot_qk * scale, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let v = select(0.0, f32(V_cache[k_base + lane * HD_PER_THREAD + e]), valid);
            acc[e] = acc[e] * rescale + v * w;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    for (var e = 0u; e < HD_PER_THREAD; e++) {
        Out[head * HD + lane * HD_PER_THREAD + e] = acc[e];
    }
}
