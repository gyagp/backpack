enable f16;
enable subgroups;

// Multi-query causal attention with fp16 KV + early exit.
// 4 query positions per WG. Uniform params for data-dependent loop bound.
// Used on D3D12 (no MMA) and as Vulkan fallback.
//
// Grid: (n_head, ceil(T/4), 1)
// WG: 128 threads (4 warps × 32 threads)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

struct AttnParams {
    kv_stride: u32,
    n_rep: u32,
    T_total: u32,
    cache_offset: u32,
    T_prefill: u32,
    scale_bits: u32,
    neg_inf_bits: u32,
    pad1: u32,
};
@group(0) @binding(4) var<uniform> params: AttnParams;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;
const QUERIES_PER_WG: u32 = 4u;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let q_block = wid.y;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let kv_stride = params.kv_stride;
    let n_rep = params.n_rep;
    let T_total = params.T_total;
    let cache_offset = params.cache_offset;
    let scale = bitcast<f32>(params.scale_bits);
    let neg_inf = bitcast<f32>(params.neg_inf_bits);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let n_head_total = kv_stride / HD * n_rep;

    let q_idx = q_block * QUERIES_PER_WG + warp_id;
    let q_abs_pos = cache_offset + q_idx;
    let q_valid = q_idx < params.T_prefill;

    let q_base = q_idx * n_head_total * HD + head * HD;
    var q: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        q[e] = 0.0;
        if (q_valid && d < HD) {
            q[e] = Q[q_base + d];
        }
    }

    var acc: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        acc[e] = 0.0;
    }
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    // Early exit: uniform loop bound from var<uniform> params
    let max_causal = min(cache_offset + q_block * QUERIES_PER_WG + QUERIES_PER_WG, T_total);

    for (var t = 0u; t < max_causal; t = t + 1u) {
        let causal_valid = q_valid && t <= q_abs_pos;

        let k_base = t * kv_stride + kv_off;
        let k_off = k_base + lane * HD_PER_THREAD;

        var partial = 0.0;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var k = 0.0;
            if (causal_valid && d < HD) {
                k = f32(K_cache[k_off + e]);
            }
            partial += q[e] * k;
        }
        let dot_qk = subgroupAdd(partial);
        let score = select(neg_inf, dot_qk * scale, causal_valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_off = k_base + lane * HD_PER_THREAD;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var v = 0.0;
            if (causal_valid && d < HD) {
                v = f32(V_cache[v_off + e]);
            }
            acc[e] = acc[e] * rescale + v * w;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    let out_base = q_idx * n_head_total * HD + head * HD;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        if (q_valid && d < HD) {
            Out[out_base + d] = acc[e];
        }
    }
}
