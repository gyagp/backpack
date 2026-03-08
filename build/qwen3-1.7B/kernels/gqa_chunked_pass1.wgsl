
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const CHUNK: u32 = 64u;
// Each thread processes HD_PER_THREAD elements, total 32 * 4 = 128
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let chunk_id = wid.y;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let n_chunks = _params_[4];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    // Load Q into registers (4 elements per thread)
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    let t_start = chunk_id * CHUNK;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var i = 0u; i < CHUNK; i = i + 1u) {
        let t = t_start + i;
        let valid = t < T_total;

        let k_base = select(0u, t * kv_stride + kv_off, valid);
        let k0 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD], valid);
        let k1 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 1u], valid);
        let k2 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 2u], valid);
        let k3 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 3u], valid);

        // Full 128-element dot product via 4-element partial + subgroupAdd
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot = subgroupAdd(partial);
        let raw_score = dot * scale;
        let score = select(neg_inf, raw_score, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = select(0u, t * kv_stride + kv_off, valid);
        let v0 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD], valid);
        let v1 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 1u], valid);
        let v2 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 2u], valid);
        let v3 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 3u], valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    // Use max_chunks for the stride to prevent overlapping writes
    // across heads. Only write if this chunk is within n_chunks.
    let partial_stride = HD + 2u;
    let base = head * max_chunks * partial_stride + chunk_id * partial_stride;
    if (chunk_id < n_chunks) {
        if (lane == 0u) {
            Partials[base] = m_prev;
            Partials[base + 1u] = l_prev;
        }
        Partials[base + 2u + lane * HD_PER_THREAD] = acc0;
        Partials[base + 2u + lane * HD_PER_THREAD + 1u] = acc1;
        Partials[base + 2u + lane * HD_PER_THREAD + 2u] = acc2;
        Partials[base + 2u + lane * HD_PER_THREAD + 3u] = acc3;
    }
}
