enable subgroups;

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const QUERIES_PER_WG: u32 = 4u;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let pastSeq = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let max_seq = _params_[5];
    let T = _params_[6];

    let head = wid.x;
    let q_block = wid.y;
    let tid = lid.x;
    let warp_id = tid / 32u;      // which of 4 query positions
    let lane = tid % 32u;

    let q_idx = q_block * QUERIES_PER_WG + warp_id;
    // Don't early return — all threads must participate in subgroupAdd.
    // Use valid flag to mask output writes.
    let valid = q_idx < T;

    let kv_h = head / (num_heads / kv_heads);

    // For head_dim=64, each thread handles 2 dims. For head_dim=128, 4 dims.
    let hd_per_thread = (head_dim + 31u) / 32u;

    // Load Q into registers
    let q_base = select(0u, (q_idx * num_heads + head) * head_dim, valid);
    var q_reg: array<f32, 4>;
    for (var i = 0u; i < hd_per_thread; i = i + 1u) {
        let d = lane * hd_per_thread + i;
        if (d < head_dim && valid) {
            q_reg[i] = Q[q_base + d];
        }
    }

    // Causal bound: this query can attend to positions [0, pastSeq + q_idx + 1)
    let causal_bound = select(0u, pastSeq + q_idx + 1u, valid);

    // Uniform loop bound across all warps in this workgroup: max causal bound
    let max_causal = pastSeq + min(q_block * QUERIES_PER_WG + QUERIES_PER_WG, T);

    var acc: array<f32, 4>;
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;

    for (var s = 0u; s < max_causal; s = s + 1u) {
        let k_base = (kv_h * max_seq + s) * head_dim;
        let causal_valid = s < causal_bound;

        // QK dot product via subgroup reduction
        var partial: f32 = 0.0;
        for (var i = 0u; i < hd_per_thread; i = i + 1u) {
            let d = lane * hd_per_thread + i;
            if (d < head_dim && causal_valid) {
                partial += q_reg[i] * K[k_base + d];
            }
        }
        let dot_qk = subgroupAdd(partial) * scale;
        let score = select(-1e30, dot_qk, causal_valid);

        // Online softmax
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        // Accumulate V
        let v_base = (kv_h * max_seq + s) * head_dim;
        for (var i = 0u; i < hd_per_thread; i = i + 1u) {
            let d = lane * hd_per_thread + i;
            if (d < head_dim) {
                let v_val = select(0.0, V[v_base + d], causal_valid);
                acc[i] = acc[i] * rescale + w * v_val;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output (only for valid query positions)
    if (valid) {
        let out_base = (q_idx * num_heads + head) * head_dim;
        for (var i = 0u; i < hd_per_thread; i = i + 1u) {
            let d = lane * hd_per_thread + i;
            if (d < head_dim) {
                Out[out_base + d] = acc[i];
            }
        }
    }
}
