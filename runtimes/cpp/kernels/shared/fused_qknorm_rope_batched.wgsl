enable subgroups;

// Batched fused QKnorm + RoPE + KV cache scatter for prefill.
// Processes T tokens: applies QK-norm, rotary embeddings, and writes K/V to cache.
//
// QKV layout: [T × (qDim + 2*kvDim)]
//   Q: [T × qDim], K: [T × kvDim], V: [T × kvDim]
//
// Grid: ((n_head + n_kv) * T, 1, 1)
//   First n_head*T WGs handle Q heads (norm + RoPE → qRotBuf)
//   Next n_kv*T WGs handle KV heads (norm + RoPE + scatter to cache)
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, 0, eps, T]

@group(0) @binding(0) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QNormW: array<f32>;
@group(0) @binding(7) var<storage, read> KNormW: array<f32>;
@group(0) @binding(8) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat_id = wid.x;
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let eps = bitcast<f32>(_params_[6]);
    let T = _params_[7];
    let n_kv = kvDim / HD;

    let total_q_wgs = n_head * T;

    if (flat_id < total_q_wgs) {
        // ── Q head processing ────────────────────────────────────────
        let t = flat_id / n_head;
        let head = flat_id % n_head;
        let qkv_stride = qDim + 2u * kvDim;

        // Source: QKV[t, head*HD .. (head+1)*HD]
        let src_base = t * qkv_stride + head * HD;

        // Load Q values
        var q: array<f32, 128>;
        for (var i = tid; i < HD; i += 128u) {
            q[i] = QKV[src_base + i];
        }

        // QK-norm: compute rstd across HD
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            sum_sq += q[i] * q[i];
        }
        let total_sq = subgroupAdd(sum_sq);
        // Cross-warp (128 threads = 4 warps)
        var _smem: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        // Use subgroup to collect
        if (lane == 0u) { _smem[warp_id] = total_sq; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = _smem[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply norm weight + RoPE
        let pos = pos_offset + t;
        let cos_base = pos * half_dim;
        let sin_base = pos * half_dim;

        let dst_base = t * n_head * HD + head * HD;
        for (var i = tid; i < HD; i += 128u) {
            let normed = q[i] * rstd * QNormW[i];
            if (i < half_dim) {
                let j = i + half_dim;
                let nj = q[j] * rstd * QNormW[j];
                let c = Cos[cos_base + i];
                let s = Sin[sin_base + i];
                QRot[dst_base + i] = normed * c - nj * s;
                QRot[dst_base + j] = nj * c + normed * s;
            }
        }
    } else {
        // ── KV head processing ────────────────────────────────────────
        let kv_flat = flat_id - total_q_wgs;
        let t = kv_flat / n_kv;
        let kv_head = kv_flat % n_kv;
        let qkv_stride = qDim + 2u * kvDim;

        // K source: QKV[t, qDim + kv_head*HD ..]
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        // V source: QKV[t, qDim + kvDim + kv_head*HD ..]
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        // Load K
        var k: array<f32, 128>;
        for (var i = tid; i < HD; i += 128u) {
            k[i] = QKV[k_src + i];
        }

        // K-norm
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            sum_sq += k[i] * k[i];
        }
        let total_sq = subgroupAdd(sum_sq);
        var _smem2: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { _smem2[warp_id] = total_sq; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = _smem2[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply K-norm + RoPE + scatter to cache
        let pos = pos_offset + t;
        let cos_base = pos * (HD / 2u);
        let sin_base = pos * (HD / 2u);
        let kv_stride = _params_[5];   // cache_offset * n_kv * HD (not used, derive from pos)
        // KV cache layout: [T_total × n_kv × HD]
        let cache_pos = pos;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;
        let half = HD / 2u;

        for (var i = tid; i < HD; i += 128u) {
            let normed = k[i] * rstd * KNormW[i];
            if (i < half) {
                let j = i + half;
                let nj = k[j] * rstd * KNormW[j];
                let c = Cos[cos_base + i];
                let s = Sin[sin_base + i];
                K_cache[k_dst + i] = normed * c - nj * s;
                K_cache[k_dst + j] = nj * c + normed * s;
            }
            // V: just copy to cache (no RoPE)
            V_cache[v_dst + i] = QKV[v_src + i];
        }
    }
}
