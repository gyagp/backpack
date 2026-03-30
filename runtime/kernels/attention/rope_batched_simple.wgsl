enable f16;
enable subgroups;

// Simple batched RoPE + QK-norm + KV cache scatter for prefill.
// Each workgroup handles ONE head for ONE token.
// Grid: (n_heads_total, T, 1) where n_heads_total = n_head + n_kv
//   WG (head < n_head, t): processes Q head
//   WG (head >= n_head, t): processes K head + V scatter
//
// QKV layout: [T × (qDim + 2*kvDim)] row-major
// Output Q: QRot[T × qDim] row-major (T × n_head × HD)
// Output K/V: scattered into KV cache at positions pos_offset..pos_offset+T-1
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, cache_len, eps_u32, n_kv]

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QNormW: array<f32>;
@group(0) @binding(7) var<storage, read> KNormW: array<f32>;
@group(0) @binding(8) var<storage, read> _params_: array<u32>;

const HD: u32 = 128u;
const HALF: u32 = 64u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = wid.x;   // which head (0..n_head-1 for Q, n_head..n_head+n_kv-1 for K)
    let t = wid.y;           // which token in the batch
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let cache_len = _params_[5];
    let eps = bitcast<f32>(_params_[6]);
    let n_kv = _params_[7];

    let qkv_stride = qDim + 2u * kvDim;
    let pos = pos_offset + t;

    if (head_idx < n_head) {
        // ── Q head processing ────────────────────────────────────────
        let src_base = t * qkv_stride + head_idx * HD;

        // Load Q values and compute RMS norm
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            let v = QKV[src_base + i];
            sum_sq += v * v;
        }
        // Warp reduce
        let warp_sum = subgroupAdd(sum_sq);
        var smem: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { smem[warp_id] = warp_sum; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = smem[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply norm + RoPE, write to QRot
        let dst_base = t * n_head * HD + head_idx * HD;
        let cos_base = pos * half_dim;
        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let qi = QKV[src_base + i] * rstd * QNormW[i];
            let qj = QKV[src_base + j] * rstd * QNormW[j];
            let c = Cos[cos_base + i];
            let s = Sin[cos_base + i];
            QRot[dst_base + i] = qi * c - qj * s;
            QRot[dst_base + j] = qj * c + qi * s;
        }
        // Non-rotary dimensions: apply norm only (no rotation)
        let rot_dim = 2u * half_dim;
        for (var i = rot_dim + tid; i < HD; i += 128u) {
            QRot[dst_base + i] = QKV[src_base + i] * rstd * QNormW[i];
        }
    } else {
        // ── KV head processing ───────────────────────────────────────
        let kv_head = head_idx - n_head;
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        // K: load, norm, RoPE, scatter to cache
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            let v = QKV[k_src + i];
            sum_sq += v * v;
        }
        let warp_sum = subgroupAdd(sum_sq);
        var smem2: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { smem2[warp_id] = warp_sum; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = smem2[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // KV cache layout: [MAX_SEQ × n_kv × HD]
        let cache_pos = cache_len + t;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let ki = QKV[k_src + i] * rstd * KNormW[i];
            let kj = QKV[k_src + j] * rstd * KNormW[j];
            let c = Cos[pos * half_dim + i];
            let s = Sin[pos * half_dim + i];
            K_cache[k_dst + i] = f16(ki * c - kj * s);
            K_cache[k_dst + j] = f16(kj * c + ki * s);
        }
        // Non-rotary dimensions: apply norm only (no rotation)
        let rot_dim_k = 2u * half_dim;
        for (var i = rot_dim_k + tid; i < HD; i += 128u) {
            K_cache[k_dst + i] = f16(QKV[k_src + i] * rstd * KNormW[i]);
        }

        // V: just copy to cache (no RoPE, no norm)
        for (var i = tid; i < HD; i += 128u) {
            V_cache[v_dst + i] = f16(QKV[v_src + i]);
        }
    }
}
