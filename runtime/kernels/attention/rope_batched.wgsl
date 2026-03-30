// @meta bindings=9
enable f16;
enable subgroups;

// Batched RoPE + KV cache scatter for prefill (NO QK-norm).
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
// Bindings 6,7 (NormQ/NormK) are unused but bound for compatibility.

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> _unused_normQ: array<f32>;
@group(0) @binding(7) var<storage, read> _unused_normK: array<f32>;
@group(0) @binding(8) var<storage, read> _params_: array<u32>;

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = wid.x;
    let t = wid.y;
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let cache_len = _params_[5];
    let n_kv = _params_[7];

    let qkv_stride = qDim + 2u * kvDim;
    let pos = pos_offset + t;

    if (head_idx < n_head) {
        // ── Q head: RoPE only, no norm ──────────────────────────────
        let src_base = t * qkv_stride + head_idx * HD;
        let dst_base = t * n_head * HD + head_idx * HD;
        let cos_base = pos * half_dim;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let qi = QKV[src_base + i];
            let qj = QKV[src_base + j];
            let c = Cos[cos_base + i];
            let s = Sin[cos_base + i];
            QRot[dst_base + i] = qi * c - qj * s;
            QRot[dst_base + j] = qj * c + qi * s;
        }
        // Non-rotary dimensions: pass through unchanged
        let rot_dim = 2u * half_dim;
        for (var i = rot_dim + tid; i < HD; i += 128u) {
            QRot[dst_base + i] = QKV[src_base + i];
        }
    } else {
        // ── KV head: RoPE + cache scatter, no norm ──────────────────
        let kv_head = head_idx - n_head;
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        let cache_pos = cache_len + t;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let ki = QKV[k_src + i];
            let kj = QKV[k_src + j];
            let c = Cos[pos * half_dim + i];
            let s = Sin[pos * half_dim + i];
            K_cache[k_dst + i] = f16(ki * c - kj * s);
            K_cache[k_dst + j] = f16(kj * c + ki * s);
        }
        // Non-rotary dimensions: pass through unchanged
        let rot_dim_k = 2u * half_dim;
        for (var i = rot_dim_k + tid; i < HD; i += 128u) {
            K_cache[k_dst + i] = f16(QKV[k_src + i]);
        }

        // V: just copy to cache
        for (var i = tid; i < HD; i += 128u) {
            V_cache[v_dst + i] = f16(QKV[v_src + i]);
        }
    }
}
