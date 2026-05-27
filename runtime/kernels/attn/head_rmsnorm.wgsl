// Apply RMSNorm in-place (per-head if needed)
// Used for qwen35moe attn_q_norm + attn_k_norm dispatch
//
// Bindings:
//   0: X         array<f32>  in-place [n_head * head_dim]
//   1: W         array<f32>  norm weights [head_dim]
//   2: _params_  [n_head, head_dim, eps_as_u32_bits]

@group(0) @binding(0) var<storage, read_write> X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W:        array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_head   = _params_[0];
    let head_dim = _params_[1];
    let eps      = bitcast<f32>(_params_[2]);
    let head_idx = wid.x;
    if (head_idx >= n_head) { return; }

    let base = head_idx * head_dim;
    let tid  = lid.x;

    // RMS = sqrt(mean(x^2) + eps)
    var sum_sq: f32 = 0.0;
    for (var j = tid; j < head_dim; j = j + 64u) {
        let v = X[base + j];
        sum_sq = sum_sq + v * v;
    }
    // Workgroup reduction via shared memory
    var<workgroup> sm: array<f32, 64>;
    sm[tid] = sum_sq;
    workgroupBarrier();
    if (tid < 32u) { sm[tid] = sm[tid] + sm[tid + 32u]; } workgroupBarrier();
    if (tid < 16u) { sm[tid] = sm[tid] + sm[tid + 16u]; } workgroupBarrier();
    if (tid < 8u)  { sm[tid] = sm[tid] + sm[tid + 8u];  } workgroupBarrier();
    if (tid < 4u)  { sm[tid] = sm[tid] + sm[tid + 4u];  } workgroupBarrier();
    if (tid < 2u)  { sm[tid] = sm[tid] + sm[tid + 2u];  } workgroupBarrier();
    if (tid < 1u)  { sm[tid] = sm[tid] + sm[tid + 1u];  } workgroupBarrier();

    let inv_rms = 1.0 / sqrt(sm[0] / f32(head_dim) + eps);
    for (var j = tid; j < head_dim; j = j + 64u) {
        X[base + j] = X[base + j] * inv_rms * W[j];
    }
}
