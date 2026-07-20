// Per-head RMSNorm over DeltaNet output, then gate with silu(z).
//
// Bindings:
//   0: y        [num_v_heads * head_v_dim]
//   1: norm_w   [head_v_dim]
//   2: z        [num_v_heads * head_v_dim]
//   3: out      [num_v_heads * head_v_dim]
//   4: _params_ [num_v_heads, head_v_dim, eps_bits]

@group(0) @binding(0) var<storage, read>       y:       array<f32>;
@group(0) @binding(1) var<storage, read>       norm_w:  array<f32>;
@group(0) @binding(2) var<storage, read>       z:       array<f32>;
@group(0) @binding(3) var<storage, read_write> out:     array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const WG: u32 = 128u;
var<workgroup> sums: array<f32, 128>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn reduce_wg(v: f32, tid: u32) -> f32 {
    sums[tid] = v;
    workgroupBarrier();
    for (var offset = 64u; offset > 0u; offset = offset / 2u) {
        if (tid < offset) { sums[tid] += sums[tid + offset]; }
        workgroupBarrier();
    }
    return sums[0];
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nv = _params_[0];
    let dv = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let h = wid.x;
    let tid = lid.x;
    if (h >= nv) { return; }

    var sum_sq: f32 = 0.0;
    for (var d = tid; d < dv; d = d + WG) {
        let v = y[h * dv + d];
        sum_sq = sum_sq + v * v;
    }
    let total = reduce_wg(sum_sq, tid);
    let scale = 1.0 / sqrt(total / f32(dv) + eps);
    for (var d = tid; d < dv; d = d + WG) {
        let idx = h * dv + d;
        out[idx] = y[idx] * scale * norm_w[d] * silu(z[idx]);
    }
}
