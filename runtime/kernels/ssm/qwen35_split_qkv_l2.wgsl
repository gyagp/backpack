// Split qwen35 convolved Q/K/V and L2-normalize Q and K per head.
//
// conv_out layout:
//   Q [num_k_heads * head_dim]
//   K [num_k_heads * head_dim]
//   V [num_v_heads * head_v_dim]
//
// Bindings:
//   0: conv_out [conv_channels]
//   1: q_out    [num_k_heads * head_dim]
//   2: k_out    [num_k_heads * head_dim]
//   3: v_out    [num_v_heads * head_v_dim]
//   4: _params_ [num_k_heads, num_v_heads, head_dim, head_v_dim, eps_bits]

enable subgroups;

@group(0) @binding(0) var<storage, read>       conv_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_out:    array<f32>;
@group(0) @binding(2) var<storage, read_write> k_out:    array<f32>;
@group(0) @binding(3) var<storage, read_write> v_out:    array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const WG: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nk = _params_[0];
    let nv = _params_[1];
    let dk = _params_[2];
    let dv = _params_[3];
    let eps = bitcast<f32>(_params_[4]);
    let tid = lid.x;

    let q_size = nk * dk;
    let k_base = q_size;
    let v_base = q_size * 2u;

    if (wid.x == 0u) {
        let h = wid.y;
        if (h >= nk) { return; }
        var sum_sq: f32 = 0.0;
        for (var d = tid; d < dk; d = d + WG) {
            let v = conv_out[h * dk + d];
            sum_sq = sum_sq + v * v;
        }
        let total = subgroupAdd(sum_sq);
        let rms = 1.0 / sqrt(max(total, eps));
        for (var d = tid; d < dk; d = d + WG) {
            q_out[h * dk + d] = conv_out[h * dk + d] * rms;
        }
    } else if (wid.x == 1u) {
        let h = wid.y;
        if (h >= nk) { return; }
        var sum_sq: f32 = 0.0;
        for (var d = tid; d < dk; d = d + WG) {
            let v = conv_out[k_base + h * dk + d];
            sum_sq = sum_sq + v * v;
        }
        let total = subgroupAdd(sum_sq);
        let rms = 1.0 / sqrt(max(total, eps));
        for (var d = tid; d < dk; d = d + WG) {
            k_out[h * dk + d] = conv_out[k_base + h * dk + d] * rms;
        }
    } else {
        let h = wid.y;
        if (h >= nv) { return; }
        for (var d = tid; d < dv; d = d + WG) {
            v_out[h * dv + d] = conv_out[v_base + h * dv + d];
        }
    }
}
