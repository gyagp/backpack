// Concat 2 inputs along any axis.
// Params: [0]=total_elements, [1]=axis_dim_A, [2]=axis_dim_out, [3]=inner_size
// Dispatch: ceil(total_elements / 512) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> Out: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let a_split = _params_[1];
    let total_split = _params_[2];
    let inner = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    for (var k = 0u; k < 2u; k++) {
        let idx = base + k;
        if (idx >= N) { break; }
        let outer = idx / (total_split * inner);
        let rem = idx % (total_split * inner);
        let split_pos = rem / inner;
        let inner_pos = rem % inner;
        var val: f32;
        if (split_pos < a_split) {
            val = t_read(&A, outer * a_split * inner + split_pos * inner + inner_pos);
        } else {
            let b_split = total_split - a_split;
            val = t_read(&B, outer * b_split * inner + (split_pos - a_split) * inner + inner_pos);
        }
        if (k == 0u) { v0 = val; } else { v1 = val; }
    }
    t_write2(&Out, base, v0, v1);
}
