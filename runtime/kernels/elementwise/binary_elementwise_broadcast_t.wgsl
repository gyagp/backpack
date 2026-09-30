// @meta noregistry=true raw_template=true
// Binary elementwise ops: Add(0), Sub(1), Mul(2), Div(3)
// With broadcasting support.
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> C: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

fn compute_op(a: f32, b: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        default: { return a + b; }
    }
}

fn broadcast_index(flat: u32, dims_offset: u32) -> u32 {
    let rank = _params_[0];
    if (rank == 0u) { return flat % _params_[dims_offset]; }
    var input_idx = 0u;
    var input_stride = 1u;
    var output_stride = 1u;
    var axis = rank;
    loop {
        if (axis == 0u) { break; }
        axis -= 1u;
        let output_dim = _params_[12u + axis];
        let coord = (flat / output_stride) % output_dim;
        let input_dim = _params_[dims_offset + axis];
        input_idx += select(coord * input_stride, 0u, input_dim == 1u);
        input_stride *= input_dim;
        output_stride *= output_dim;
    }
    return input_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let op = _params_[2];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let i0 = base;
    let a0_idx = broadcast_index(i0, 4u);
    let b0_idx = broadcast_index(i0, 8u);
    let r0 = compute_op(t_read(&A, a0_idx), t_read(&B, b0_idx), op);

    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i1 = base + 1u;
        let a1_idx = broadcast_index(i1, 4u);
        let b1_idx = broadcast_index(i1, 8u);
        r1 = compute_op(t_read(&A, a1_idx), t_read(&B, b1_idx), op);
    }

    t_write2(&C, base, r0, r1);
}
