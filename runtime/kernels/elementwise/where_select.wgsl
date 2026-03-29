// Where — conditional select: out = cond ? X : Y
// Cond is packed as bytes in u32 (bool array).
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> Cond: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<${T}>;
@group(0) @binding(2) var<storage, read> Y: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Out: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

fn read_cond(idx: u32) -> bool {
    let byte_idx = idx / 4u;
    let bit_pos = (idx % 4u) * 8u;
    return ((Cond[byte_idx] >> bit_pos) & 0xFFu) != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_cond = _params_[1];
    let N_x = _params_[2];
    let N_y = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let c0_idx = select(base, base % N_cond, N_cond < N && N_cond > 0u);
    let x0_idx = select(base, base % N_x, N_x < N && N_x > 0u);
    let y0_idx = select(base, base % N_y, N_y < N && N_y > 0u);
    let r0 = select(t_read(&Y, y0_idx), t_read(&X, x0_idx), read_cond(c0_idx));

    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i1 = base + 1u;
        let c1_idx = select(i1, i1 % N_cond, N_cond < N && N_cond > 0u);
        let x1_idx = select(i1, i1 % N_x, N_x < N && N_x > 0u);
        let y1_idx = select(i1, i1 % N_y, N_y < N && N_y > 0u);
        r1 = select(t_read(&Y, y1_idx), t_read(&X, x1_idx), read_cond(c1_idx));
    }

    t_write2(&Out, base, r0, r1);
}
