// Equal — compare two tensors, output packed bool (u32)
// Each thread at idx % 4 == 0 packs 4 consecutive bool results into one u32.
// Dispatch: (ceil(N/256), 1, 1)
// Params: [N, N_B]

${T_READ}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_B = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let b_idx = select(idx, idx % N_B, N_B < N && N_B > 0u);
    let eq = select(0u, 1u, t_read(&A, idx) == t_read(&B, b_idx));
    // Pack bool into bytes within u32
    let u32_idx = idx / 4u;
    let byte_pos = (idx % 4u) * 8u;
    // Note: atomicOr would be ideal but not available. Write full u32.
    // This is a simplification that works when N is a multiple of 4.
    if (idx % 4u == 0u) {
        var packed: u32 = 0u;
        for (var i = 0u; i < 4u && (idx + i) < N; i++) {
            let bi = select(idx + i, (idx + i) % N_B, N_B < N && N_B > 0u);
            let e = select(0u, 1u, t_read(&A, idx + i) == t_read(&B, bi));
            packed |= e << (i * 8u);
        }
        Out[u32_idx] = packed;
    }
}
