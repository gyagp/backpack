// Where — conditional select: out = cond ? X : Y
// Cond is packed as bytes in u32 (bool array).
// Dispatch: (ceil(N/256), 1, 1)

@group(0) @binding(0) var<storage, read> Cond: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_cond = _params_[1];
    let N_x = _params_[2];
    let N_y = _params_[3];
    let idx = gid.x;
    if (idx >= N) { return; }
    let c_idx = select(idx, idx % N_cond, N_cond < N && N_cond > 0u);
    let x_idx = select(idx, idx % N_x, N_x < N && N_x > 0u);
    let y_idx = select(idx, idx % N_y, N_y < N && N_y > 0u);
    let byte_idx = c_idx / 4u;
    let bit_pos = (c_idx % 4u) * 8u;
    let cond_val = (Cond[byte_idx] >> bit_pos) & 0xFFu;
    Out[idx] = select(Y[y_idx], X[x_idx], cond_val != 0u);
}
