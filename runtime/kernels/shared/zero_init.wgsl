// Zero-init a buffer (used by SSM h_state reset, MoE intermediate clears, etc.)
//
// Bindings:
//   0: buf   array<f32> (write)
//   1: _params_ — [N]

@group(0) @binding(0) var<storage, read_write> buf:     array<f32>;
@group(0) @binding(1) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i < N) { buf[i] = 0.0; }
}
