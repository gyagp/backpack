// F32 elementwise multiply: out = a * b
//
// Bindings:
//   0: a       array<f32> (read)
//   1: b       array<f32> (read)
//   2: out     array<f32> (write — may alias a or b)
//   3: _params_ — [N]

@group(0) @binding(0) var<storage, read>       a:       array<f32>;
@group(0) @binding(1) var<storage, read>       b:       array<f32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i >= N) { return; }
    out[i] = a[i] * b[i];
}
