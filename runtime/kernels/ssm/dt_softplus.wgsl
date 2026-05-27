// SSM dt softplus + bias — Mamba delta-time activation
//
// In Mamba's decode step, dt is computed as:
//   dt = softplus(linear(x) + dt_bias)
// where softplus(z) = log(1 + exp(z))
//
// Bindings:
//   0: dt_proj_out [d_inner]  — raw linear projection output
//   1: dt_bias     [d_inner]
//   2: dt_out      [d_inner]
//   3: _params_              — [d_inner]

@group(0) @binding(0) var<storage, read>       proj:    array<f32>;
@group(0) @binding(1) var<storage, read>       bias:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dt_out:  array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let c = gid.x;
    if (c >= d_inner) { return; }
    let z = proj[c] + bias[c];
    // softplus, numerically stable: max(z, 0) + log(1 + exp(-|z|))
    let abs_z = abs(z);
    dt_out[c] = max(z, 0.0) + log(1.0 + exp(-abs_z));
}
