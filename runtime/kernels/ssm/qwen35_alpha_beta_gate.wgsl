// qwen35 DeltaNet scalar gates:
//   beta = sigmoid(beta_proj)
//   gate = softplus(alpha_proj + dt_bias) * ssm_a
//
// Bindings:
//   0: beta_proj  [num_v_heads]
//   1: alpha_proj [num_v_heads]
//   2: dt_bias    [num_v_heads]
//   3: ssm_a      [num_v_heads]
//   4: beta_out   [num_v_heads]
//   5: gate_out   [num_v_heads]
//   6: _params_   [num_v_heads, alpha_offset]

@group(0) @binding(0) var<storage, read>       beta_proj:  array<f32>;
@group(0) @binding(1) var<storage, read>       alpha_proj: array<f32>;
@group(0) @binding(2) var<storage, read>       dt_bias:    array<f32>;
@group(0) @binding(3) var<storage, read>       ssm_a:      array<f32>;
@group(0) @binding(4) var<storage, read_write> beta_out:   array<f32>;
@group(0) @binding(5) var<storage, read_write> gate_out:   array<f32>;
@group(0) @binding(6) var<storage, read>       _params_:   array<u32>;

fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = _params_[0];
    let alpha_offset = _params_[1];
    let i = gid.x;
    if (i >= n) { return; }
    let b = beta_proj[i];
    beta_out[i] = 1.0 / (1.0 + exp(-b));
    gate_out[i] = softplus(alpha_proj[alpha_offset + i] + dt_bias[i]) * ssm_a[i];
}
