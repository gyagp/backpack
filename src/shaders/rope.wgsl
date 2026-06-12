@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : array<vec4u, 2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let head_dim = params[0].x;
    let num_heads = params[0].y;
    let theta_base_bits = params[0].z;
    let theta_base = bitcast<f32>(theta_base_bits);
    let seq_pos_offset = params[1].x;
    let M = params[1].y;

    let idx = gid.x;
    let total = M * num_heads * head_dim;
    if (idx >= total) {
        return;
    }

    let token_idx = idx / (num_heads * head_dim);
    let within_token = idx % (num_heads * head_dim);
    let d = within_token % head_dim;
    let pos = seq_pos_offset + token_idx;
    let half_dim = head_dim / 2u;

    if (d < half_dim) {
        let freq = 1.0 / pow(theta_base, f32(d) / f32(half_dim));
        let angle = f32(pos) * freq;
        let cos_a = cos(angle);
        let sin_a = sin(angle);

        let base = token_idx * num_heads * head_dim + within_token - d;
        let x0 = input[base + d];
        let x1 = input[base + d + half_dim];

        output[base + d] = x0 * cos_a - x1 * sin_a;
        output[base + d + half_dim] = x0 * sin_a + x1 * cos_a;
    }
}
