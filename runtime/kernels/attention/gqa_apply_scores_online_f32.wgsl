@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = params[0]; let head_dim = params[1];
    let total_seq = params[2]; let kv_heads = params[3];
    let kv_stride_raw = params[5];
    let kv_stride = select(kv_stride_raw, total_seq, kv_stride_raw == 0u);
    let head = wid.y;
    if (head >= num_heads) { return; }
    let kv_head = head / (num_heads / kv_heads);
    let q_base = head * head_dim;
    var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;
    var running_max = -1e30; var running_sum = 0.0;
    for (var position = 0u; position < total_seq; position++) {
        let score = scores[head * kv_stride + position];
        let next_max = max(running_max, score);
        let previous_factor = exp(running_max - next_max);
        let score_factor = exp(score - next_max);
        let next_sum = running_sum * previous_factor + score_factor;
        let rescale = running_sum * previous_factor / max(next_sum, 1e-10);
        let weight = score_factor / max(next_sum, 1e-10);
        let value_base = (kv_head * kv_stride + position) * head_dim;
        acc0 = acc0 * rescale + weight * value[value_base + lane];
        if (lane + 64u < head_dim) { acc1 = acc1 * rescale + weight * value[value_base + lane + 64u]; }
        if (lane + 128u < head_dim) { acc2 = acc2 * rescale + weight * value[value_base + lane + 128u]; }
        if (lane + 192u < head_dim) { acc3 = acc3 * rescale + weight * value[value_base + lane + 192u]; }
        running_max = next_max; running_sum = next_sum;
    }
    output[q_base + lane] = acc0;
    if (lane + 64u < head_dim) { output[q_base + lane + 64u] = acc1; }
    if (lane + 128u < head_dim) { output[q_base + lane + 128u] = acc2; }
    if (lane + 192u < head_dim) { output[q_base + lane + 192u] = acc3; }
}
