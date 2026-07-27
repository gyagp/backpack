enable subgroups;
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
var<workgroup> score_scratch: array<f32, 8>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_id) sg_id: u32,
        @builtin(num_subgroups) num_sg: u32) {
    let num_heads = params[0]; let head_dim = params[1];
    let total_seq = params[2]; let kv_heads = params[3];
    let scale = bitcast<f32>(params[4]);
    let kv_stride_raw = params[5];
    let kv_stride = select(kv_stride_raw, total_seq, kv_stride_raw == 0u);
    let head = wid.y; let tile_start = wid.x * 8u;
    if (head >= num_heads || tile_start >= total_seq) { return; }
    let kv_head = head / (num_heads / kv_heads);
    let q_base = head * head_dim;
    for (var offset = 0u; offset < 8u && tile_start + offset < total_seq; offset++) {
        let position = tile_start + offset;
        let k_base = (kv_head * kv_stride + position) * head_dim;
        var score = 0.0;
        for (var dimension = lane; dimension < head_dim; dimension += 64u) {
            score += q[q_base + dimension] * k[k_base + dimension];
        }
        score = subgroupAdd(score);
        if (sg_lane == 0u) { score_scratch[sg_id] = score; }
        workgroupBarrier();
        if (lane == 0u) {
            var total = 0.0;
            for (var subgroup = 0u; subgroup < num_sg; subgroup++) { total += score_scratch[subgroup]; }
            scores[head * kv_stride + position] = total * scale;
        }
        workgroupBarrier();
    }
}
