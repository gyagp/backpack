enable subgroups;

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each thread scans N/256 elements
    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;
    var i = tid;
    for (; i < N; i = i + 256u) {
        let v = Logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i32(i);
        }
    }

    // Warp reduce: find max within 32 lanes
    for (var offset = 16u; offset > 0u; offset = offset >> 1u) {
        let other_val = subgroupShuffleXor(bitcast<i32>(local_max), offset);
        let other_idx = subgroupShuffleXor(local_idx, offset);
        let ov = bitcast<f32>(other_val);
        if (ov > local_max) {
            local_max = ov;
            local_idx = other_idx;
        }
    }

    // Write warp results to shared memory
    if (lane == 0u) {
        wg_max_val[warp_id] = local_max;
        wg_max_idx[warp_id] = local_idx;
    }
    workgroupBarrier();

    // Final reduce across 8 warps (done by thread 0)
    if (tid == 0u) {
        var best_val = wg_max_val[0];
        var best_idx = wg_max_idx[0];
        for (var w = 1u; w < 8u; w = w + 1u) {
            if (wg_max_val[w] > best_val) {
                best_val = wg_max_val[w];
                best_idx = wg_max_idx[w];
            }
        }
        Result[0] = best_idx;
    }
}
