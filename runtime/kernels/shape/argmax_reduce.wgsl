// @meta bindings=3
enable subgroups;

// Argmax Phase 2: reduce NUM_WG partial results from Phase 1.
// Grid: (1, 1, 1) — single workgroup of 256 threads.
// Each partial is 2 u32s: [bitcast<u32>(max_val), bitcast<u32>(max_idx)].
//
// Bindings: 0=Partials(u32), 1=Result(i32), 2=Params(u32)

@group(0) @binding(0) var<storage, read> Partials: array<u32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let NUM_WG = _params_[0];
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each thread scans a portion of the partial results
    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;

    var i = tid;
    for (; i < NUM_WG; i = i + 256u) {
        let pv = bitcast<f32>(Partials[i * 2u]);
        let pi = bitcast<i32>(Partials[i * 2u + 1u]);
        if (pv > local_max) {
            local_max = pv;
            local_idx = pi;
        }
    }

    // Warp reduce
    for (var offset = 16u; offset > 0u; offset = offset >> 1u) {
        let other_val = subgroupShuffleXor(bitcast<i32>(local_max), offset);
        let other_idx = subgroupShuffleXor(local_idx, offset);
        let ov = bitcast<f32>(other_val);
        if (ov > local_max) {
            local_max = ov;
            local_idx = other_idx;
        }
    }

    // Workgroup reduce across 8 warps
    if (lane == 0u) {
        wg_max_val[warp_id] = local_max;
        wg_max_idx[warp_id] = local_idx;
    }
    workgroupBarrier();

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
