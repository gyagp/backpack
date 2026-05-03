// @meta bindings=4
enable subgroups;

// Multi-workgroup argmax — Phase 1: each workgroup scans a chunk and writes
// partial (max_val, max_idx) to the Partials buffer.
//
// Grid: (NUM_WG, 1, 1) where NUM_WG = params[1]
// Bindings: 0=Logits(f32), 1=Result(i32), 2=Params(u32), 3=Partials(u32)

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;
@group(0) @binding(3) var<storage, read_write> Partials: array<u32>;

const WG_SIZE: u32 = 256u;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let NUM_WG = _params_[1];
    let tid = lid.x;
    let gid = wid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each workgroup scans its chunk
    let chunk = (N + NUM_WG - 1u) / NUM_WG;
    let start = gid * chunk;
    let end = min(start + chunk, N);

    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;

    // Strided scan within chunk for coalesced memory access
    var i = start + tid;
    for (; i < end; i = i + WG_SIZE) {
        let v = Logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i32(i);
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
        // Write partial result: [val_u32, idx_i32] per workgroup
        Partials[gid * 2u] = bitcast<u32>(best_val);
        Partials[gid * 2u + 1u] = bitcast<u32>(best_idx);

        // If single workgroup, write result directly
        if (NUM_WG == 1u) {
            Result[0] = best_idx;
        }
    }
}
