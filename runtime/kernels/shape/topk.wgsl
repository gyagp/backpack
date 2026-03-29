// TopK — selection sort for K largest/smallest values.
// Params: [0]=totalSlices, [1]=dimSize, [2]=k, [3]=largest
// Dispatch: (totalSlices, 1, 1) — one thread per slice

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> data: array<${T}>;
@group(0) @binding(1) var<storage, read_write> out_values: array<${T}>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<i32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let totalSlices = _params_[0];
    let dimSize = _params_[1];
    let k = _params_[2];
    let largest = _params_[3];
    let slice = gid.x;
    if (slice >= totalSlices) { return; }

    let base_in = slice * dimSize;
    let base_out = slice * k;

    for (var ki = 0u; ki < k; ki++) {
        var best_val: f32 = select(1e30, -1e30, largest != 0u);
        var best_idx: u32 = 0u;

        for (var d = 0u; d < dimSize; d++) {
            let v = t_read(&data, base_in + d);
            var already = false;
            for (var p = 0u; p < ki; p++) {
                if (u32(out_indices[base_out + p]) == d) { already = true; break; }
            }
            if (already) { continue; }

            if (largest != 0u) {
                if (v > best_val) { best_val = v; best_idx = d; }
            } else {
                if (v < best_val) { best_val = v; best_idx = d; }
            }
        }

        t_write(&out_values, base_out + ki, best_val);
        out_indices[base_out + ki] = i32(best_idx);
    }
}
