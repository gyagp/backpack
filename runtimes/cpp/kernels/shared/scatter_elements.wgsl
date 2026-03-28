// ScatterElements on data along last axis.
// Copies data → output, then scatters updates at index positions.
// Params: [0]=dataN, [1]=dataDim, [2]=idxN, [3]=idxDim, [4]=mode (0=copy, 1=scatter)
// Dispatch: (ceil(dataN/256), 1, 1) for copy, (ceil(idxN/256), 1, 1) for scatter

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> data: array<${T}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> updates: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let mode = _params_[4];
    if (mode == 0u) {
        // Copy pass: data → output
        let dataN = _params_[0];
        let idx = gid.x;
        if (idx >= dataN) { return; }
        t_write(&output, idx, t_read(&data, idx));
    } else {
        // Scatter pass: write updates at indexed positions
        let dataDim = _params_[1];
        let idxN = _params_[2];
        let idxDim = _params_[3];
        let idx = gid.x;
        if (idx >= idxN) { return; }
        let slice = idx / idxDim;
        var gi = indices[idx];
        if (gi < 0) { gi = gi + i32(dataDim); }
        let dst = slice * dataDim + u32(gi);
        t_write(&output, dst, t_read(&updates, idx));
    }
}
