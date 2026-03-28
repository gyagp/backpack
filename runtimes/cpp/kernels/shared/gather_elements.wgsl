// GatherElements on data along last axis (axis=-1).
// Params: [0]=N (total elements in indices), [1]=dimSize (data last dim), [2]=outDim
// Dispatch: (ceil(N/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> data: array<${T}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let dataDim = _params_[1];
    let outDim = _params_[2];
    let idx = gid.x;
    if (idx >= N) { return; }
    let slice = idx / outDim;
    var gi = indices[idx];
    if (gi < 0) { gi = gi + i32(dataDim); }
    t_write(&output, idx, t_read(&data, slice * dataDim + u32(gi)));
}
