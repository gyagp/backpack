// Gather — index-based lookup along axis 0
// Data is read as u32 (works for f32, packed fp16, etc.)
// Dispatch: (ceil(nIdx * sliceSize / 256), 1, 1)

@group(0) @binding(0) var<storage, read> Data: array<u32>;
@group(0) @binding(1) var<storage, read> Indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let sliceSize = _params_[1];
    let dataStride = _params_[2];

    let total = nIdx * sliceSize;
    let idx = gid.x;
    if (idx >= total) { return; }

    let i = idx / sliceSize;
    let j = idx % sliceSize;
    let dataIdx = u32(Indices[i]) * dataStride + j;
    Out[idx] = Data[dataIdx];
}
