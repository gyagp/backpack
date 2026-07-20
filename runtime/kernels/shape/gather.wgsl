// Gather — index-based lookup along any axis
// Data is read as u32 (works for f32, packed fp16, etc.)
// Params: [outer, axisDim, innerU32, nIdx]
// Dispatch: (ceil(outer * nIdx * innerU32 / 256), 1, 1)

@group(0) @binding(0) var<storage, read> Data: array<u32>;
@group(0) @binding(1) var<storage, read> Indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outer = _params_[0];
    let axisDim = _params_[1];
    let innerU32 = _params_[2];
    let nIdx = _params_[3];

    let total = outer * nIdx * innerU32;
    let idx = gid.x;
    if (idx >= total) { return; }

    let j = idx % innerU32;
    let row = idx / innerU32;
    let i = row % nIdx;
    let outerIdx = row / nIdx;
    var gathered = Indices[i];
    if (gathered < 0) { gathered += i32(axisDim); }
    if (gathered < 0 || gathered >= i32(axisDim)) {
        Out[idx] = 0u;
        return;
    }
    let dataIdx = (outerIdx * axisDim + u32(gathered)) * innerU32 + j;
    Out[idx] = Data[dataIdx];
}
