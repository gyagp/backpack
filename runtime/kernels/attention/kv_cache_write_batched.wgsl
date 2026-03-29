// Batched KV cache write — T tokens via workgroup_id.y.
// new_kv: [T, kv_heads * head_dim], cache: [kv_heads, max_seq, head_dim]
// Writes at positions [pastSeq, pastSeq+T)
// Params: [0]=kv_heads, [1]=head_dim, [2]=pastSeq, [3]=max_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), T, 1)

@group(0) @binding(0) var<storage, read> new_kv: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let pastSeq = _params_[2];
    let max_seq = _params_[3];
    let tok = wid.y;

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;
    let write_pos = pastSeq + tok;

    cache[(h * max_seq + write_pos) * head_dim + d] = new_kv[tok * total_elems + h * head_dim + d];
}
