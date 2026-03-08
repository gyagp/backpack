@group(0) @binding(0) var<storage, read> EmbeddingTable: array<f32>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let idx = gid.x;
    if (idx >= E) { return; }
    let token = u32(TokenId[0]);
    X[idx] = EmbeddingTable[token * E + idx];
}
