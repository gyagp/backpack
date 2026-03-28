// Embedding gather: out[i] = table[token_id * E + i]
// Dispatch: (ceil(E/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> EmbeddingTable: array<${T}>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let token = u32(TokenId[0]);
    let base_offset = token * E;

    let base = gid.x * 2u;
    if (base >= E) { return; }

    let v0 = t_read(&EmbeddingTable, base_offset + base);
    var v1: f32 = 0.0;
    if (base + 1u < E) {
        v1 = t_read(&EmbeddingTable, base_offset + base + 1u);
    }
    t_write2(&X, base, v0, v1);
}
