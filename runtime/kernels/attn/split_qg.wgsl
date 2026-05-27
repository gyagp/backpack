// Joint Q+gate output splitter for qwen35moe attention layers
//
// qwen35moe's wq output shape: [(head_dim * 2) * n_head] per token
//   first head_dim elements per head = Q
//   next head_dim elements per head = gate (pre-sigmoid)
//
// This kernel writes Q to one buffer and gate to another, contiguous per-head.
//
// Bindings:
//   0: Qcur_full   [(2 * head_dim) * n_head] — joint output of wq matmul
//   1: Q_out       [head_dim * n_head]
//   2: gate_out    [head_dim * n_head]
//   3: _params_   — [n_head, head_dim]

@group(0) @binding(0) var<storage, read>       Qfull:    array<f32>;
@group(0) @binding(1) var<storage, read_write> Q_out:    array<f32>;
@group(0) @binding(2) var<storage, read_write> gate_out: array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head   = _params_[0];
    let head_dim = _params_[1];
    let total = n_head * head_dim;
    let i = gid.x;
    if (i >= total) { return; }
    let head_idx  = i / head_dim;
    let inner_idx = i % head_dim;
    let src_base = head_idx * (head_dim * 2u);
    Q_out[i]    = Qfull[src_base + inner_idx];
    gate_out[i] = Qfull[src_base + head_dim + inner_idx];
}
