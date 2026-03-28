// KV cache append — Copy new K or V into present_key/value at position offset.
// new_kv: [batch, 1, kv_heads * head_dim]  (from current step)
// present: [batch, kv_heads, total_seq, head_dim]  (output = past + new)
// past: [batch, kv_heads, past_seq, head_dim]  (input)
// Params: [0]=kv_heads, [1]=head_dim, [2]=past_seq, [3]=total_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> new_kv: array<${T}>;
@group(0) @binding(1) var<storage, read> past: array<${T}>;
@group(0) @binding(2) var<storage, read_write> present: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let past_seq = _params_[2];
    let total_seq = _params_[3];

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;

    // Copy past values: present[h, 0..past_seq-1, d] = past[h, 0..past_seq-1, d]
    for (var s = 0u; s < past_seq; s++) {
        t_write(&present, (h * total_seq + s) * head_dim + d, t_read(&past, (h * past_seq + s) * head_dim + d));
    }

    // Append new value: present[h, past_seq, d] = new_kv[h * head_dim + d]
    t_write(&present, (h * total_seq + past_seq) * head_dim + d, t_read(&new_kv, h * head_dim + d));
}
