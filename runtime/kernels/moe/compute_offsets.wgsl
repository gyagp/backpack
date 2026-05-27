// MoE compute expert offsets — converts indices to per-direction row offsets
//
// Reads moeIndicesBuf[k] (top-k expert indices, uint32) and writes per-slot
// row offsets for the 3 indirect matmul directions:
//   gate_offsets[k] = idx * IM_e   (offset into ffn_gate_exps as row index)
//   up_offsets[k]   = idx * IM_e   (offset into ffn_up_exps)
//   down_offsets[k] = idx * E      (offset into ffn_down_exps)
//
// These offsets are then read by the indirect IQ matmul kernels via a
// separate binding, indexed by slot_idx (a per-dispatch uniform).
//
// Bindings:
//   0: indices       array<u32>  read  — [k] expert indices
//   1: gate_offsets  array<u32>  write — [k]
//   2: up_offsets    array<u32>  write — [k]
//   3: down_offsets  array<u32>  write — [k]
//   4: _params_                       — [k, IM_e, E]

@group(0) @binding(0) var<storage, read>       indices:     array<u32>;
@group(0) @binding(1) var<storage, read_write> gate_off:    array<u32>;
@group(0) @binding(2) var<storage, read_write> up_off:      array<u32>;
@group(0) @binding(3) var<storage, read_write> down_off:    array<u32>;
@group(0) @binding(4) var<storage, read>       _params_:    array<u32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k    = _params_[0];
    let IM_e = _params_[1];
    let E    = _params_[2];
    let s = gid.x;
    if (s >= k) { return; }
    let idx = indices[s];
    gate_off[s] = idx * IM_e;
    up_off[s]   = idx * IM_e;
    down_off[s] = idx * E;
}
