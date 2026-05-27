// Copy one f32 buffer to another (utility for layout shuffles + residuals)
//
// Bindings:
//   0: src      array<f32> (read)
//   1: dst      array<f32> (write)
//   2: _params_ — [N, src_offset_words, dst_offset_words]

@group(0) @binding(0) var<storage, read>       src:     array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:     array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N        = _params_[0];
    let src_off  = _params_[1];
    let dst_off  = _params_[2];
    let i = gid.x;
    if (i >= N) { return; }
    dst[dst_off + i] = src[src_off + i];
}
