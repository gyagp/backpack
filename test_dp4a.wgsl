requires packed_4x8_integer_dot_product;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> A: array<u32>;
@group(0) @binding(1) var<storage, read_write> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<i32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let a = A[lid.x];
    let b = B[lid.x];
    C[lid.x] = dot4I8Packed(a, b);
}
