@group(0) @binding(0) var<storage,read_write> X:array<f32>;
@group(0) @binding(1) var<storage,read> A:array<f32>;
@group(0) @binding(2) var<storage,read> P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let n=P[0];let i=gid.x;if(i<n){X[i]+=A[i];}}
