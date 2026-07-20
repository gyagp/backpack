// @meta bindings=4
@group(0) @binding(0) var<storage,read>A:array<f32>;
@group(0) @binding(1) var<storage,read>G:array<f32>;
@group(0) @binding(2) var<storage,read_write>O:array<f32>;
@group(0) @binding(3) var<storage,read>P:array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id)gid:vec3<u32>){let i=gid.x;if(i<P[0]){O[i]=A[i]/(1.0+exp(-G[i]));}}
