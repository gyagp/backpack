// @meta bindings=3
@group(0) @binding(0) var<storage,read> GU:array<f32>;
@group(0) @binding(1) var<storage,read_write> O:array<f32>;
@group(0) @binding(2) var<storage,read> P:array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id)gid:vec3<u32>){let M=P[0];let N=P[1];let i=gid.x;if(i>=M*N){return;}let r=i/N;let d=i%N;let b=r*2u*N;let g=GU[b+d];O[i]=g/(1.0+exp(-g))*GU[b+N+d];}
