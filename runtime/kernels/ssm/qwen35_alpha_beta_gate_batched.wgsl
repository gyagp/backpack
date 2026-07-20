// @meta bindings=6
// Params [T,rank]. Raw rows contain beta projection followed by alpha.
@group(0) @binding(0) var<storage,read> Raw:array<f32>;
@group(0) @binding(1) var<storage,read> Dt:array<f32>;
@group(0) @binding(2) var<storage,read> A:array<f32>;
@group(0) @binding(3) var<storage,read_write> Beta:array<f32>;
@group(0) @binding(4) var<storage,read_write> Gate:array<f32>;
@group(0) @binding(5) var<storage,read> P:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let T=P[0];let R=P[1];let i=gid.x;if(i>=T*R){return;}let t=i/R;let j=i%R;let b=Raw[t*2u*R+j];let a=Raw[t*2u*R+R+j]+Dt[j];Beta[i]=1.0/(1.0+exp(-b));Gate[i]=(max(a,0.0)+log(1.0+exp(-abs(a))))*A[j];}
