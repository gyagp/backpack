// @meta bindings=4
// Params [T,n_head,head_dim].
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read_write> Q:array<f32>;
@group(0) @binding(2) var<storage,read_write> G:array<f32>;
@group(0) @binding(3) var<storage,read> P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let T=P[0];let nh=P[1];let hd=P[2];let D=nh*hd;let i=gid.x;if(i>=T*D){return;}
 let t=i/D;let j=i%D;let h=j/hd;let d=j%hd;let src=t*2u*D+h*2u*hd+d;
 Q[i]=X[src];G[i]=X[src+hd];
}
