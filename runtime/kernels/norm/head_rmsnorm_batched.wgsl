// @meta bindings=3
enable subgroups;
// Params [T,n_head,head_dim,eps]. Grid (n_head,T).
@group(0) @binding(0) var<storage,read_write> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<f32>;
@group(0) @binding(2) var<storage,read> P:array<u32>;
var<workgroup>sums:array<f32,8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
 let T=P[0];let nh=P[1];let hd=P[2];let eps=bitcast<f32>(P[3]);let h=wid.x;let t=wid.y;let d=lid.x;
 if(h>=nh||t>=T){return;}let idx=(t*nh+h)*hd+d;let v=select(0.0,X[idx],d<hd);
 let s=subgroupAdd(v*v);if((d&31u)==0u){sums[d/32u]=s;}workgroupBarrier();
 var total=0.0;for(var j=0u;j<(hd+31u)/32u;j++){total+=sums[j];}let r=inverseSqrt(total/f32(hd)+eps);
 if(d<hd){X[idx]=v*r*W[d];}
}
