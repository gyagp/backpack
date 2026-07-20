// @meta bindings=9
enable f16;
// Batched Q RoPE and K RoPE + KV write. Params:
// [T,n_head,n_kv,head_dim,pos0,cache0,s0,s1] followed by [s2,s3,rope_half].
@group(0) @binding(0) var<storage,read> Q:array<f32>;
@group(0) @binding(1) var<storage,read> K:array<f32>;
@group(0) @binding(2) var<storage,read> V:array<f32>;
@group(0) @binding(3) var<storage,read_write> QO:array<f32>;
@group(0) @binding(4) var<storage,read_write> KC:array<f16>;
@group(0) @binding(5) var<storage,read_write> VC:array<f16>;
@group(0) @binding(6) var<storage,read> Cos:array<f32>;
@group(0) @binding(7) var<storage,read> Sin:array<f32>;
@group(0) @binding(8) var<storage,read> P:array<u32>;
fn local_pair(p:u32,s0:u32,s1:u32,s2:u32)->u32{if(p<s0){return p;}if(p<s0+s1){return p-s0;}if(p<s0+s1+s2){return p-s0-s1;}return p-s0-s1-s2;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
 let T=P[0];let nh=P[1];let nk=P[2];let hd=P[3];let pos0=P[4];let cache0=P[5];
 let s0=P[6];let s1=P[7];let s2=P[8];let s3=P[9];let rh=P[10];let pairs=s0+s1+s2+s3;
 let h=wid.x;let t=wid.y;let d=lid.x;if(t>=T||d>=hd){return;}let pos=pos0+t;
 if(h<nh){let base=(t*nh+h)*hd;var x=Q[base+d];if(d<2u*pairs){let p=d%pairs;let ti=pos*rh+p;let a=Q[base+p];let b=Q[base+p+pairs];x=select(a*Cos[ti]-b*Sin[ti],a*Sin[ti]+b*Cos[ti],d>=pairs);}QO[base+d]=x;}
 if(h<nk){let base=(t*nk+h)*hd;var x=K[base+d];if(d<2u*pairs){let p=d%pairs;let ti=pos*rh+local_pair(p,s0,s1,s2);let a=K[base+p];let b=K[base+p+pairs];x=select(a*Cos[ti]-b*Sin[ti],a*Sin[ti]+b*Cos[ti],d>=pairs);}let dst=((cache0+t)*nk+h)*hd+d;KC[dst]=f16(x);VC[dst]=f16(V[base+d]);}
}
