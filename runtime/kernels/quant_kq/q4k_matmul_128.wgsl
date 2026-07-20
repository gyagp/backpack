@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,256>;
fn u8at(b:u32,o:u32)->u32{let a=b*4u+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let nb=P[2];let rs=P[3];let yo=P[4];let col=wid.y*4u+warp;var acc=0.0;
 for(var b=0u;b<nb;b++){let kb=b*256u;let k0=kb+tid;let k1=k0+128u;sx[tid]=select(0.0,X[wid.x*K+k0],k0<K);sx[tid+128u]=select(0.0,X[wid.x*K+k1],k1<K);workgroupBarrier();
  if(col<N){let bb=col*rs+b*36u;let dd=unpack2x16float(W[bb]);for(var sb=0u;sb<8u;sb++){var sc:u32;var mn:u32;if(sb<4u){sc=u8at(bb,4u+sb)&63u;mn=u8at(bb,8u+sb)&63u;}else{let j=sb-4u;let dv=u8at(bb,4u+j);let mv=u8at(bb,8u+j);let md=u8at(bb,12u+j);sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}let qv=u8at(bb,16u+(sb/2u)*32u+lane);let q=select(qv&15u,qv>>4u,(sb&1u)==1u);acc+=sx[sb*32u+lane]*(dd.x*f32(sc)*f32(q)-dd.y*f32(mn));}}
  workgroupBarrier();}
 sx[tid]=acc;workgroupBarrier();
 for(var off=16u;off>0u;off=off/2u){if(lane<off){sx[tid]+=sx[tid+off];}workgroupBarrier();}
 let s=sx[warp*32u];if(lane==0u&&col<N){Y[wid.x*N+col+yo]=s+B[col];}
}
