enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,2048>;
fn u8at(b:u32,o:u32)->u32{let a=b+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];
 let m0=wid.x*8u;let col=wid.y*8u+warp;var acc:array<f32,8>;
 for(var b=0u;b<nb;b++){
  let ki=b*256u+tid;for(var m=0u;m<8u;m++){sx[m*256u+tid]=select(0.0,X[(m0+m)*K+ki],m0+m<M&&ki<K);}workgroupBarrier();
  if(col<N){let bb=(col*rs+b*36u)*4u;let dd=unpack2x16float(W[bb/4u]);
   for(var sb=0u;sb<8u;sb++){var sc:u32;var mn:u32;if(sb<4u){sc=u8at(bb,4u+sb)&63u;mn=u8at(bb,8u+sb)&63u;}else{let j=sb-4u;let dv=u8at(bb,4u+j);let mv=u8at(bb,8u+j);let md=u8at(bb,12u+j);sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
    let g=sb/2u;let qb=u8at(bb,16u+g*32u+lane);let q=select(qb&15u,qb>>4u,(sb&1u)==1u);let w=dd.x*f32(sc)*f32(q)-dd.y*f32(mn);let k=sb*32u+lane;
    for(var m=0u;m<8u;m++){acc[m]+=sx[m*256u+k]*w;}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<8u;m++){let s=subgroupAdd(acc[m]);if(lane==0u&&col<N&&m0+m<M){Y[(m0+m)*N+col]=s+B[col];}}
}
