enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,1024>;
fn u8at(b:u32,o:u32)->u32{let a=b+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
fn i8at(b:u32,o:u32)->i32{let u=u8at(b,o);return select(i32(u),i32(u)-256,u>=128u);}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];
 let m0=wid.x*4u;let col=wid.y*8u+warp;var acc:array<f32,4>;
 for(var b=0u;b<nb;b++){
  let ki=b*256u+tid;for(var m=0u;m<4u;m++){sx[m*256u+tid]=select(0.0,X[(m0+m)*K+ki],m0+m<M&&ki<K);}workgroupBarrier();
  if(col<N){let bb=col*rs*4u+b*210u;let dh=u8at(bb,208u)|(u8at(bb,209u)<<8u);let d=unpack2x16float(dh).x;let si=lane/16u;
   for(var g=0u;g<2u;g++){let qlo=g*64u;let qho=128u+g*32u;let sco=192u+g*8u;let kb=g*128u;let ql0=u8at(bb,qlo+lane);let ql1=u8at(bb,qlo+32u+lane);let qh=u8at(bb,qho+lane);
    let q1=i32((ql0&15u)|(((qh>>0u)&3u)<<4u))-32;let q2=i32((ql1&15u)|(((qh>>2u)&3u)<<4u))-32;let q3=i32((ql0>>4u)|(((qh>>4u)&3u)<<4u))-32;let q4=i32((ql1>>4u)|(((qh>>6u)&3u)<<4u))-32;
    let w1=d*f32(i8at(bb,sco+si))*f32(q1);let w2=d*f32(i8at(bb,sco+si+2u))*f32(q2);let w3=d*f32(i8at(bb,sco+si+4u))*f32(q3);let w4=d*f32(i8at(bb,sco+si+6u))*f32(q4);
    for(var m=0u;m<4u;m++){acc[m]+=sx[m*256u+kb+lane]*w1+sx[m*256u+kb+32u+lane]*w2+sx[m*256u+kb+64u+lane]*w3+sx[m*256u+kb+96u+lane]*w4;}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<4u;m++){let s=subgroupAdd(acc[m]);if(lane==0u&&col<N&&m0+m<M){Y[(m0+m)*N+col]=s+B[col];}}
}
