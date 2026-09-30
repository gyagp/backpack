// @meta bindings=6 generated=false registry=q6k_matmul_prequant_batched_dp4a
requires packed_4x8_integer_dot_product;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>W:array<u32>;
@group(0) @binding(3)var<storage,read>Bias:array<f32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
var<workgroup>sxq:array<u32,512>;var<workgroup>sxs:array<f32,512>;var<workgroup>scratch:array<f32,256>;
fn load_u8(base:u32,off:u32)->u32{let a=base+off;return(W[a/4u]>>((a&3u)*8u))&255u;}
fn load_i8(base:u32,off:u32)->i32{let v=load_u8(base,off);return select(i32(v),i32(v)-256,v>=128u);}
fn load_u32(base:u32,off:u32)->u32{let a=base+off;let wi=a/4u;let s=(a&3u)*8u;if(s==0u){return W[wi];}return(W[wi]>>s)|(W[wi+1u]<<(32u-s));}
fn reduce32(v:f32,tid:u32)->f32{scratch[tid]=v;workgroupBarrier();for(var off=16u;off>0u;off>>=1u){if((tid&31u)<off){scratch[tid]+=scratch[tid+off];}workgroupBarrier();}return scratch[(tid/32u)*32u];}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*8u;let col=wid.y*8u+warp;
 let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];let qStride=(K+3u)/4u;let sStride=qStride;var acc:array<f32,8>;
 for(var b=0u;b<nb;b++){
  let lm=tid/32u;let ll=tid&31u;let lr=row0+lm;let lq=lm*64u+ll*2u;
  if(lr<M){let qb=lr*qStride+b*64u+ll*2u;sxq[lq]=XQ[qb];sxq[lq+1u]=XQ[qb+1u];}else{sxq[lq]=0u;sxq[lq+1u]=0u;}
  if(lr<M){sxs[lq]=XS[lr*sStride+b*64u+ll*2u];sxs[lq+1u]=XS[lr*sStride+b*64u+ll*2u+1u];}else{sxs[lq]=0.0;sxs[lq+1u]=0.0;}
  workgroupBarrier();
  if(col<N){let bb=col*rs*4u+b*210u;let dh=load_u8(bb,208u)|(load_u8(bb,209u)<<8u);let d=unpack2x16float(dh).x;
   for(var half=0u;half<2u;half++){let pack=lane+half*32u;let index=pack*4u;let group=index/128u;let within=index-group*128u;let quarter=within/32u;let local=within&31u;
    let qlo=group*64u+select(local,32u+local,quarter==1u||quarter==3u);let qho=128u+group*32u+local;
    let ql=load_u32(bb,qlo);let qh=load_u32(bb,qho);let low=select(ql&0x0F0F0F0Fu,(ql>>4u)&0x0F0F0F0Fu,quarter>=2u);
    let values=low|(((qh>>(quarter*2u))&0x03030303u)<<4u);let signedValues=((values^0x80808080u)-0x20202020u)^0x80808080u;
    let si=group*8u+quarter*2u+local/16u;let ws=d*f32(load_i8(bb,192u+si));
    for(var m=0u;m<8u;m++){let row=row0+m;if(row<M){acc[m]+=f32(dot4I8Packed(sxq[m*64u+pack],signedValues))*sxs[m*64u+pack]*ws;}}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<8u;m++){let total=reduce32(acc[m],tid);let row=row0+m;if(lane==0u&&col<N&&row<M){Y[row*N+col]=total+Bias[col];}}
}
