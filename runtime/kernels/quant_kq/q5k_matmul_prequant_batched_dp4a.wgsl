// @meta bindings=6 generated=false registry=q5k_matmul_prequant_batched_dp4a
requires packed_4x8_integer_dot_product;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>W:array<u32>;
@group(0) @binding(3)var<storage,read>Bias:array<f32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
const BLOCK_WORDS:u32=44u;
var<workgroup>sxq:array<u32,512>;var<workgroup>sxs:array<f32,64>;var<workgroup>sxsum:array<f32,256>;
fn reduce32(v:f32,tid:u32)->f32{sxsum[tid]=v;workgroupBarrier();for(var off=16u;off>0u;off>>=1u){if((tid&31u)<off){sxsum[tid]+=sxsum[tid+off];}workgroupBarrier();}return sxsum[(tid/32u)*32u];}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*8u;let col=wid.y*8u+warp;
 let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];var acc:array<f32,8>;
 let qStride=(K+3u)/4u;let sStride=(K+31u)/32u;
 for(var b=0u;b<nb;b++){
  let lm=tid/32u;let ll=tid&31u;let lr=row0+lm;let lq=lm*64u+ll*2u;
  if(lr<M){let qb=lr*qStride+b*64u+ll*2u;sxq[lq]=XQ[qb];sxq[lq+1u]=XQ[qb+1u];}else{sxq[lq]=0u;sxq[lq+1u]=0u;}
  if(tid<64u){let sm=tid/8u;let ss=tid&7u;let sr=row0+sm;if(sr<M){sxs[tid]=XS[sr*sStride+b*8u+ss];}else{sxs[tid]=0.0;}}
  workgroupBarrier();let sa0=sxq[lq];let sa1=sxq[lq+1u];sxsum[tid]=f32(dot4I8Packed(sa0,0x01010101u)+dot4I8Packed(sa1,0x01010101u));workgroupBarrier();
  if(col<N){
  let sb=lane/4u;let j=sb&3u;let qgroup=sb/2u;let high=(sb&1u)!=0u;let elem0=(lane&3u)*8u;
  let base=col*rs+b*BLOCK_WORDS;let dm=unpack2x16float(W[base]);let shift=j*8u;
  let dv=(W[base+1u]>>shift)&255u;let mv=(W[base+2u]>>shift)&255u;var sc:u32;var mn:u32;
  if(sb<4u){sc=dv&63u;mn=mv&63u;}else{let md=(W[base+3u]>>shift)&255u;sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
  let payload=base+12u+qgroup*8u+elem0/4u;let p0=W[payload];let p1=W[payload+1u];let mask=0x0F0F0F0Fu;
  let lo0=select(p0&mask,(p0>>4u)&mask,high);let lo1=select(p1&mask,(p1>>4u)&mask,high);
  let qh0=((W[base+4u+elem0/4u]>>sb)&0x01010101u)<<4u;
  let qh1=((W[base+5u+elem0/4u]>>sb)&0x01010101u)<<4u;
  let w0=lo0|qh0;let w1=lo1|qh1;
  for(var m=0u;m<8u;m++){let row=row0+m;if(row<M){
   let aq0=sxq[m*64u+lane*2u];let aq1=sxq[m*64u+lane*2u+1u];let asum=sxsum[m*32u+lane];
   let dot=dot4I8Packed(aq0,w0)+dot4I8Packed(aq1,w1);
   acc[m]+=sxs[m*8u+sb]*(dm.x*f32(sc)*f32(dot)-dm.y*f32(mn)*asum);
  }}
 }workgroupBarrier();}
 for(var m=0u;m<8u;m++){let total=reduce32(acc[m],tid);let row=row0+m;if(lane==0u&&col<N&&row<M){Y[row*N+col]=total+Bias[col];}}
}
