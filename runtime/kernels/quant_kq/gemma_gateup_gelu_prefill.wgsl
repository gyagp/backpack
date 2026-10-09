// @meta noregistry=true
// Preserve combined-scale rounding and the separate GELU product boundary.
// The host supplies P[3]=0xffffffff; these bit identities block reassociation.
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>B:array<u32>;
@group(0) @binding(3)var<storage,read>S:array<u32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
var<workgroup>sxq:array<u32,512>;
var<workgroup>sxs:array<f32,64>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*8u;let col=wid.y*8u+warp;
 let K=P[0];let N=P[1];let H=N/2u;let M=P[2];let qStride=(K+3u)/4u;
 let sStride=(K+31u)/32u;let nblocks=K/32u;let words=K/8u;
 var gate:array<f32,8>;var up:array<f32,8>;
 for(var g=0u;g<K;g+=256u){
  let lr=row0+warp;let lq=warp*64u+lane*2u;
  if(lr<M){let qb=lr*qStride+g/4u+lane*2u;sxq[lq]=XQ[qb];sxq[lq+1u]=XQ[qb+1u];}
  else{sxq[lq]=0u;sxq[lq+1u]=0u;}
  if(tid<64u){let m=tid/8u;let sb=tid&7u;let row=row0+m;
   sxs[tid]=select(0.0,XS[row*sStride+g/32u+sb],row<M);}
  workgroupBarrier();
  if(col<H){
   let qg=B[col*words+g/8u+lane];let uc=col+H;let qu=B[uc*words+g/8u+lane];
   let gb0=qg&255u;let gb1=(qg>>8u)&255u;let gb2=(qg>>16u)&255u;let gb3=qg>>24u;
   let gw0=(((gb0&15u)-8u)&255u)|((((gb0>>4u)-8u)&255u)<<8u)|((((gb1&15u)-8u)&255u)<<16u)|((((gb1>>4u)-8u)&255u)<<24u);
   let gw1=(((gb2&15u)-8u)&255u)|((((gb2>>4u)-8u)&255u)<<8u)|((((gb3&15u)-8u)&255u)<<16u)|((((gb3>>4u)-8u)&255u)<<24u);
   let ub0=qu&255u;let ub1=(qu>>8u)&255u;let ub2=(qu>>16u)&255u;let ub3=qu>>24u;
   let uw0=(((ub0&15u)-8u)&255u)|((((ub0>>4u)-8u)&255u)<<8u)|((((ub1&15u)-8u)&255u)<<16u)|((((ub1>>4u)-8u)&255u)<<24u);
   let uw1=(((ub2&15u)-8u)&255u)|((((ub2>>4u)-8u)&255u)<<8u)|((((ub3&15u)-8u)&255u)<<16u)|((((ub3>>4u)-8u)&255u)<<24u);
   let sb=lane/4u;let wb=g/32u+sb;let gsi=col*nblocks+wb;let usi=uc*nblocks+wb;
   let gsp=unpack2x16float(S[gsi/2u]);let usp=unpack2x16float(S[usi/2u]);
   let gws=select(gsp.x,gsp.y,(gsi&1u)!=0u);let uws=select(usp.x,usp.y,(usi&1u)!=0u);
   for(var m=0u;m<8u;m++){if(row0+m<M){let base=m*64u+lane*2u;
    let gd=dot4I8Packed(sxq[base],gw0)+dot4I8Packed(sxq[base+1u],gw1);
    let ud=dot4I8Packed(sxq[base],uw0)+dot4I8Packed(sxq[base+1u],uw1);
    // P[3] is all ones: retain the baseline scale-product rounding before FMA.
    let xs=sxs[m*8u+sb];let mask=P[3];
    let gs=bitcast<f32>(bitcast<u32>(xs*gws)&mask);
    let us=bitcast<f32>(bitcast<u32>(xs*uws)&mask);
    gate[m]=fma(f32(gd),gs,gate[m]);up[m]=fma(f32(ud),us,up[m]);}}
  }
  workgroupBarrier();
 }
 for(var m=0u;m<8u;m++){
  let gv=subgroupAdd(gate[m]);let uv=subgroupAdd(up[m]);let row=row0+m;
  if(lane==0u&&col<H&&row<M){let gelu=0.5*gv*(1.0+tanh(0.7978845608*(gv+0.044715*gv*gv*gv)));let orderedGelu=bitcast<f32>(bitcast<u32>(gelu)&P[3]);Y[row*H+col]=orderedGelu*uv;}
 }
}
