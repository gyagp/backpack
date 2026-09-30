// @meta bindings=6 generated=false registry=q4k_matmul_prequant_dp4a_reduc16
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read> XQ:array<u32>;
@group(0) @binding(1) var<storage,read> XS:array<f32>;
@group(0) @binding(2) var<storage,read> W:array<u32>;
@group(0) @binding(3) var<storage,read> Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
@group(0) @binding(5) var<storage,read> P:array<u32>;
const BLOCK_WORDS:u32=36u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(workgroup_id) wid:vec3<u32>){
 let tid=lid.x;let half_lane=tid&15u;let row=wid.y*16u+tid/16u;
 let K=P[0];let N=P[1];let nb=P[2];let rs=P[3];var acc=0.0;
 for(var b=0u;b<nb;b++){
  if(row<N){
   let sb=half_lane/2u;let j=sb&3u;let qgroup=sb/2u;
   let high=(sb&1u)!=0u;let part=(half_lane&1u)*4u;
   let aqb=b*64u+sb*8u+part;let aq0=XQ[aqb];let aq1=XQ[aqb+1u];
   let aq2=XQ[aqb+2u];let aq3=XQ[aqb+3u];
   let asum=dot4I8Packed(aq0,0x01010101u)+dot4I8Packed(aq1,0x01010101u)
           +dot4I8Packed(aq2,0x01010101u)+dot4I8Packed(aq3,0x01010101u);
   let base=row*rs+b*BLOCK_WORDS;let dm=unpack2x16float(W[base]);
   let shift=j*8u;let dv=(W[base+1u]>>shift)&255u;
   let mv=(W[base+2u]>>shift)&255u;var sc:u32;var mn:u32;
   if(sb<4u){sc=dv&63u;mn=mv&63u;}else{
    let md=(W[base+3u]>>shift)&255u;
    sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);
   }
   let payload=base+4u+qgroup*8u+part;
   let p0=W[payload];let p1=W[payload+1u];
   let p2=W[payload+2u];let p3=W[payload+3u];
   let mask=0x0F0F0F0Fu;
   let w0=select(p0&mask,(p0>>4u)&mask,high);
   let w1=select(p1&mask,(p1>>4u)&mask,high);
   let w2=select(p2&mask,(p2>>4u)&mask,high);
   let w3=select(p3&mask,(p3>>4u)&mask,high);
   let dot=dot4I8Packed(aq0,w0)+dot4I8Packed(aq1,w1)
          +dot4I8Packed(aq2,w2)+dot4I8Packed(aq3,w3);
   acc+=XS[b*8u+sb]*(dm.x*f32(sc)*f32(dot)-dm.y*f32(mn)*f32(asum));
  }
 }
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);
 acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);
 if(half_lane==0u&&row<N){Y[row]=acc+Bias[row];}
}
