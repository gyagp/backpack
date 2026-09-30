// @meta bindings=6 generated=false registry=q6k_matmul_prequant_batched_dp4a_reduc16
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>W:array<u32>;
@group(0) @binding(3)var<storage,read>Bias:array<f32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
fn load_u8(base:u32,off:u32)->u32{let a=base+off;return(W[a/4u]>>((a&3u)*8u))&255u;}
fn load_i8(base:u32,off:u32)->i32{let v=load_u8(base,off);return select(i32(v),i32(v)-256,v>=128u);}
fn load_u32(base:u32,off:u32)->u32{let a=base+off;let wi=a/4u;let s=(a&3u)*8u;if(s==0u){return W[wi];}return(W[wi]>>s)|(W[wi+1u]<<(32u-s));}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let lane=tid&15u;let row0=wid.x*8u;let col=wid.y*16u+tid/16u;
 let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];let stride=(K+3u)/4u;var acc:array<f32,8>;
 if(col<N){for(var b=0u;b<nb;b++){
  let bb=col*rs*4u+b*210u;let dh=load_u8(bb,208u)|(load_u8(bb,209u)<<8u);let d=unpack2x16float(dh).x;
  for(var part=0u;part<4u;part++){let pack=lane+part*16u;let index=pack*4u;let group=index/128u;let within=index-group*128u;let quarter=within/32u;let local=within&31u;
   let qlo=group*64u+select(local,32u+local,quarter==1u||quarter==3u);let qho=128u+group*32u+local;
   let ql=load_u32(bb,qlo);let qh=load_u32(bb,qho);let low=select(ql&0x0F0F0F0Fu,(ql>>4u)&0x0F0F0F0Fu,quarter>=2u);
   let values=low|(((qh>>(quarter*2u))&0x03030303u)<<4u);let signedValues=((values^0x80808080u)-0x20202020u)^0x80808080u;
   let si=group*8u+quarter*2u+local/16u;let ws=d*f32(load_i8(bb,192u+si));
   for(var m=0u;m<8u;m++){let row=row0+m;if(row<M){let qi=row*stride+b*64u+pack;acc[m]+=f32(dot4I8Packed(XQ[qi],signedValues))*XS[qi]*ws;}}
  }
 }}
 for(var m=0u;m<8u;m++){var total=acc[m];total+=subgroupShuffleXor(total,8u);total+=subgroupShuffleXor(total,4u);total+=subgroupShuffleXor(total,2u);total+=subgroupShuffleXor(total,1u);let row=row0+m;if(lane==0u&&col<N&&row<M){Y[row*N+col]=total+Bias[col];}}
}
