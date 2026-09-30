// @meta bindings=6 generated=false registry=q6k_matmul_prequant_dp4a_reduc16
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read> XQ:array<u32>;
@group(0) @binding(1) var<storage,read> XS:array<f32>;
@group(0) @binding(2) var<storage,read> W:array<u32>;
@group(0) @binding(3) var<storage,read> Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
@group(0) @binding(5) var<storage,read> P:array<u32>;
fn load_u8(base:u32,off:u32)->u32{let a=base+off;return(W[a/4u]>>((a&3u)*8u))&255u;}
fn load_i8(base:u32,off:u32)->i32{let v=load_u8(base,off);return select(i32(v),i32(v)-256,v>=128u);}
fn load_u32(base:u32,off:u32)->u32{let a=base+off;let wi=a/4u;let s=(a&3u)*8u;
 if(s==0u){return W[wi];}return(W[wi]>>s)|(W[wi+1u]<<(32u-s));}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(workgroup_id) wid:vec3<u32>){
 let tid=lid.x;let lane=tid&15u;let K=P[0];let N=P[1];let nb=P[2];
 let rs=P[3];let yoff=P[4];let col=wid.y*16u+tid/16u;var acc=0.0;
 if(col<N){for(var b=0u;b<nb;b++){
  let bb=col*rs*4u+b*210u;let dh=load_u8(bb,208u)|(load_u8(bb,209u)<<8u);
  let d=unpack2x16float(dh).x;
  for(var part=0u;part<4u;part++){
   let pack=lane+part*16u;let index=pack*4u;let group=index/128u;
   let within=index-group*128u;let quarter=within/32u;let local=within&31u;
   let qlo=group*64u+select(local,32u+local,quarter==1u||quarter==3u);
   let qho=128u+group*32u+local;let ql=load_u32(bb,qlo);let qh=load_u32(bb,qho);
   let low=select(ql&0x0F0F0F0Fu,(ql>>4u)&0x0F0F0F0Fu,quarter>=2u);
   let values=low|(((qh>>(quarter*2u))&0x03030303u)<<4u);
   let signed_values=((values^0x80808080u)-0x20202020u)^0x80808080u;
   let si=group*8u+quarter*2u+local/16u;
   let ws=f32(load_i8(bb,192u+si));let xs=XS[b*8u+index/32u];
   acc+=f32(dot4I8Packed(XQ[b*64u+pack],signed_values))*xs*d*ws;
  }
 }}
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);
 acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);
 if(lane==0u&&col<N){Y[wid.x*N+col+yoff]=acc+Bias[col];}
}
