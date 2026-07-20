// @meta bindings=6
requires packed_4x8_integer_dot_product;
enable subgroups;

// DP4A Q8 prefill matmul. A workgroup computes 32 columns x 4 rows and
// quantizes each activation block once for all columns.
@group(0) @binding(0) var<storage,read>X:array<f32>;
@group(0) @binding(1) var<storage,read>W:array<u32>;
@group(0) @binding(2) var<storage,read>S:array<u32>;
@group(0) @binding(3) var<storage,read>Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write>Y:array<f32>;
@group(0) @binding(5) var<storage,read>P:array<u32>;
const ROWS:u32=4u;const COLS:u32=4u;const BK:u32=256u;
var<workgroup>xq:array<u32,256>;var<workgroup>xs:array<f32,32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let K=P[0];let N=P[1];let M=P[2];let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*ROWS;
 let blocks=K/32u;let words=K/4u;var cols:array<u32,COLS>;var valid:array<bool,COLS>;
 for(var c=0u;c<COLS;c++){cols[c]=wid.y*32u+warp*COLS+c;valid[c]=cols[c]<N;}
 var acc:array<f32,ROWS*COLS>;for(var i=0u;i<ROWS*COLS;i++){acc[i]=0.0;}
 for(var kb=0u;kb<K;kb+=BK){let block=tid/32u;let e=tid&31u;let pl=e&3u;let pg=e/4u;
  for(var r=0u;r<ROWS;r++){let row=row0+r;let xv=select(0.0,X[row*K+kb+tid],row<M);var am=abs(xv);
   am=max(am,subgroupShuffleXor(am,16u));am=max(am,subgroupShuffleXor(am,8u));am=max(am,subgroupShuffleXor(am,4u));am=max(am,subgroupShuffleXor(am,2u));am=max(am,subgroupShuffleXor(am,1u));
   let sc=am/127.0;if(e==0u){xs[r*8u+block]=sc;}let safe=select(1.0,sc,sc!=0.0);let q=u32(clamp(i32(round(xv/safe)),-127,127))&255u;var pack=q<<(pl*8u);pack|=subgroupShuffleXor(pack,1u);pack|=subgroupShuffleXor(pack,2u);if(pl==0u){xq[r*64u+block*8u+pg]=pack;}}
  workgroupBarrier();let xb=lane/4u;let wb=kb/32u+xb;
  for(var c=0u;c<COLS;c++){if(valid[c]){let col=cols[c];let off=col*words+kb/4u+lane*2u;let w0=W[off];let w1=W[off+1u];let si=col*blocks+wb;let sp=unpack2x16float(S[si/2u]);let ws=select(sp.x,sp.y,(si&1u)!=0u);
    for(var r=0u;r<ROWS;r++){let base=r*64u+lane*2u;let d=dot4I8Packed(xq[base],w0)+dot4I8Packed(xq[base+1u],w1);acc[c*ROWS+r]+=f32(d)*ws*xs[r*8u+xb];}}}workgroupBarrier();}
 for(var c=0u;c<COLS;c++){
  for(var r=0u;r<ROWS;r++){let sum=subgroupAdd(acc[c*ROWS+r]);if(lane==0u&&valid[c]){let row=row0+r;if(row<M){Y[row*N+cols[c]]=sum+Bias[cols[c]];}}}
 }
}
