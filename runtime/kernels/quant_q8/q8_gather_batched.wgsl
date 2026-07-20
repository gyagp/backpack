// @meta bindings=5
// Gather rows from a Q8_0 table into fp32. Grid: ceil(T*D/256).
@group(0) @binding(0) var<storage,read> W:array<u32>;
@group(0) @binding(1) var<storage,read> S:array<u32>;
@group(0) @binding(2) var<storage,read> Tokens:array<i32>;
@group(0) @binding(3) var<storage,read_write> O:array<f32>;
@group(0) @binding(4) var<storage,read> P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let T=P[0];let D=P[1];let V=P[2];let i=gid.x;if(i>=T*D){return;}
 let t=i/D;let d=i%D;let raw=Tokens[t];let tok=select(0u,u32(raw),raw>=0&&u32(raw)<V);
 let p=W[tok*(D/4u)+d/4u];let q=f32(extractBits(i32(p),8u*(d&3u),8u));
 let si=tok*(D/32u)+d/32u;let sp=unpack2x16float(S[si/2u]);
 O[i]=q*select(sp.x,sp.y,(si&1u)!=0u);
}
