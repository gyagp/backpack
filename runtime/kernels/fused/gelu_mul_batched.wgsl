// @meta bindings=3

@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M=P[0]; let N=P[1]; let flat=gid.x;
    if (flat>=M*N) { return; }
    let row=flat/N; let i=flat%N; let base=row*2u*N;
    let g=GateUp[base+i]; let u=GateUp[base+N+i];
    let gelu=0.5*g*(1.0+tanh(0.7978845608*(g+0.044715*g*g*g)));
    Out[flat]=gelu*u;
}
