// @meta noregistry=true
requires packed_4x8_integer_dot_product;
enable f16;
enable subgroups;
const workgroup_size_x: u32 = 256;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales_a: array<f16>;
@group(0) @binding(2) var<storage, read> input_b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> scales_b: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f16>>;
struct Uniforms {
  batch_count: u32,
  M: u32,
  N: u32,
  K: u32,
  K8: u32,
  K16: u32,
  num_M_tile: u32,
  num_N_tile: u32,
  zero_blocks_per_col: u32,
  weight_idx: u32
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<u32>;
alias input_a_indices_t = vec3<u32>;
alias output_element_t = f16;

  alias output_type = i32;
  const default_zero_point = 8;
  const bit_mask = 0xFu;
  fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> output_type {
    return output_type(default_zero_point);
  }
    alias mul_precision = output_element_t;
    fn DequantizedFrom4BitsTo8Bits(in: vec2<u32>, zero: i32) -> vec4<u32>
    {
        var out = vec4<u32>(0);
        var value_lower = vec4<i32>(unpack4xU8(in[0] & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        var value_upper = vec4<i32>(unpack4xU8((in[0] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        out[0] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
        out[1] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
        value_lower = vec4<i32>(unpack4xU8(in[1] & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        value_upper = vec4<i32>(unpack4xU8((in[1] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        out[2] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
        out[3] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
        return out;
    }
    fn SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:output_element_t) -> output_element_t
    {
        var local_sum = dot4I8Packed(a1[0], b1[0]);
        local_sum += dot4I8Packed(a1[1], b1[1]);
        local_sum += dot4I8Packed(a1[2], b1[2]);
        local_sum += dot4I8Packed(a1[3], b1[3]);
        local_sum += dot4I8Packed(a2[0], b2[0]);
        local_sum += dot4I8Packed(a2[1], b2[1]);
        local_sum += dot4I8Packed(a2[2], b2[2]);
        local_sum += dot4I8Packed(a2[3], b2[3]);
        return output_element_t(mul_precision(local_sum) * mul_precision(scale));
    }
const tile_size = 64;
const subtile_size = 16;
const tile_size_k =  32;
const vec_factor = 4;
const u32_factor = 4;
const tile_size_k_vec = 2;

var<workgroup> tile_A : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;
var<workgroup> scale_A : array<output_element_t, tile_size>;
var<workgroup> tile_B : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;
var<workgroup> scale_B : array<output_element_t, tile_size>;

fn loadSHMA(batch:u32, a_global_base:u32, kidx_v:u32, row: u32, col: u32)
{
    let a_global = a_global_base + row;
    if (a_global >= uniforms.M)
    {
        return;
    }
    tile_A[col][row] = input_a[batch*uniforms.M*uniforms.K16+a_global*uniforms.K16+kidx_v+col];
    if (col == 0)
    {

        scale_A[row] = scales_a[batch*uniforms.M*(uniforms.K/128) + a_global*(uniforms.K/128) + kidx_v/8];
    }
}

    fn loadSHMB(b_global_base:u32, kidx_v:u32, row: u32, col: u32)
    {
        let b_global = b_global_base + row;
        if (b_global >= uniforms.N)
        {
            return;
        }
        const actual_weight_idx : u32 = 0;
        let b_value = input_b[b_global * uniforms.K16+kidx_v + col];
        let block_idx = kidx_v/(32/16);
        let zero = mm_read_zero(b_global, block_idx, uniforms.N, uniforms.zero_blocks_per_col);
        tile_B[col][row] = DequantizedFrom4BitsTo8Bits(b_value, zero);
        if (col == 0)
        {

            let b_scale_offset = actual_weight_idx * uniforms.N * (uniforms.K/32);
            scale_B[row] = scales_b[b_scale_offset + b_global*(uniforms.K/32) + block_idx];
        }
    }

@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

    let batch = workgroup_idx / (uniforms.num_M_tile * uniforms.num_N_tile);
    if (batch >= uniforms.batch_count) {
        return;
    }
    let a_global_base = u32((workgroup_idx / uniforms.num_N_tile) % uniforms.num_M_tile) * tile_size;
    let b_global_base = (workgroup_idx % uniforms.num_N_tile) * tile_size;
    let load_AorB = u32(local_idx/128);
    let load_row = u32((local_idx%128)/2);
    let load_col = u32(local_idx%2);

    var subtile_id = u32(local_idx / subtile_size);
    var subtile_idx = u32(subtile_id / 4);
    var subtile_idy = u32(subtile_id % 4);
    var base_A = subtile_idx * 16;
    var base_B = subtile_idy * 16;

    var a_idx = u32(local_idx % subtile_size);

    var lane_output1: vec4<output_element_t>;
    var lane_output2: vec4<output_element_t>;
    var lane_output3: vec4<output_element_t>;
    var lane_output4: vec4<output_element_t>;
    for (var kidx_v:u32 = 0; kidx_v < uniforms.K16; kidx_v+=tile_size_k_vec)
    {

        if (load_AorB == 0)
        {
            loadSHMA(batch, a_global_base, kidx_v, load_row, load_col);
        }
        else
        {
            loadSHMB(b_global_base, kidx_v, load_row, load_col);
        }
        workgroupBarrier();

        var own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
        var own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
        var own_scale_a: output_element_t = scale_A[base_A + a_idx];

        if (sg_size == 16)
        {
            var own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
            var own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
            var own_scale_b: output_element_t  = scale_B[base_B + sg_id];

            lane_output1[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 0), own_a1, subgroupShuffle(own_b1, 0), subgroupShuffle(own_scale_b, 0) * own_scale_a);
            lane_output1[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 1), own_a1, subgroupShuffle(own_b1, 1), subgroupShuffle(own_scale_b, 1) * own_scale_a);
            lane_output1[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 2), own_a1, subgroupShuffle(own_b1, 2), subgroupShuffle(own_scale_b, 2) * own_scale_a);
            lane_output1[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 3), own_a1, subgroupShuffle(own_b1, 3), subgroupShuffle(own_scale_b, 3) * own_scale_a);

            lane_output2[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 4), own_a1, subgroupShuffle(own_b1, 4), subgroupShuffle(own_scale_b, 4) * own_scale_a);
            lane_output2[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 5), own_a1, subgroupShuffle(own_b1, 5), subgroupShuffle(own_scale_b, 5) * own_scale_a);
            lane_output2[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 6), own_a1, subgroupShuffle(own_b1, 6), subgroupShuffle(own_scale_b, 6) * own_scale_a);
            lane_output2[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 7), own_a1, subgroupShuffle(own_b1, 7), subgroupShuffle(own_scale_b, 7) * own_scale_a);

            lane_output3[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 8), own_a1, subgroupShuffle(own_b1, 8), subgroupShuffle(own_scale_b, 8) * own_scale_a);
            lane_output3[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 9), own_a1, subgroupShuffle(own_b1, 9), subgroupShuffle(own_scale_b, 9) * own_scale_a);
            lane_output3[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 10), own_a1, subgroupShuffle(own_b1, 10), subgroupShuffle(own_scale_b, 10) * own_scale_a);
            lane_output3[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 11), own_a1, subgroupShuffle(own_b1, 11), subgroupShuffle(own_scale_b, 11) * own_scale_a);

            lane_output4[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 12), own_a1, subgroupShuffle(own_b1, 12), subgroupShuffle(own_scale_b, 12) * own_scale_a);
            lane_output4[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 13), own_a1, subgroupShuffle(own_b1, 13), subgroupShuffle(own_scale_b, 13) * own_scale_a);
            lane_output4[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 14), own_a1, subgroupShuffle(own_b1, 14), subgroupShuffle(own_scale_b, 14) * own_scale_a);
            lane_output4[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 15), own_a1, subgroupShuffle(own_b1, 15), subgroupShuffle(own_scale_b, 15) * own_scale_a);
        }
        else
        {

            lane_output1[0] += SDP8AI(own_a0, tile_B[0][base_B + 0], own_a1, tile_B[1][base_B + 0],  own_scale_a * scale_B[base_B + 0]);
            lane_output1[1] += SDP8AI(own_a0, tile_B[0][base_B + 1], own_a1, tile_B[1][base_B + 1],  own_scale_a * scale_B[base_B + 1]);
            lane_output1[2] += SDP8AI(own_a0, tile_B[0][base_B + 2], own_a1, tile_B[1][base_B + 2],  own_scale_a * scale_B[base_B + 2]);
            lane_output1[3] += SDP8AI(own_a0, tile_B[0][base_B + 3], own_a1, tile_B[1][base_B + 3],  own_scale_a * scale_B[base_B + 3]);

            lane_output2[0] += SDP8AI(own_a0, tile_B[0][base_B + 4], own_a1, tile_B[1][base_B + 4],  own_scale_a * scale_B[base_B + 4]);
            lane_output2[1] += SDP8AI(own_a0, tile_B[0][base_B + 5], own_a1, tile_B[1][base_B + 5],  own_scale_a * scale_B[base_B + 5]);
            lane_output2[2] += SDP8AI(own_a0, tile_B[0][base_B + 6], own_a1, tile_B[1][base_B + 6],  own_scale_a * scale_B[base_B + 6]);
            lane_output2[3] += SDP8AI(own_a0, tile_B[0][base_B + 7], own_a1, tile_B[1][base_B + 7],  own_scale_a * scale_B[base_B + 7]);

            lane_output3[0] += SDP8AI(own_a0, tile_B[0][base_B + 8], own_a1, tile_B[1][base_B + 8],  own_scale_a * scale_B[base_B + 8]);
            lane_output3[1] += SDP8AI(own_a0, tile_B[0][base_B + 9], own_a1, tile_B[1][base_B + 9],  own_scale_a * scale_B[base_B + 9]);
            lane_output3[2] += SDP8AI(own_a0, tile_B[0][base_B + 10], own_a1, tile_B[1][base_B + 10],  own_scale_a * scale_B[base_B + 10]);
            lane_output3[3] += SDP8AI(own_a0, tile_B[0][base_B + 11], own_a1, tile_B[1][base_B + 11],  own_scale_a * scale_B[base_B + 11]);

            lane_output4[0] += SDP8AI(own_a0, tile_B[0][base_B + 12], own_a1, tile_B[1][base_B + 12],  own_scale_a * scale_B[base_B + 12]);
            lane_output4[1] += SDP8AI(own_a0, tile_B[0][base_B + 13], own_a1, tile_B[1][base_B + 13],  own_scale_a * scale_B[base_B + 13]);
            lane_output4[2] += SDP8AI(own_a0, tile_B[0][base_B + 14], own_a1, tile_B[1][base_B + 14],  own_scale_a * scale_B[base_B + 14]);
            lane_output4[3] += SDP8AI(own_a0, tile_B[0][base_B + 15], own_a1, tile_B[1][base_B + 15],  own_scale_a * scale_B[base_B + 15]);
        }
        workgroupBarrier();
    }

    let a_global = a_global_base + base_A + a_idx;
    let b_global = b_global_base + base_B;
    let output_idx = (batch * uniforms.M * uniforms.N + a_global * uniforms.N + b_global)/4;
    if (a_global < uniforms.M && b_global < uniforms.N)
    {
        output[output_idx]=lane_output1;;
        output[output_idx+1]=lane_output2;;
        output[output_idx+2]=lane_output3;;
        output[output_idx+3]=lane_output4;;
    }

}
