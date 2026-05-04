"""
Extract L0 intermediate values from ORT for comparison with Backpack runtime.
Uses io_binding to capture internal node outputs.
"""
import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto
import os, sys

model_dir = r"e:\workspace\project\test\ai-models\gemma-4-E4B\webgpu"
onnx_dir = os.path.join(model_dir, "onnx")
decoder_path = os.path.join(onnx_dir, "decoder_model_merged_q4f16.onnx")
embed_path = os.path.join(onnx_dir, "embed_tokens_q4f16.onnx")

# 1. Get embedding
embed_sess = ort.InferenceSession(embed_path, providers=['CPUExecutionProvider'])
input_ids = np.array([[10979]], dtype=np.int64)  # "Hi" in Gemma-4 vocab
embed_out = embed_sess.run(None, {"input_ids": input_ids})
embedding = embed_out[0]  # [1,1,2560]
ple_inputs = embed_out[1]  # [1,1,42,256]
E = embedding.shape[-1]
scaled = embedding * np.sqrt(E)

print(f"Embedding (already scaled by 50.5 inside model) first4: {embedding[0,0,:4]}")
print(f"Embedding norm: {np.linalg.norm(embedding[0,0]):.4f}")
print(f"PLE inputs shape: {ple_inputs.shape}")
print(f"PLE inputs[0,0,0,:4]: {ple_inputs[0,0,0,:4]}")

# 2. Modify decoder ONNX graph to expose L0 intermediates
print("\n=== Modifying ONNX graph ===")
model = onnx.load(decoder_path)
graph = model.graph

# Tensors we want to capture
tensors_to_capture = [
    '/model/layers.0/input_layernorm/output_0',  # inputNorm output
    '/model/layers.0/attn/q_proj/MatMul/output_0',  # raw Q proj output
    '/model/layers.0/attn/k_proj/MatMul/output_0',  # raw K proj output
    '/model/layers.0/attn/v_proj/MatMul/output_0',  # raw V proj output
    '/model/layers.0/attn/v_norm/SimplifiedLayerNormalization/output_0',  # V after norm
    '/model/layers.0/attn/q_rotary/RotaryEmbedding/output_0',  # Q after RoPE
    '/model/layers.0/attn/k_rotary/RotaryEmbedding/output_0',  # K after RoPE
    '/model/layers.0/attn/GroupQueryAttention/output_0',  # attention output
    '/model/layers.0/attn/o_proj/MatMul/output_0',
    '/model/layers.0/post_attention_layernorm/output_0',
    '/model/layers.0/Add_1/output_0',
    '/model/layers.0/pre_feedforward_layernorm/output_0',
    '/model/layers.0/mlp/gate_up_proj/MatMul/output_0',
    '/model/layers.0/mlp/Mul/output_0',  # gelu(gate)*up
    '/model/layers.0/mlp/down_proj/MatMul/output_0',
    '/model/layers.0/post_feedforward_layernorm/output_0',
    '/model/layers.0/Add_2/output_0',  # after MLP residual add
    '/model/layers.0/layer_scalar/Mul/output_0',  # layer output
]

short_names = [
    'L0_inputNorm', 'L0_q_proj', 'L0_k_proj', 'L0_v_proj',
    'L0_v_norm', 'L0_q_rope', 'L0_k_rope',
    'L0_attnOut', 'L0_oproj', 'L0_postAttnNorm', 'L0_residual1', 'L0_preFfnNorm',
    'L0_gateup', 'L0_gelu_mul', 'L0_downproj', 'L0_postFfnNorm',
    'L0_residual2', 'L0_output',
]

# Add as graph outputs
for tensor_name in tensors_to_capture:
    output = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT16, None)
    graph.output.append(output)

# Save modified model with external data
modified_path = os.path.join(model_dir, "decoder_debug.onnx")
onnx.save(model, modified_path, save_as_external_data=True,
          all_tensors_to_one_file=True,
          location="decoder_debug.onnx_data",
          size_threshold=1024)
print(f"Saved modified model to {modified_path}")

# 3. Run with ORT
print("\n=== Running inference ===")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # prevent optimizations removing our outputs
sess = ort.InferenceSession(modified_path, sess_options=so, providers=['CPUExecutionProvider'])

print("Inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: shape={inp.shape} type={inp.type}")
print("Outputs:")
for out in sess.get_outputs():
    print(f"  {out.name}: shape={out.shape}")

# Prepare inputs
n_layers = 42
inputs = {
    'inputs_embeds': embedding.astype(np.float32),      # [1,1,2560] — already scaled by 50.5
    'attention_mask': np.ones((1, 1), dtype=np.int64),
    'position_ids': np.array([[0]], dtype=np.int64),
    'per_layer_inputs': (ple_inputs * 0).astype(np.float32),  # zero PLE for debugging
    'num_logits_to_keep': np.array(1, dtype=np.int64),
}

# Add empty KV caches based on what the model actually expects
for inp in sess.get_inputs():
    if inp.name.startswith('past_key_values.'):
        # Parse shape to determine head_dim
        shape = inp.shape  # e.g. ['batch_size', 2, 'past_sequence_length', 256]
        hd = shape[-1]  # last dim is head_dim
        n_kv = shape[1] if isinstance(shape[1], int) else 2
        inputs[inp.name] = np.zeros((1, n_kv, 0, hd), dtype=np.float16)

print(f"\nPrepared {len(inputs)} inputs")
outputs = sess.run(None, inputs)

# Map output names
output_names = [o.name for o in sess.get_outputs()]
output_dict = dict(zip(output_names, outputs))

print("\n=== L0 Intermediate Values ===")
for tensor_name, short_name in zip(tensors_to_capture, short_names):
    if tensor_name in output_dict:
        val = output_dict[tensor_name].astype(np.float32)
        flat = val.flatten()
        print(f"\n{short_name}:")
        print(f"  shape: {val.shape}")
        print(f"  first8: {flat[:8]}")
        print(f"  norm: {np.linalg.norm(flat):.4f}")
        print(f"  min={flat.min():.4f}, max={flat.max():.4f}")
    else:
        print(f"\n{short_name}: NOT FOUND in outputs")

# Also show KV cache for L0
for name in ['present.0.key', 'present.0.value']:
    if name in output_dict:
        val = output_dict[name]
        print(f"\n{name}: shape={val.shape} dtype={val.dtype}")
        flat = val.astype(np.float32).flatten()
        print(f"  first8: {flat[:8]}")
        print(f"  norm(head0): {np.linalg.norm(flat[:256]):.4f}")
        print(f"  norm(head1): {np.linalg.norm(flat[256:512]):.4f}")

# Also show final logits
logits_name = 'logits'
if logits_name in output_dict:
    logits = output_dict[logits_name].astype(np.float32).flatten()
    print(f"\nLogits first5: {logits[:5]}")
    print(f"Logit[10979] = {logits[10979]:.4f}")
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"Top-5: {[(int(t), logits[t]) for t in top5]}")

# Clean up
os.remove(modified_path)
if os.path.exists(modified_path + "_data"):
    os.remove(modified_path + "_data")
print("\nCleaned up temporary files")
