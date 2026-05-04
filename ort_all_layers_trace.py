"""
Trace ALL 42 layer outputs from Gemma-4-E4B ONNX decoder model using ORT.
Captures /model/layers.{i}/layer_scalar/Mul/output_0 for each layer (0-41).
Uses ACTUAL PLE inputs (not zeroed).
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

# 1. Get embedding for token 10979 ("Hi")
print("=== Step 1: Embedding ===")
embed_sess = ort.InferenceSession(embed_path, providers=['CPUExecutionProvider'])
input_ids = np.array([[10979]], dtype=np.int64)
embed_out = embed_sess.run(None, {"input_ids": input_ids})
embedding = embed_out[0]   # [1,1,2560]
ple_inputs = embed_out[1]  # [1,1,42,256]

print(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding[0,0].astype(np.float32)):.4f}")
print(f"Embedding first4: {embedding[0,0,:4]}")
print(f"PLE inputs shape: {ple_inputs.shape}")
print(f"PLE inputs[0,0,0,:4]: {ple_inputs[0,0,0,:4]}")
print(f"PLE inputs norm (layer 0): {np.linalg.norm(ple_inputs[0,0,0].astype(np.float32)):.4f}")

# 2. Modify decoder ONNX graph to expose all 42 layer outputs
print("\n=== Step 2: Modifying ONNX graph to expose all layer outputs ===")
model = onnx.load(decoder_path)
graph = model.graph

n_layers = 42
tensors_to_capture = []
short_names = []
for i in range(n_layers):
    tensor_name = f"/model/layers.{i}/layer_scalar/Mul/output_0"
    tensors_to_capture.append(tensor_name)
    short_names.append(f"L{i}_output")

# Add as graph outputs
for tensor_name in tensors_to_capture:
    output = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT16, None)
    graph.output.append(output)

# Save modified model with external data
modified_path = os.path.join(model_dir, "decoder_all_layers_debug.onnx")
onnx.save(model, modified_path, save_as_external_data=True,
          all_tensors_to_one_file=True,
          location="decoder_all_layers_debug.onnx_data",
          size_threshold=1024)
print(f"Saved modified model to {modified_path}")

# 3. Run inference with actual PLE inputs
print("\n=== Step 3: Running inference ===")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

sess = ort.InferenceSession(modified_path, sess_options=so, providers=['CPUExecutionProvider'])

print("Model inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: shape={inp.shape} type={inp.type}")

# Prepare inputs — use ACTUAL PLE inputs (not zeroed)
inputs = {
    'inputs_embeds': embedding.astype(np.float32),
    'attention_mask': np.ones((1, 1), dtype=np.int64),
    'position_ids': np.array([[0]], dtype=np.int64),
    'per_layer_inputs': ple_inputs.astype(np.float32),  # actual PLE, NOT zeroed
    'num_logits_to_keep': np.array(1, dtype=np.int64),
}

# Add empty KV caches
for inp in sess.get_inputs():
    if inp.name.startswith('past_key_values.'):
        shape = inp.shape
        hd = shape[-1]
        n_kv = shape[1] if isinstance(shape[1], int) else 2
        inputs[inp.name] = np.zeros((1, n_kv, 0, hd), dtype=np.float16)

print(f"Prepared {len(inputs)} inputs")
print("Running inference (this may take a while on CPU)...")
outputs = sess.run(None, inputs)

# Map output names
output_names = [o.name for o in sess.get_outputs()]
output_dict = dict(zip(output_names, outputs))

# 4. Print per-layer output norms
print("\n=== All Layer Output Norms ===")
print(f"{'Layer':<8} {'Norm':>12} {'Min':>12} {'Max':>12} {'First4'}")
print("-" * 80)
for tensor_name, short_name in zip(tensors_to_capture, short_names):
    if tensor_name in output_dict:
        val = output_dict[tensor_name].astype(np.float32)
        flat = val.flatten()
        norm = np.linalg.norm(flat)
        print(f"{short_name:<8} {norm:>12.4f} {flat.min():>12.4f} {flat.max():>12.4f} {flat[:4]}")
    else:
        print(f"{short_name:<8} NOT FOUND in outputs")

# 5. Print final logits info
print("\n=== Final Logits ===")
logits_name = 'logits'
if logits_name in output_dict:
    logits = output_dict[logits_name].astype(np.float32).flatten()
    print(f"Logits shape: {output_dict[logits_name].shape}")
    print(f"Logits first5: {logits[:5]}")
    print(f"Logit[10979] = {logits[10979]:.4f}")
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"Top-5 tokens: {[(int(t), float(logits[t])) for t in top5]}")
else:
    print("logits NOT FOUND in outputs")

# 6. Clean up temporary files
try:
    os.remove(modified_path)
except OSError:
    pass
data_path = modified_path + "_data"
if not os.path.exists(data_path):
    data_path = os.path.join(model_dir, "decoder_all_layers_debug.onnx_data")
try:
    os.remove(data_path)
except OSError:
    pass
print("\nCleaned up temporary files")
