"""
Compare Gemma-4-E4B embedding dequantization between ORT and manual Q4 decode.
"""
import numpy as np
import onnxruntime as ort
import onnx
import os

model_dir = r"e:\workspace\project\test\ai-models\gemma-4-E4B\webgpu"
onnx_dir = os.path.join(model_dir, "onnx")
embed_path = os.path.join(onnx_dir, "embed_tokens_q4f16.onnx")

# Load with ORT
embed_sess = ort.InferenceSession(embed_path, providers=['CPUExecutionProvider'])
input_ids = np.array([[2439]], dtype=np.int64)
embed_out = embed_sess.run(None, {"input_ids": input_ids})
ort_emb = embed_out[0][0, 0]  # [2560]
print(f"ORT embedding for token 2439:")
print(f"  first8: {ort_emb[:8]}")
print(f"  norm: {np.linalg.norm(ort_emb):.4f}")

# Now load the ONNX model to examine the embedding tensor
model = onnx.load(embed_path)
graph = model.graph

print(f"\n=== Embed model initializers ===")
for init in graph.initializer:
    dims = list(init.dims)
    print(f"  {init.name}: type={init.data_type} dims={dims}")

# Find the embedding Q4 tensor
for init in graph.initializer:
    if "embed_tokens" in init.name and "quant" in init.name and "per_layer" not in init.name:
        print(f"\n=== Q4 embedding tensor: {init.name} ===")
        print(f"  data_type={init.data_type} dims={list(init.dims)}")
        print(f"  raw_data size={len(init.raw_data)} bytes")

        # Parse Q4 data for token 2439
        nVocab = init.dims[0]
        packedK = init.dims[1] if len(init.dims) >= 2 else 0
        embK = packedK * 2  # 2 nibbles per byte
        print(f"  nVocab={nVocab} packedK={packedK} embK={embK}")

        qData = np.frombuffer(init.raw_data, dtype=np.uint8)
        print(f"  qData shape={qData.shape}")

        # Get row for token 2439
        rowOffset = 2439 * packedK
        rowBytes = qData[rowOffset:rowOffset + packedK]
        # Unpack nibbles
        low = rowBytes & 0x0F
        high = rowBytes >> 4
        nibbles = np.empty(embK, dtype=np.uint8)
        nibbles[0::2] = low
        nibbles[1::2] = high
        print(f"  Token 2439 nibbles first16: {nibbles[:16]}")

        # Find scales
        for sinit in graph.initializer:
            if "embed_tokens" in sinit.name and "scales" in sinit.name and "per_layer" not in sinit.name:
                print(f"\n  Scales tensor: {sinit.name} dims={list(sinit.dims)}")
                # Parse scales
                scales_raw = np.frombuffer(sinit.raw_data, dtype=np.float16)
                nGroups = sinit.dims[1] if len(sinit.dims) >= 2 else 1
                blockSize = embK // nGroups if nGroups > 0 else embK
                print(f"  nGroups={nGroups} blockSize={blockSize}")

                # Get scales for token 2439
                rowScales = scales_raw[2439 * nGroups : (2439 + 1) * nGroups]
                print(f"  Token 2439 scales first8: {rowScales[:8].astype(np.float32)}")
                break

        # Find zero points
        zpData = None
        for zinit in graph.initializer:
            if "embed_tokens" in zinit.name and "zp" in zinit.name and "per_layer" not in zinit.name:
                print(f"\n  ZP tensor: {zinit.name} dims={list(zinit.dims)}")
                zpData = np.frombuffer(zinit.raw_data, dtype=np.uint8)
                # Get ZP for token 2439
                nGroups_zp = nGroups
                zpRow = zpData[2439 * nGroups_zp // 2 : (2439 * nGroups_zp + nGroups_zp + 1) // 2]
                print(f"  Token 2439 ZP raw bytes: {zpRow[:8]}")
                break

        # Manual dequant for token 2439
        print(f"\n=== Manual dequant for token 2439 ===")
        dequant = np.zeros(embK, dtype=np.float32)
        for g in range(nGroups):
            scale = float(rowScales[g])
            zp = 8  # default
            if zpData is not None:
                zpIdx = 2439 * nGroups + g
                zpByte = zpData[zpIdx // 2]
                zp = int((zpByte >> 4) if (zpIdx & 1) else (zpByte & 0x0F))
            colBase = g * blockSize
            for j in range(blockSize):
                col = colBase + j
                if col >= embK:
                    break
                byteIdx = 2439 * packedK + col // 2
                byte = int(qData[byteIdx])
                nibble = (byte >> 4) if (col & 1) else (byte & 0x0F)
                dequant[col] = (float(nibble) - float(zp)) * scale

        print(f"  Manual first8: {dequant[:8]}")
        print(f"  Manual norm: {np.linalg.norm(dequant):.4f}")
        print(f"  ORT    first8: {ort_emb[:8]}")
        print(f"  ORT    norm: {np.linalg.norm(ort_emb):.4f}")

        # Check max abs diff
        diff = np.abs(dequant - ort_emb)
        print(f"\n  Max abs diff: {np.max(diff):.6f}")
        print(f"  Mean abs diff: {np.mean(diff):.6f}")

        # Check if our dequant matches ORT at all
        if np.max(diff) < 0.01:
            print("  ✓ Dequant matches ORT!")
        else:
            print("  ✗ Dequant does NOT match ORT!")
            # Find first mismatch
            for i in range(min(16, embK)):
                if abs(dequant[i] - ort_emb[i]) > 0.01:
                    print(f"    Mismatch at [{i}]: manual={dequant[i]:.6f} ORT={ort_emb[i]:.6f}")
        break

# Also check token 0 (our runtime seems to be using token 0)
print(f"\n=== Also checking token 0 ===")
input_ids_0 = np.array([[0]], dtype=np.int64)
embed_out_0 = embed_sess.run(None, {"input_ids": input_ids_0})
ort_emb_0 = embed_out_0[0][0, 0]
print(f"ORT embedding for token 0 first8: {ort_emb_0[:8]}")
print(f"ORT embedding for token 0 norm: {np.linalg.norm(ort_emb_0):.4f}")

# Check what our runtime should produce: raw Q4 dequant (no ×50.5) then ×50.596
# If our C++ raw embedding for token 0 is [-0.035034 0.004379 0.030655 -0.008759 0.008759 -0.021896 0.008759 0.021896]
# Then scaled: [-0.035034*50.596, ...] = [-1.7726, 0.2216, 1.5510, -0.4432]
# These match our runtime's scaled embedding: [-1.7726 0.2216 1.5510 -0.4432] norm=50.57

# Manual Q4 dequant for token 0
row0_offset = 0 * packedK
row0_bytes = qData[row0_offset:row0_offset + packedK]
low0 = row0_bytes & 0x0F
high0 = row0_bytes >> 4
nibbles0 = np.empty(embK, dtype=np.uint8)
nibbles0[0::2] = low0
nibbles0[1::2] = high0
dequant0 = np.zeros(embK, dtype=np.float32)
rowScales0 = scales_raw[0 * nGroups : (0 + 1) * nGroups]
for g in range(nGroups):
    scale = float(rowScales0[g])
    zp = 8
    if zpData is not None:
        zpIdx = 0 * nGroups + g
        zpByte = int(zpData[zpIdx // 2])
        zp = (zpByte >> 4) if (zpIdx & 1) else (zpByte & 0x0F)
    colBase = g * blockSize
    for j in range(blockSize):
        col = colBase + j
        if col >= embK: break
        byteIdx = 0 * packedK + col // 2
        byte_val = int(qData[byteIdx])
        nibble = (byte_val >> 4) if (col & 1) else (byte_val & 0x0F)
        dequant0[col] = (float(nibble) - float(zp)) * scale
print(f"Python manual dequant token 0 first8: {dequant0[:8]}")
print(f"Python manual dequant token 0 norm: {np.linalg.norm(dequant0):.4f}")
print(f"C++ runtime claims raw first8: [-0.035034 0.004379 0.030655 -0.008759 0.008759 -0.021896 0.008759 0.021896]")
print(f"C++ runtime claims raw norm: 0.9995")
print(f"\n=== Nodes in embed model ===")
for node in graph.node:
    print(f"  {node.op_type} name={node.name}")
    print(f"    inputs: {list(node.input)}")
    print(f"    outputs: {list(node.output)}")
    for attr in node.attribute:
        print(f"    attr: {attr.name}={attr.i if attr.type==2 else attr.ints if attr.type==7 else '?'}")
