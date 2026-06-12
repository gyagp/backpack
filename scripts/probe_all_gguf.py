#!/usr/bin/env python3
"""Batch probe script that opens each GGUF file, parses header, extracts arch + config."""

import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747

GGUF_VALUE_READERS = {
    0: ("B", 1),    # UINT8
    1: ("b", 1),    # INT8
    2: ("H", 2),    # UINT16
    3: ("h", 2),    # INT16
    4: ("I", 4),    # UINT32
    5: ("i", 4),    # INT32
    6: ("f", 4),    # FLOAT32
    7: ("?", 1),    # BOOL
    10: ("Q", 8),   # UINT64
    11: ("q", 8),   # INT64
    12: ("d", 8),   # FLOAT64
}

MODELS_DIR = Path("E:/workspace/project/agents/ai-models")

MODELS = [
    ("CodeLlama-7b-Instruct-hf", "codellama-7b-instruct.Q4_K_M.gguf"),
    ("DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B.Q4_K_M.gguf"),
    ("DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
    ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
    ("Ministral-8B-Instruct-2410", "Ministral-8B-Instruct-2410-Q4_K_M.gguf"),
    ("Mistral-7B-Instruct-v0.2", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"),
    ("Mistral-Nemo-Instruct-2407", "Mistral-Nemo-Instruct-2407.Q4_K_M.gguf"),
    ("Nemotron-Cascade-8B-Thinking", "Nemotron-Cascade-8B-Thinking.Q4_K_M.gguf"),
    ("Nemotron-Mini-4B-Instruct", "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"),
    ("Phi-3-mini-128k-instruct", "Phi-3-mini-128k-instruct.Q4_K_M.gguf"),
    ("Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct-Q4_K_M.gguf"),
    ("Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
    ("Phi-4-mini-instruct", "Phi-4-mini-instruct-Q4_K_M.gguf"),
    ("Phi-4-mini-reasoning", "Phi-4-mini-reasoning-Q4_K_M.gguf"),
    ("Qwen2-0.5B-Instruct", "qwen2-0_5b-instruct-Q4_K_M.gguf"),
    ("Qwen2-1.5B-Instruct", "qwen2-1_5b-instruct-Q4_K_M.gguf"),
    ("Qwen2-7B-Instruct", "Qwen2-7B-Instruct.Q4_K_M.gguf"),
    ("Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"),
    ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-Coder-0.5B-Instruct", "Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-Coder-1.5B-Instruct", "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
    ("Qwen3-0.6B", "Qwen3-0.6B.Q4_K_M.gguf"),
    ("Qwen3-1.7B", "Qwen3-1.7B.Q4_K_M.gguf"),
    ("Qwen3-4B", "Qwen3-4B.Q4_K_M.gguf"),
    ("Qwen3-8B", "Qwen3-8B.Q4_K_M.gguf"),
    ("SOLAR-10.7B-Instruct-v1.0", "solar-10.7b-instruct-v1.0.Q4_K_M.gguf"),
    ("SmolLM-1.7B-Instruct", "SmolLM-1.7B-Instruct.Q4_K_M.gguf"),
    ("SmolLM-135M-Instruct", "SmolLM-135M-Instruct.Q4_K_M.gguf"),
    ("SmolLM-360M-Instruct", "SmolLM-360M-Instruct.Q4_K_M.gguf"),
    ("SmolLM2-1.7B-Instruct", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
    ("SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    ("SmolLM2-360M-Instruct", "SmolLM2-360M-Instruct-Q4_K_M.gguf"),
    ("TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ("Yi-1.5-6B-Chat", "Yi-1.5-6B-Chat.Q4_K_M.gguf"),
    ("Yi-1.5-9B-Chat", "Yi-1.5-9B-Chat-Q4_K_M.gguf"),
    ("Yi-Coder-1.5B-Chat", "Yi-Coder-1.5B-Chat.Q4_K_M.gguf"),
    ("gemma-2-2b-it", "gemma-2-2b-it-Q4_K_M.gguf"),
    ("gemma-2-9b-it", "gemma-2-9b-it.Q4_K_M.gguf"),
    ("gemma-2b-it", "gemma-2b-it-Q4_K_M.gguf"),
    ("gemma-3-1b-it", "gemma-3-1b-it.Q4_K_M.gguf"),
    ("gemma-7b-it", "gemma-7b-it.Q4_K_M-v2.gguf"),
    ("gpt-oss-20b", "gpt-oss-20b-Q4_K_M.gguf"),
    ("granite-3.1-2b-instruct", "granite-3.1-2b-instruct-Q4_K_M.gguf"),
    ("granite-3.1-8b-instruct", "granite-3.1-8b-instruct-Q4_K_M.gguf"),
    ("granite-3.2-2b-instruct", "granite-3.2-2b-instruct-Q4_K_M.gguf"),
    ("granite-3.2-8b-instruct", "granite-3.2-8b-instruct-Q4_K_M.gguf"),
    ("granite-3.3-2b-instruct", "granite-3.3-2b-instruct-Q4_K_M.gguf"),
    ("granite-3.3-8b-instruct", "granite-3.3-8b-instruct-Q4_K_M.gguf"),
    ("internlm2-chat-1_8b", "internlm2-chat-1_8b.Q4_K_M.gguf"),
    ("internlm2-chat-7b", "internlm2_5-7b-chat.i1-Q4_K_M.gguf"),
    ("internlm2_5-7b-chat", "internlm2_5-7b-chat.i1-Q4_K_M.gguf"),
]

KNOWN_ARCHS = {
    "llama", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3",
    "gpt-oss", "granite", "internlm2", "nemotron", "starcoder2",
}


def read_string(f):
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8")


def read_value(f, vtype):
    if vtype in GGUF_VALUE_READERS:
        fmt, size = GGUF_VALUE_READERS[vtype]
        return struct.unpack("<" + fmt, f.read(size))[0]
    if vtype == 8:  # STRING
        return read_string(f)
    if vtype == 9:  # ARRAY
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def probe_gguf(filepath):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}")

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata = {}
        for _ in range(metadata_kv_count):
            key = read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = read_value(f, vtype)
            metadata[key] = val

    arch = metadata.get("general.architecture", "<missing>")
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
        "architecture": arch,
        "recognized": arch in KNOWN_ARCHS,
    }


def main():
    passed = 0
    failed = 0
    results = []

    for i, (model_dir, gguf_file) in enumerate(MODELS, 1):
        filepath = MODELS_DIR / model_dir / "gguf" / gguf_file
        try:
            info = probe_gguf(filepath)
            status = "PASS" if info["recognized"] else "WARN"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            results.append((i, model_dir, status, info["architecture"],
                            info["tensor_count"], info["version"]))
            print(f"[{status}] {i:2d}/56  {model_dir:<40s}  "
                  f"arch={info['architecture']:<12s}  "
                  f"tensors={info['tensor_count']}  v{info['version']}")
        except Exception as e:
            failed += 1
            results.append((i, model_dir, "FAIL", str(e), 0, 0))
            print(f"[FAIL] {i:2d}/56  {model_dir:<40s}  error={e}")

    print(f"\n{'='*70}")
    print(f"Total: {len(MODELS)}  Passed: {passed}  Failed: {failed}")

    if failed > 0:
        print("\nFailed/warned models:")
        for i, model, status, arch, *_ in results:
            if status != "PASS":
                print(f"  {i:2d}. {model}: [{status}] {arch}")
        sys.exit(1)
    else:
        print("All 56 GGUF files parsed successfully with recognized architectures.")


if __name__ == "__main__":
    main()
