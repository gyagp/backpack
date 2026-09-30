"""Check mixed IQ decoding against llama.cpp's independent gguf Python package.

Run with a Python environment containing gguf and numpy. All fixtures and the
test executable live below gitignore/; no GPU is required for this format test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess

import gguf
import numpy as np


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=root / "gitignore/runtime/build/backpack_gguf_test.exe")
    parser.add_argument("--model", type=Path, help="Also sample every quantization in a real GGUF")
    args = parser.parse_args()
    output = root / "gitignore/runtime/tests/gguf-reference"
    output.mkdir(parents=True, exist_ok=True)
    fixtures: list[str] = []

    def fixture(name: str, raw: np.ndarray, quant_type: gguf.GGMLQuantizationType) -> None:
        values = gguf.dequantize(raw, quant_type).astype("<f4")
        path = output / (name + ".bin")
        with path.open("wb") as stream:
            stream.write(struct.pack("<4I", int(quant_type), values.shape[0], values.shape[1], raw.nbytes))
            stream.write(raw.tobytes())
            stream.write(values.tobytes())
        fixtures.append(str(path))

    rng = np.random.default_rng(104)
    for quant_type in (gguf.GGMLQuantizationType.IQ2_XXS,
                       gguf.GGMLQuantizationType.IQ2_XS,
                       gguf.GGMLQuantizationType.IQ1_S):
        _, block_bytes = gguf.GGML_QUANT_SIZES[quant_type]
        raw = rng.integers(0, 256, (513, block_bytes), dtype=np.uint8)
        # Include zero, negative, normal, and small scales while avoiding NaN.
        scales = rng.uniform(-2, 2, 513).astype("<f2")
        scales[:4] = [0, -0.5, 1, 2**-14]
        raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
        fixture("random-" + quant_type.name, raw, quant_type)

    if args.model:
        model = gguf.GGUFReader(str(args.model))
        seen = set()
        for tensor in model.tensors:
            kind = tensor.tensor_type
            if kind in seen or kind in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                continue
            seen.add(kind)
            _, block_bytes = gguf.GGML_QUANT_SIZES[kind]
            blocks = tensor.data.view(np.uint8).reshape(-1, block_bytes)
            indices = np.linspace(0, len(blocks) - 1, min(128, len(blocks)), dtype=np.int64)
            fixture("model-" + kind.name, np.ascontiguousarray(blocks[indices]), kind)

    return subprocess.call([str(args.executable), *fixtures])


if __name__ == "__main__":
    raise SystemExit(main())
