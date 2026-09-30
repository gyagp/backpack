"""Compare GGUF token IDs with a pinned local llama.cpp tokenizer executable."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import subprocess


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--executable", type=Path, default=root / "gitignore/runtime/build/backpack_tokenizer_test.exe")
    args = parser.parse_args()
    out = root / "gitignore/runtime/tests/tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    texts = ["Hello, world!", "1234567890  3.1415926", "We're ready. I'LL test contractions.",
             "你好，世界！🌏🚀", "  whitespace\n\n\tend\r\n", "café naïve привет", "a1b2 42x \n 123456",
             "<|im_start|>user\n你好  123<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"]
    rng = random.Random(104)
    pieces = ["a", "xy", "é", "e\u0301", "你好", "Ⅳ", "١", "123", " ", "  ", "\t", "\n", "\r\n",
              "🚀", "!", "?", "'S", "'ve", "ไทย", "क", "ि", "\U00010400", "\u00a0", "\u2028"]
    texts += ["".join(rng.choices(pieces, k=20)) for _ in range(32)]
    results = []
    for index, text in enumerate(texts):
        path = out / f"prompt-{index}.txt"
        path.write_bytes(text.encode("utf-8"))
        commands = [[str(args.executable.resolve()), str(args.model.resolve()), str(path)],
                    [str(args.reference.resolve()), "-m", str(args.model.resolve()), "-f", str(path), "--ids", "--no-bos"]]
        ids = []
        for command in commands:
            result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=30, check=True)
            ids.append(json.loads(result.stdout))
        results.append({"text": text, "backpack": ids[0], "reference": ids[1], "pass": ids[0] == ids[1]})
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [index for index, item in enumerate(results) if not item["pass"]]
    print(f"{len(results)-len(failed)}/{len(results)} tokenizer comparisons passed; failures: {failed}")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
