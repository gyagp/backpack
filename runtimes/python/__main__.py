"""Entry point for: python -m runtimes.python

Auto-detects model format from the --model path:
  - Directory with model.onnx → ONNX runtime
  - .gguf file or directory with .gguf files → GGUF runtime
"""
import os
import sys


def main():
    # Quick-scan argv for --model to detect format before full arg parsing
    model_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            break
        if arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]
            break

    use_onnx = False
    if model_path:
        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "model.onnx")):
                use_onnx = True
        # Also check if it's a parent dir with an onnx subdirectory
        elif not model_path.endswith('.gguf'):
            # Try as model name in repo
            from model_parser.resolver import DEFAULT_MODEL_REPO
            repo_path = os.path.join(DEFAULT_MODEL_REPO, model_path)
            if os.path.isdir(repo_path):
                if os.path.exists(os.path.join(repo_path, "model.onnx")):
                    use_onnx = True

    if use_onnx:
        from runtimes.python.onnx_runtime import main as onnx_main
        onnx_main()
    else:
        from runtimes.python.gguf_runtime import main as gguf_main
        gguf_main()


main()
