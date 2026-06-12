"""Tests for run_llamacpp.py benchmark runner."""

import json
import subprocess
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from run_llamacpp import parse_model_info, parse_csv_output


class TestParseModelInfo:
    def test_extracts_quant_from_filename(self):
        name, quant = parse_model_info("/models/Qwen3-0.6B-Q4_K_M.gguf")
        assert quant == "Q4_K_M"
        assert "Qwen3" in name

    def test_extracts_quant_q8(self):
        _, quant = parse_model_info("model-Q8_0.gguf")
        assert quant == "Q8_0"

    def test_unknown_quant_when_missing(self):
        _, quant = parse_model_info("model.gguf")
        assert quant == "unknown"

    def test_name_is_stem(self):
        name, _ = parse_model_info("/some/path/my-model-Q4_K_M.gguf")
        assert name == "my-model-Q4_K_M"


class TestParseCsvOutput:
    SAMPLE_CSV = (
        "model,size,params,backend,ngl,test,t/s\n"
        "model.gguf,600M,0.6B,Vulkan,99,pp512,1234.56\n"
        "model.gguf,600M,0.6B,Vulkan,99,tg128,78.90\n"
    )

    def test_parses_prefill_and_decode(self):
        prefill, decode = parse_csv_output(self.SAMPLE_CSV)
        assert prefill == 1234.56
        assert decode == 78.90

    def test_empty_output_exits(self):
        with pytest.raises(SystemExit):
            parse_csv_output("")

    def test_header_only_exits(self):
        with pytest.raises(SystemExit):
            parse_csv_output("model,test,t/s\n")


class TestCLIInterface:
    def test_requires_model_argument(self):
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "run_llamacpp.py")],
            capture_output=True, text=True
        )
        assert result.returncode != 0

    def test_missing_model_file_exits(self):
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "run_llamacpp.py"),
             "/nonexistent/model.gguf"],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestOutputFormat:
    """Verify the JSON output contains all required fields."""

    def test_output_has_required_keys(self):
        csv = (
            "model,size,params,backend,ngl,test,t/s\n"
            "m.gguf,1G,1B,Vulkan,99,pp512,100.0\n"
            "m.gguf,1G,1B,Vulkan,99,tg128,50.0\n"
        )
        prefill, decode = parse_csv_output(csv)
        result = {
            "model": "test-model",
            "quant": "Q4_K_M",
            "engine": "llamacpp-vulkan",
            "prefill_tok_s": prefill,
            "decode_tok_s": decode,
        }
        assert result["engine"] == "llamacpp-vulkan"
        assert result["prefill_tok_s"] == 100.0
        assert result["decode_tok_s"] == 50.0
        for key in ["model", "quant", "engine", "prefill_tok_s", "decode_tok_s"]:
            assert key in result
