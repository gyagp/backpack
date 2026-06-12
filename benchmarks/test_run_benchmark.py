#!/usr/bin/env python3
"""Tests for the benchmark orchestrator (run_benchmark.py)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmark


class TestGenerateMarkdown(unittest.TestCase):
    def test_basic_table(self):
        results = [
            {"model": "Qwen3-1.7B", "quant": "Q4_K_M", "engine": "backpack",
             "prefill_tok_s": 100.5, "decode_tok_s": 50.3},
            {"model": "Qwen3-1.7B", "quant": "Q4_K_M", "engine": "llamacpp",
             "prefill_tok_s": 120.0, "decode_tok_s": 60.0},
        ]
        md = run_benchmark.generate_markdown(results)
        self.assertIn("| Qwen3-1.7B | Q4_K_M | backpack | 100.50 | 50.30 |", md)
        self.assertIn("| Qwen3-1.7B | Q4_K_M | llamacpp | 120.00 | 60.00 |", md)
        self.assertIn("Prefill tok/s", md)
        self.assertIn("Decode tok/s", md)

    def test_empty_results(self):
        md = run_benchmark.generate_markdown([])
        self.assertIn("| Model |", md)
        lines = [l for l in md.strip().splitlines() if l.startswith("|")]
        self.assertEqual(len(lines), 2)  # header + separator

    def test_na_values(self):
        results = [{"model": "X", "quant": "Q4", "engine": "e"}]
        md = run_benchmark.generate_markdown(results)
        self.assertIn("N/A", md)

    def test_missing_fields_use_unknown(self):
        results = [{"prefill_tok_s": 10.0, "decode_tok_s": 20.0}]
        md = run_benchmark.generate_markdown(results)
        self.assertIn("unknown", md)


class TestRunBackpack(unittest.TestCase):
    @mock.patch("run_benchmark.subprocess.run")
    def test_success(self, mock_run):
        output = json.dumps({"prefill_tok_s": 100, "decode_tok_s": 50,
                             "model": "test", "quant": "Q4_K_M"})
        mock_run.return_value = mock.Mock(returncode=0, stdout=output, stderr="")
        result = run_benchmark.run_backpack("model.gguf", "bench_backpack")
        self.assertEqual(result["engine"], "backpack")
        self.assertEqual(result["prefill_tok_s"], 100)

    @mock.patch("run_benchmark.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error")
        result = run_benchmark.run_backpack("model.gguf", "bench_backpack")
        self.assertIsNone(result)

    @mock.patch("run_benchmark.subprocess.run", side_effect=FileNotFoundError)
    def test_not_found(self, mock_run):
        result = run_benchmark.run_backpack("model.gguf", "bench_backpack")
        self.assertIsNone(result)

    @mock.patch("run_benchmark.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 600))
    def test_timeout(self, mock_run):
        result = run_benchmark.run_backpack("model.gguf", "bench_backpack")
        self.assertIsNone(result)


class TestRunLlamacpp(unittest.TestCase):
    @mock.patch("run_benchmark.subprocess.run")
    def test_success(self, mock_run):
        output = json.dumps({"prefill_tok_s": 120, "decode_tok_s": 60,
                             "model": "test", "quant": "Q4_K_M", "engine": "llamacpp"})
        mock_run.return_value = mock.Mock(returncode=0, stdout=output, stderr="")
        result = run_benchmark.run_llamacpp("model.gguf")
        self.assertEqual(result["engine"], "llamacpp")

    @mock.patch("run_benchmark.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="err")
        result = run_benchmark.run_llamacpp("model.gguf")
        self.assertIsNone(result)


class TestRunOrtWebgpu(unittest.TestCase):
    @mock.patch("run_benchmark.subprocess.run")
    def test_success(self, mock_run):
        output = json.dumps({"prefill_tok_s": 90, "decode_tok_s": 45,
                             "model": "test", "quant": "Q4_K_M", "engine": "ort-webgpu"})
        mock_run.return_value = mock.Mock(returncode=0, stdout=output, stderr="")
        result = run_benchmark.run_ort_webgpu("model.onnx")
        self.assertEqual(result["engine"], "ort-webgpu")
        self.assertEqual(result["prefill_tok_s"], 90)

    @mock.patch("run_benchmark.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="err")
        result = run_benchmark.run_ort_webgpu("model.onnx")
        self.assertIsNone(result)

    @mock.patch("run_benchmark.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 600))
    def test_timeout(self, mock_run):
        result = run_benchmark.run_ort_webgpu("model.onnx")
        self.assertIsNone(result)


class TestConfigLoading(unittest.TestCase):
    def test_config_file_format(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "benchmark_config.json")
        with open(config_path) as f:
            config = json.load(f)
        self.assertIn("models", config)
        self.assertIsInstance(config["models"], list)
        self.assertEqual(len(config["models"]), 3)
        models_str = " ".join(config["models"]).lower()
        self.assertIn("qwen3", models_str)
        self.assertIn("phi-4", models_str)
        self.assertIn("llama-3.2", models_str)


class TestMainIntegration(unittest.TestCase):
    """Test main() with mocked subprocess calls to verify end-to-end flow."""

    def test_generates_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "test.gguf")
            with open(model_file, "w") as f:
                f.write("fake")

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"models": [model_file]}, f)

            bp_out = json.dumps({"model": "test", "quant": "Q4_K_M",
                                 "prefill_tok_s": 100, "decode_tok_s": 50})
            lc_out = json.dumps({"model": "test", "quant": "Q4_K_M",
                                 "engine": "llamacpp",
                                 "prefill_tok_s": 120, "decode_tok_s": 60})
            ort_out = json.dumps({"model": "test", "quant": "Q4_K_M",
                                  "engine": "ort-webgpu",
                                  "prefill_tok_s": 90, "decode_tok_s": 45})

            def fake_run(cmd, **kwargs):
                cmd_str = str(cmd)
                if "bench_backpack" in cmd_str:
                    return mock.Mock(returncode=0, stdout=bp_out, stderr="")
                if "run_ort_webgpu" in cmd_str:
                    return mock.Mock(returncode=0, stdout=ort_out, stderr="")
                return mock.Mock(returncode=0, stdout=lc_out, stderr="")

            with mock.patch("run_benchmark.subprocess.run", side_effect=fake_run), \
                 mock.patch("run_benchmark.find_bench_backpack", return_value="bench_backpack"), \
                 mock.patch("sys.argv", ["run_benchmark.py", "--config", config_path,
                                         "--output-dir", tmpdir]):
                run_benchmark.main()

            json_path = os.path.join(tmpdir, "results.json")
            md_path = os.path.join(tmpdir, "results.md")
            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(md_path))

            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 3)

            with open(md_path) as f:
                md = f.read()
            self.assertIn("backpack", md)
            self.assertIn("llamacpp", md)
            self.assertIn("ort-webgpu", md)
            self.assertIn("100.00", md)

    def test_no_models_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "empty.json")
            with open(config_path, "w") as f:
                json.dump({"models": []}, f)
            with mock.patch("sys.argv", ["run_benchmark.py", "--config", config_path]):
                with self.assertRaises(SystemExit) as ctx:
                    run_benchmark.main()
                self.assertEqual(ctx.exception.code, 1)


class TestAcceptanceCriteria(unittest.TestCase):
    """Verify acceptance criteria are met."""

    def test_file_exists(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_benchmark.py")
        self.assertTrue(os.path.isfile(path))

    def test_config_has_target_models(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "benchmark_config.json")
        with open(config_path) as f:
            config = json.load(f)
        models = config["models"]
        names = " ".join(models).lower()
        self.assertIn("qwen3-1.7b", names)
        self.assertIn("phi-4-mini", names)
        self.assertIn("llama-3.2-1b", names)

    def test_script_calls_both_engines(self):
        import inspect
        src = inspect.getsource(run_benchmark.main)
        self.assertIn("run_backpack", src)
        self.assertIn("run_llamacpp", src)
        self.assertIn("run_ort_webgpu", src)

    def test_generates_json_and_md(self):
        import inspect
        src = inspect.getsource(run_benchmark.main)
        self.assertIn("results.json", src)
        self.assertIn("results.md", src)

    def test_markdown_has_required_columns(self):
        results = [{"model": "M", "quant": "Q", "engine": "E",
                     "prefill_tok_s": 1.0, "decode_tok_s": 2.0}]
        md = run_benchmark.generate_markdown(results)
        self.assertIn("Model", md)
        self.assertIn("Quant", md)
        self.assertIn("Engine", md)
        self.assertIn("Prefill tok/s", md)
        self.assertIn("Decode tok/s", md)


if __name__ == "__main__":
    unittest.main()
