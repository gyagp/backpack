import unittest
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from evolution.benchmark_ort import chat_answer, parse_benchmark, parse_comparable_benchmark, find_bin_dir, revision, graph_capture_enabled, run_conformance, prepare_native_model, benchmark_token_counts, require_warm_rewind_support, validate_reference_view, qwen_static_capture_environment


class OrtBenchmarkParserTest(unittest.TestCase):
    def test_comparable_protocol_counts_calls_and_includes_first_sampling(self):
        output = """Prompt processing (time to first token):
 avg (us): 100000
 n: 5 * 512 token(s)
Token generation:
 avg (us): 10000
 n: 635 * 1 token(s)
Token sampling:
 avg (us): 500
 n: 5 * 1 token(s)
"""
        result = parse_comparable_benchmark(output, 512, 128, 5)
        self.assertAlmostEqual(512000 / 100.5, result["prefill_tok_s"])
        self.assertEqual(100, result["decode_tok_s"])
        self.assertEqual(0.5, result["first_token_sample_ms"])
        self.assertEqual((1, 5, 5), (result["measured_processes"],
            result["measured_repetitions_per_process"], result["measured_repetitions"]))
        with self.assertRaisesRegex(RuntimeError, "counts differ"):
            parse_comparable_benchmark(output.replace("635 *", "640 *"), 512, 128, 5)
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            parse_comparable_benchmark(output.split("Token sampling:")[0], 512, 128, 5)

    def test_extracts_only_generated_chat_answer(self) -> None:
        output = "header\nOutput: 4\n-------------\nPrompt length: 10"
        self.assertEqual("4", chat_answer(output))

    def test_strips_qwen_reasoning_before_final_answer(self) -> None:
        output = ("header\nOutput: Thinking Process:\n2 + 2 is 4.cw\n</think>\n\n4\n"
                  "-------------\nPrompt length: 10")
        self.assertEqual("4", chat_answer(output))

    def test_uses_text_after_last_think_block(self) -> None:
        output = "Output: stale</think>draft</think> final answer\n-------------"
        self.assertEqual("final answer", chat_answer(output))

    def test_parses_separate_prefill_and_decode_rates(self) -> None:
        output = """Prompt processing (time to first token):
 avg (us): 177610
 avg (tokens/s): 720.682
Token generation:
 avg (us): 8409.5
 avg (tokens/s): 118.913
"""
        self.assertEqual((720.682, 118.913), parse_benchmark(output))

    def test_rejects_incomplete_output(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_benchmark("Token generation: avg (tokens/s): 12")


class OrtReferenceArtifactTest(unittest.TestCase):
    def test_native_view_cannot_replace_weights_graph_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, view = Path(tmp) / "source", Path(tmp) / "view"
            source.mkdir(); view.mkdir()
            for name in ["text.onnx", "text.onnx.data", "tokenizer.json"]:
                (source / name).write_bytes(b"original")
                (view / name).write_bytes(b"original")
            (source / "genai_config.json").write_text("original configuration")
            (view / "genai_config.json").write_text("native backend configuration")
            validate_reference_view(source, view)
            for name in ["text.onnx", "text.onnx.data", "tokenizer.json"]:
                (view / name).write_bytes(b"different")
                with self.assertRaisesRegex(RuntimeError, "changed original artifact"):
                    validate_reference_view(source, view)
                (view / name).write_bytes(b"original")
            (view / "text.onnx.data").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing original artifact"):
                validate_reference_view(source, view)

    def test_static_capture_requires_capability_capacity_and_exact_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"model": {"type": "qwen3_5", "vocab_size": 248320, "context_length": 640,
                "decoder": {"num_hidden_layers": 64, "hidden_size": 5120,
                    "session_options": {"provider_options": [{"webgpu": {"enableGraphCapture": "1"}}]}}},
                "search": {"max_length": 640, "chunk_size": 32}}
            (root / "genai_config.json").write_text(json.dumps(config))
            artifacts = {}
            for name in ["onnxruntime.dll", "onnxruntime-genai.dll", "genai_state_reference.exe", "model_benchmark.exe", "dxcompiler.dll"]:
                (root / name).write_bytes(name.encode())
                artifacts[name] = hashlib.sha256(name.encode()).hexdigest()
            metadata = {"capabilities": ["qwen38_static_capture_b1_c640"], "artifacts": artifacts}
            self.assertEqual({"ORTGENAI_QWEN_STATIC_CAPTURE": "1"},
                qwen_static_capture_environment(root, root, metadata, 512, 128))
            published = {"capabilities": metadata["capabilities"],
                         "artifacts": list(artifacts), "artifact_hashes": artifacts}
            self.assertEqual({"ORTGENAI_QWEN_STATIC_CAPTURE": "1"},
                qwen_static_capture_environment(root, root, published, 512, 128))
            with self.assertRaisesRegex(RuntimeError, "artifact hash mapping"):
                qwen_static_capture_environment(root, root,
                    {"capabilities": metadata["capabilities"], "artifacts": list(artifacts)}, 512, 128)
            with self.assertRaisesRegex(RuntimeError, "capacity640"):
                qwen_static_capture_environment(root, root, metadata, 1024, 128)
            with self.assertRaisesRegex(RuntimeError, "does not declare"):
                qwen_static_capture_environment(root, root, {}, 512, 128)
            config["search"]["chunk_size"] = 64
            (root / "genai_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "validated batch1"):
                qwen_static_capture_environment(root, root, metadata, 512, 128)
            config["search"]["chunk_size"] = 32
            (root / "genai_config.json").write_text(json.dumps(config))
            (root / "onnxruntime-genai.dll").write_bytes(b"unvalidated build")
            with self.assertRaisesRegex(RuntimeError, "artifact hash differs"):
                qwen_static_capture_environment(root, root, metadata, 512, 128)
            with self.assertRaisesRegex(RuntimeError, "artifact hash differs"):
                qwen_static_capture_environment(root, root, published, 512, 128)

    def test_static_conformance_uses_bounded_capacity_and_checks_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "genai_state_reference.exe").touch()
            environment = {"ORTGENAI_QWEN_STATIC_CAPTURE": "1"}
            def execute(command, cwd, timeout, settings):
                request = json.loads(Path(command[2]).read_text())
                self.assertEqual((640, 1, 2), (request["max_seq_len"], request["warmup_runs"], request["repetitions"]))
                self.assertEqual(environment, settings)
                Path(command[3]).write_text(json.dumps({"text": "4", "tokens": [19, 248044],
                    "graph_capture_requested": True, "warmup_runs": 1,
                    "runs": [{"tokens": [19, 248044]}, {"tokens": [2523]}]}))
                return "native log"
            with patch("evolution.benchmark_ort.run", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "across resets"):
                    run_conformance(root, root / "model", "What is 2+2?", True, root / "evidence",
                                    capacity=640, environment=environment)

    def test_known_multimodal_rewind_failure_cannot_publish_warm_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            metadata = {"onnxruntime_genai_revision": "66a5cacc86171d866e2125846521a32e17c7efce"}
            for model_type in ["gemma4", "qwen3_5"]:
                (model / "genai_config.json").write_text(json.dumps({"model": {"type": model_type}}))
                with self.assertRaisesRegex(RuntimeError, "rewind is not conformant"):
                    require_warm_rewind_support(model, metadata)
            # Qwen 2B/4B use DecoderOnly_State, which supplies its own rewind.
            (model / "genai_config.json").write_text(json.dumps({"model": {"type": "qwen3_5_text"}}))
            require_warm_rewind_support(model, metadata)

    def test_fresh_metadata_and_native_reference_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "old", root / "new"
            old.mkdir(); new.mkdir()
            for directory in [old, new]:
                (directory / "model_benchmark.exe").touch()
            (old / "model_chat.exe").touch()
            (old / "build-manifest.json").write_text(json.dumps({
                "date": "2026-09-29T16:00:00Z", "onnxruntime_revision": "old", "onnxruntime_genai_revision": "old"}))
            (new / "genai_state_reference.exe").touch()
            (new / "build-metadata.json").write_text(json.dumps({
                "builtAt": "2026-09-30T01:21:56+08:00", "repositories": {
                    "onnxruntime": {"commit": "96f73115c95968a3f31f2a110b33c164d847dda4"},
                    "onnxruntime-genai": {"commit": "66a5cacc86171d866e2125846521a32e17c7efce"}}}))
            self.assertEqual(new, find_bin_dir(root))
            self.assertEqual("ort-96f73115c9-genai-66a5cacc86-20260930", revision(new))

    def test_capture_setting_belongs_to_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "genai_config.json").write_text(json.dumps({"model": {
                "vision": {"session_options": {"provider_options": [{"webgpu": {"enableGraphCapture": "0"}}]}},
                "decoder": {"session_options": {"provider_options": [{"webgpu": {"enableGraphCapture": "1"}}]}}}}))
            self.assertIs(graph_capture_enabled(model), True)

    def test_structured_reference_preserves_request_and_capture_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "genai_state_reference.exe").touch()
            def execute(command, cwd, timeout):
                request = json.loads(Path(command[2]).read_text())
                self.assertEqual('Answer "4" exactly.', request["prompt"])
                self.assertIs(request["graph_capture"], False)
                self.assertEqual(root, cwd)
                self.assertEqual((2, 1), (request["repetitions"], request["warmup_runs"]))
                Path(command[3]).write_text(json.dumps({"text": "4", "graph_capture_requested": False,
                    "warmup_runs": 1, "tokens": [19], "runs": [{"tokens": [19]}, {"tokens": [19]}]}))
                return "reference log"
            with patch("evolution.benchmark_ort.run", side_effect=execute):
                answer, command = run_conformance(root, root / "model", 'Answer "4" exactly.', False, root / "evidence")
            self.assertEqual("4", answer)
            self.assertTrue((Path(command[2]).parent / "run.log").is_file())

    def test_native_model_view_preserves_artifacts_and_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; source.mkdir()
            config = {"model": {
                "decoder": {"session_options": {"provider_options": [{"webgpu": {"enableGraphCapture": "1", "adapterIndex": "0"}}]}},
                "embedding": {"session_options": {"provider_options": [{"webgpu": {"enableGraphCapture": "0"}}]}}}}
            original = json.dumps(config)
            (source / "genai_config.json").write_text(original)
            (source / "model.onnx").write_bytes(b"original model")
            view = prepare_native_model(source, root / "evidence")
            self.assertEqual(original, (source / "genai_config.json").read_text())
            self.assertEqual(b"original model", (view / "model.onnx").read_bytes())
            prepared = json.loads((view / "genai_config.json").read_text())["model"]
            for name, capture in [("decoder", "1"), ("embedding", "0")]:
                opts = prepared[name]["session_options"]["provider_options"][0]["webgpu"]
                self.assertEqual("D3D12", opts["dawnBackendType"])
                self.assertEqual(capture, opts["enableGraphCapture"])
                self.assertNotIn("adapterIndex", opts)
            self.assertIs(graph_capture_enabled(view), True)

    def test_benchmark_requires_reported_counts(self):
        self.assertEqual((512, 128), benchmark_token_counts("Batch size: 1, prompt tokens: 512, tokens to generate: 128"))
        with self.assertRaisesRegex(RuntimeError, "actual token counts"):
            benchmark_token_counts("Prompt processing: avg (tokens/s): 100")

    def test_capture_mismatch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "genai_state_reference.exe").touch()
            def execute(command, cwd, timeout):
                Path(command[3]).write_text(json.dumps({"text": "4", "graph_capture_requested": False}))
                return ""
            with patch("evolution.benchmark_ort.run", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "different graph-capture"):
                    run_conformance(root, root / "model", "Question", True, root / "evidence")


if __name__ == "__main__":
    unittest.main()
