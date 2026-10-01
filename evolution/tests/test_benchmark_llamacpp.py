from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evolution.benchmark_llamacpp import conformance_passed, final_answer, main


class LlamaBenchmarkTest(unittest.TestCase):
    def test_gemma_reasoning_does_not_satisfy_final_answer_conformance(self):
        output = "<|channel>thought\nRecall Paris.<channel|>The capital is Lyon. [end of text]"
        answer = final_answer(output)
        self.assertEqual("The capital is Lyon.", answer)
        self.assertFalse(conformance_passed(answer, "Paris"))
        self.assertEqual("Paris.", final_answer("<|channel>thought\nReasoning.<channel|>Paris. [end of text]"))

    def test_unfinished_gemma_reasoning_has_no_final_answer(self):
        self.assertEqual("", final_answer("<|channel>thought\nThe answer might be Paris"))

    def run_adapter(self, depth: int):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "b123/vulkan"
            binary.mkdir(parents=True)
            for name in ("llama-bench.exe", "llama-completion.exe"):
                (binary / name).touch()
            model = root / "model.gguf"
            model.touch()
            output = io.StringIO()
            prefill = {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 1200, "samples_ts": [1200]}
            decode = {"n_prompt": 0, "n_gen": 128, "n_depth": depth, "avg_ts": 45, "samples_ts": [45]}
            responses = [subprocess.CompletedProcess([], 0, "4 [end of text]", ""),
                         subprocess.CompletedProcess([], 0, json.dumps([prefill]), ""),
                         subprocess.CompletedProcess([], 0, json.dumps([decode]), "")]
            argv = ["benchmark", "--root", str(root), "--model", str(model),
                    "--prompt", "What is 2 + 2?", "--required-fact", "4",
                    "--expected-output", "4", "--prompt-format", "qwen-nothink"]
            with patch("sys.argv", argv), patch("subprocess.run", side_effect=responses) as run, \
                    contextlib.redirect_stdout(output):
                result = main()
            return result, output.getvalue(), [call.args[0] for call in run.call_args_list]

    def test_decode_has_prefilled_context_and_qwen_uses_non_thinking_template(self):
        code, stdout, commands = self.run_adapter(512)
        self.assertEqual(0, code)
        chat, prefill, decode = commands
        self.assertIn("--no-conversation", chat)
        self.assertTrue(chat[chat.index("-p") + 1].endswith("<think>\n\n</think>\n\n"))
        self.assertEqual("0", prefill[prefill.index("-d") + 1])
        self.assertEqual("512", decode[decode.index("-d") + 1])
        self.assertEqual("0", decode[decode.index("-p") + 1])
        metrics = json.loads(next(line.split(" ", 1)[1] for line in stdout.splitlines()
                                  if line.startswith("EVOLUTION_METRICS ")))
        self.assertEqual(512, metrics["decode_context_tokens"])
        self.assertEqual([45], metrics["decode_samples_tok_s"])
        repetitions = int(prefill[prefill.index("-r") + 1])
        self.assertEqual((1, repetitions, repetitions), (metrics["measured_processes"],
            metrics["measured_repetitions_per_process"], metrics["measured_repetitions"]))

    def test_empty_context_result_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "populated-context"):
            self.run_adapter(0)


if __name__ == "__main__":
    unittest.main()
