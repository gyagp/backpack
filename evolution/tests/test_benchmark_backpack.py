from __future__ import annotations

import unittest

from evolution.benchmark_backpack import validate_results


class ExactBenchmarkTest(unittest.TestCase):
    def fixture(self):
        return {
            "benchmark_protocol": "llm-fixed-text-reuse-first-token-v2",
            "warmup_runs": 1, "reuse_generator": True, "prefill_includes_first_token": True,
            "max_seq_len": 5, "backend": "d3d12",
            "runs": [{"repetition": i, "prompt": "A A", "input_tokens": 2, "generated_tokens": 3,
                      "decode_sample_tokens": 2, "final_position": 4,
                      "prompt_token_ids": [32, 357], "generated_token_ids": [7, 8, 9],
                      "prefill_ms": 20, "decode_ms": 8} for i in range(2)],
        }

    def validate(self, data):
        return validate_results(data, 2, 3, 2, 1, 5, "A A")

    def test_rates_use_actual_measured_counts(self):
        rates = self.validate(self.fixture())
        self.assertEqual(100, rates["prefill_tok_s"])
        self.assertEqual(250, rates["decode_tok_s"])

    def test_hidden_warmup_and_extra_decode_are_rejected(self):
        for key, value in [("decode_sample_tokens", 3), ("final_position", 5),
                           ("generated_tokens", 4), ("input_tokens", 3)]:
            data = self.fixture()
            data["runs"][0][key] = value
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                self.validate(data)

    def test_repetition_drift_and_nonfinite_time_are_rejected(self):
        data = self.fixture()
        data["runs"][1]["generated_token_ids"] = [7, 8, 10]
        with self.assertRaisesRegex(RuntimeError, "deterministic"):
            self.validate(data)
        data = self.fixture()
        data["runs"][0]["decode_ms"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "duration"):
            self.validate(data)


if __name__ == "__main__":
    unittest.main()
