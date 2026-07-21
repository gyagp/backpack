import unittest

from evolution.benchmark_ort import parse_benchmark


class OrtBenchmarkParserTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
