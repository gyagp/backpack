from __future__ import annotations

import io
import json
import math
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

from evolution.server import Handler


def strict_json(data: bytes | str):
    def reject_constant(value: str):
        raise ValueError(f"Non-JSON number: {value}")
    return json.loads(data, parse_constant=reject_constant)


class ServerJsonTest(unittest.TestCase):
    def http_response(self, payload):
        class TestHandler(Handler):
            def do_GET(self):
                self._send_json(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=2) as response:
                body = response.read()
                self.assertEqual("application/json; charset=utf-8", response.headers["Content-Type"])
                self.assertEqual(len(body), int(response.headers["Content-Length"]))
                return strict_json(body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    def test_http_zero_error_comparison_preserves_counts_and_verdict(self):
        payload = {"evaluations": [{"base_median": 1, "candidate_median": 0,
                    "delta_percent": math.inf, "verdict": "positive",
                    "details": {"separated_delta_percent": math.inf}}],
                   "other": [-math.inf, math.nan, (1.25, None, True)],
                   "text": "Infinity and NaN are valid strings; 测试"}
        result = self.http_response(payload)
        row = result["evaluations"][0]
        self.assertEqual((1, 0, "positive"), (row["base_median"], row["candidate_median"], row["verdict"]))
        self.assertIsNone(row["delta_percent"])
        self.assertIsNone(row["details"]["separated_delta_percent"])
        self.assertEqual([None, None, [1.25, None, True]], result["other"])
        self.assertEqual(payload["text"], result["text"])
        self.assertTrue(math.isinf(payload["evaluations"][0]["delta_percent"]))
        self.assertTrue(math.isnan(payload["other"][1]))
        self.assertIsInstance(payload["other"][2], tuple)

    def test_http_finite_values_are_unchanged(self):
        payload = {"int": 0, "float": -0.125, "nested": [False, None, 17.36],
                   "string": "NaN Infinity -Infinity", "empty": {}}
        self.assertEqual(payload, self.http_response(payload))

    def test_events_use_strict_json_and_leave_the_queued_value_unchanged(self):
        payload = {"verdict": "accept", "delta_percent": math.inf,
                   "details": [math.nan, {"count": 0}]}
        message = {"event": "task-evaluated", "data": payload}

        class Subscriber:
            def __init__(self):
                self.sent = False

            def get(self, timeout):
                if self.sent:
                    raise BrokenPipeError("Test client disconnected")
                self.sent = True
                return message

        subscriber = Subscriber()
        unsubscribed = []
        bus = SimpleNamespace(subscribe=lambda: subscriber, unsubscribe=unsubscribed.append)
        stream = io.BytesIO()
        handler = SimpleNamespace(server=SimpleNamespace(events=bus), wfile=stream,
                                  send_response=lambda *args: None,
                                  send_header=lambda *args: None, end_headers=lambda: None)
        Handler._events(handler)
        lines = stream.getvalue().splitlines()
        self.assertIn(b"event: task-evaluated", lines)
        values = [strict_json(line[6:]) for line in lines if line.startswith(b"data: ")]
        self.assertEqual([{}, {"verdict": "accept", "delta_percent": None,
                              "details": [None, {"count": 0}]}], values)
        self.assertEqual([subscriber], unsubscribed)
        self.assertTrue(math.isinf(payload["delta_percent"]))
        self.assertTrue(math.isnan(payload["details"][0]))


if __name__ == "__main__":
    unittest.main()
