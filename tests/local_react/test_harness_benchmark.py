"""Offline validation of benchmark boundaries and failure reporting."""
import importlib.util
import argparse
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("bench_agent_harness", Path(__file__).resolve().parents[2] / "scripts" / "bench_agent_harness.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(TestCase):
    def test_only_local_origins_accepted(self):
        self.assertEqual(bench.local_url("http://127.0.0.1:8000/"), "http://127.0.0.1:8000")
        for url in ("https://production.example", "http://localhost@production.example", "http://localhost:8000/path", "http://localhost:8000?key=private"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.local_url(url)

    def test_wrong_harness_or_model_prevents_comparison(self):
        settings = {"agent_mode": "local_react", "llm_model_fast": "fast", "llm_model_strong": "strong"}
        values = {"legacy": {**settings, "local_react_harness": "legacy"},
                  "langchain": {**settings, "local_react_harness": "langchain"}}
        bench.check_health(values)
        values["langchain"]["llm_model_fast"] = "different"
        with self.assertRaises(ValueError):
            bench.check_health(values)

    def test_errors_not_dropped_and_small_sample_p95_suppressed(self):
        rows = [{"harness": "legacy", "final_receipt_ms": 100, "first_answer_delta_ms": 50, "checks": {"ok": True}},
                {"harness": "legacy", "error_type": "ValueError"}]
        result = bench.summarize(rows)["legacy"]
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(result["p95_final_receipt_ms"])
        self.assertEqual(result["median_final_receipt_ms"], 100)

    def test_stream_requires_final_and_uses_correlated_request(self):
        class Response:
            headers = {"Content-Type": "text/event-stream", "x-ci-request-id": "request-fixture"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                return iter([b'data: {"type":"stage"}\n', b'data: {"type":"token","delta":"answer"}\n',
                             b'data: {"type":"final","session_id":"session","response":{"answer_markdown":"answer"}}\n'])
        with patch.object(bench, "urlopen", return_value=Response()):
            record = bench.measure("http://localhost:8000", "question", None, [])
        self.assertEqual(record["request_id"], "request-fixture")
        self.assertIsNotNone(record["first_progress_ms"])
        self.assertIsNotNone(record["first_answer_delta_ms"])
        self.assertGreaterEqual(record["stream_end_ms"], record["final_receipt_ms"])
