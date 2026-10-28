"""Exercise the real FastAPI stack with in-process mocked chat/history."""
from __future__ import annotations

import importlib
import os
import tempfile
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx


class FakeHistory:
    def create_session(self, *args, **kwargs):
        return {"session_id": "fixture-session"}

    def get_session(self, *args, **kwargs):
        return {"session_id": "fixture-session"}

    def append_turn(self, *, assistant_payload, **kwargs):
        return {"message_id": "fixture-message", "content_markdown": assistant_payload["answer_markdown"]}


class APIObservationTests(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {
            "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
            "CI_AGENT_MODE": "local_react", "CI_LOCAL_REACT_HARNESS": "legacy",
            "CI_HISTORY_BACKEND": "local", "CI_HISTORY_DIR": cls.temp.name,
            "CI_FILES_DIR": cls.temp.name, "CI_LOCAL_USER_EMAIL": "fixture@example.invalid",
        })
        cls.env.start()
        cls.main = importlib.import_module("app.main")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.temp.cleanup()

    async def test_actual_middleware_order_and_stream_contract(self):
        self.assertIs(self.main.app.user_middleware[0].cls, self.main.RequestObservabilityMiddleware)
        closed = []

        def agent(**kwargs):
            try:
                yield {"type": "stage", "stage": "route", "status": "start", "label": "Understanding question"}
                yield {"type": "token", "delta": "Verified answer"}
                yield {"type": "final", "response": {"answer_markdown": "Verified answer", "timing": {"total_ms": 3}}}
            finally:
                closed.append(True)

        with patch.object(self.main, "history_store", FakeHistory()), \
                patch.object(self.main, "invoke_agent_stream", agent), \
                patch("requests.sessions.Session.request", side_effect=AssertionError("No network in offline API test")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.main.app), base_url="http://testserver") as client:
                response = await client.post("/api/chat/stream", json={"message": "fixture analytical question", "client_history": []})
        self.assertEqual(response.status_code, 200)
        import json
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        payload = next(event for event in events if event["type"] == "final")["response"]
        self.assertEqual(payload["request_id"], response.headers["x-ci-request-id"])
        self.assertEqual(payload["timing"]["total_ms"], 3)
        stages = {item["name"] for item in payload["timing"]["api"]["intervals"]}
        self.assertTrue({"identity", "history.get_session", "history.enqueue_write", "citations", "exports"} <= stages)
        self.assertEqual(closed, [True])

    async def test_json_route_retains_response_and_adds_request_timing(self):
        with patch.object(self.main, "history_store", FakeHistory()), \
                patch.object(self.main, "invoke_agent", return_value={"answer_markdown": "Verified answer"}), \
                patch("requests.sessions.Session.request", side_effect=AssertionError("No network in offline API test")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.main.app), base_url="http://testserver") as client:
                response = await client.post("/api/chat", json={"message": "fixture analytical question", "client_history": []})
        self.assertEqual(response.status_code, 200)
        payload = response.json()["response"]
        self.assertEqual(payload["answer_markdown"], "Verified answer")
        self.assertEqual(payload["request_id"], response.headers["x-ci-request-id"])
        self.assertTrue(any(item["name"] == "agent.invoke" for item in payload["timing"]["api"]["intervals"]))
