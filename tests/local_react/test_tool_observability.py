"""Offline MCP transport, trust boundary, server lifecycle and overlap regressions."""
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import sys
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'app' / 'backend'), str(ROOT / 'agent' / 'src')]
from app.local_react.mcp_observability import (LoopbackTraceMiddleware, bounded_server_observability,
                                            trace_headers_for_url, install_loopback_tracing, tool_request_trace)
from app.local_react.tools import ToolRuntime
from app.local_react.wren_client import WrenMCPClient


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wren = module_at('observability_wren_server', 'wren/scripts/fallback_mcp_server.py')
tavily = module_at('observability_tavily_server', 'scripts/tavily_mcp_server.py')


class ToolObservabilityTests(unittest.TestCase):
    def setUp(self):
        # A local .env can enable tracing; tests never export to a cloud collector.
        self.tracing = patch('app.local_react.langsmith_trace.tracing_enabled', return_value=False)
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def response(self, result):
        response = Mock(status_code=200, text='', headers={'content-type': 'application/json'})
        response.json.return_value = {'result': result}
        return response

    def test_mcp_initialization_and_error_metadata_and_close(self):
        client = WrenMCPClient('http://127.0.0.1:7080/mcp')
        client._session = Mock()
        tool = {'content': [{'type': 'text', 'text': json.dumps({'ok': True, 'rows': [],
            '_ci_observability': {'component': 'wren_server', 'stages': [{'stage': 'sql.connect', 'ms': 8, 'ok': True}], 'secret': 'discard'}})}]}
        client._session.post.side_effect = [self.response({}), self.response({}), self.response(tool),
            self.response({'isError': True, 'content': [{'type': 'text', 'text': 'failure'}]})]
        self.assertNotIn('_ci_observability', client.call_tool('run_sql', {}))
        self.assertEqual([s['stage'] for s in client.last_observability['stages']],
                         ['mcp.initialize', 'mcp.initialized_notification', 'mcp.tools.call'])
        self.assertNotIn('secret', client.last_observability['server'])
        with self.assertRaises(RuntimeError):
            client.call_tool('run_sql', {})
        self.assertFalse(client.last_observability['stages'][-1]['ok'])
        client.close()
        client._session.close.assert_called_once()

    def test_alias_failure_is_retained(self):
        # Tool-level MCP errors must surface immediately and must NOT be masked by
        # trying the next alias (that used to hide real warehouse/auth failures).
        from app.local_react.wren_client import WrenToolError
        client = WrenMCPClient()
        client._initialized = True
        client._session = Mock()
        client._session.post.side_effect = [self.response({'isError': True}), self.response({'content': [{'type': 'text', 'text': '{"ok": true}'}]})]
        with self.assertRaises(WrenToolError):
            client.run_sql('SELECT ...')
        # Only the failing tool call was attempted; next alias was not used.
        self.assertEqual(client._session.post.call_count, 1)

    def test_parent_headers_are_only_sent_to_loopback_http(self):
        with patch('app.local_react.langsmith_trace.current_trace_headers', return_value={'langsmith-trace': 'parent'}):
            self.assertEqual(trace_headers_for_url('http://127.0.0.1:7080/mcp'), {'langsmith-trace': 'parent'})
            for url in ['https://public.example/mcp', 'http://wren-mcp:7080/mcp', 'https://localhost:7080/mcp']:
                self.assertEqual(trace_headers_for_url(url), {})

    def test_incoming_external_and_forwarded_parents_ignored(self):
        accepted = []
        @contextmanager
        def capture(headers):
            accepted.append(headers)
            yield
        async def app(scope, receive, send):
            pass
        wrapper = LoopbackTraceMiddleware(app)
        async def check():
            for host, extra in [('127.0.0.1', []), ('10.0.0.4', []), ('127.0.0.1', [(b'x-forwarded-for', b'10.0.0.4')])]:
                await wrapper({'type': 'http', 'client': (host, 20), 'headers': [(b'langsmith-trace', b'parent'), (b'authorization', b'never pass')] + extra}, None, None)
        with patch('app.local_react.langsmith_trace.propagated_trace', capture):
            asyncio.run(check())
        self.assertEqual(accepted, [{'langsmith-trace': 'parent'}])

    def test_sql_success_and_execute_failure_close_connections(self):
        for fails in [False, True]:
            connection = Mock()
            cursor = Mock(description=[('total',)], fetchall=Mock(return_value=[(10,)]))
            cursor.__enter__ = Mock(return_value=cursor)
            cursor.__exit__ = Mock(return_value=False)
            if fails:
                cursor.execute.side_effect = RuntimeError('offline execute failure')
            connection.cursor.return_value = cursor
            with patch.object(wren, '_connect', return_value=connection):
                result = wren.tool_run_sql('SELECT * FROM genie_monthly_traffic', 2)
            stages = result['_ci_observability']['stages']
            self.assertEqual([s['stage'] for s in stages][:3], ['sql.validate', 'sql.connect', 'sql.execute'])
            self.assertEqual(stages[-1]['stage'], 'sql.close')
            connection.close.assert_called_once()
            if fails:
                self.assertFalse(result['ok'])
                self.assertFalse(stages[2]['ok'])
            else:
                self.assertEqual(result['rows'], [{'total': 10}])
                self.assertIn('sql.fetch', [s['stage'] for s in stages])

    def test_sql_validation_failure_never_connects(self):
        with patch.object(wren, '_connect') as connect:
            result = wren.tool_run_sql('DELETE FROM genie_monthly_traffic')
        connect.assert_not_called()
        self.assertFalse(result['ok'])
        self.assertEqual([s['stage'] for s in result['_ci_observability']['stages']], ['sql.validate'])

    def test_tavily_error_still_reports_server_timing(self):
        with patch.object(tavily, '_search_impl', return_value={'ok': False, 'error': 'offline failure'}):
            result = tavily._search('query')
        self.assertFalse(result['_ci_observability']['stages'][0]['ok'])
        self.assertGreaterEqual(result['_ci_observability']['stages'][0]['ms'], 0)

    def test_returned_tavily_failure_marks_trace_before_scope_exit(self):
        run = SimpleNamespace(error=None, add_outputs=Mock())
        @contextmanager
        def traced(name, **kwargs):
            try:
                yield run
            finally:
                self.assertIsNotNone(run.error)
        with patch('app.local_react.langsmith_trace.chain_span', traced), patch.object(
            tavily, '_search_impl', return_value={'ok': False, 'error': 'offline provider response'}
        ):
            result = tavily._search('query')
        self.assertFalse(result['ok'])
        self.assertEqual(run.error, 'operation returned an error')
        self.assertEqual(run.add_outputs.call_args.args[0], {'ok': False, 'error': 'Operation returned failure'})

    def test_failed_alias_attempts_keep_server_and_transport_evidence(self):
        from app.local_react.wren_client import WrenToolError
        client = WrenMCPClient()
        client._initialized = True
        client._session = Mock()
        failure = {'content': [{'type': 'text', 'text': json.dumps({'ok': False, 'error': 'offline failure',
            '_ci_observability': {'component': 'wren_server', 'stages': [{'stage': 'sql.execute', 'ms': 5, 'ok': False}]}})}]}
        client._session.post.side_effect = [self.response(failure), self.response({'isError': True})]
        with self.assertRaises(WrenToolError) as ctx:
            client.run_sql('SELECT ...')
        self.assertIn('offline failure', str(ctx.exception))
        # Real tool error stops alias fan-out; server observability from the failing call remains.
        self.assertEqual(client._session.post.call_count, 1)
        self.assertFalse(client.last_observability['server']['stages'][0]['ok'])
        self.assertFalse(client.last_observability['stages'][0]['ok'])

    def test_mcp_failure_then_rest_attempt_preserved(self):
        runtime = ToolRuntime()
        client = Mock(last_observability={'transport': 'mcp', 'stages': [{'stage': 'mcp.initialize', 'ms': 7, 'ok': False}]})
        client.call_tool.side_effect = RuntimeError('offline MCP failure')
        rest = SimpleNamespace(status='ok', text='answer', sources=[])
        with patch.dict('os.environ', {'TAVILY_MCP_URL': 'http://127.0.0.1:7081/mcp', 'TAVILY_MCP_ENABLED': 'true'}), \
             patch('app.local_react.wren_client.WrenMCPClient', return_value=client), \
             patch('dualsource.tools.tavily.TavilyClient') as rest_client:
            rest_client.return_value.search.return_value = rest
            result = runtime.execute('web_search', {'query': 'airport 2026'})
        self.assertEqual(result['backend'], 'rest_after_mcp_error')
        self.assertNotIn('_ci_observability', result)
        attempts = runtime.tool_calls[0]['observability']['attempts']
        self.assertEqual([a['transport'] for a in attempts], ['mcp', 'tavily_rest'])
        client.close.assert_called_once()

    def test_parallel_tools_have_timeline_and_union_not_sum(self):
        runtime = ToolRuntime()
        barrier = threading.Barrier(2)
        release = threading.Barrier(2)
        def fetch(*args, **kwargs):
            barrier.wait(timeout=3)
            release.wait(timeout=3)
            return {'ok': True, 'text': 'offline', 'sources': []}
        with patch.object(runtime, '_web_search_fetch', side_effect=fetch):
            workers = [threading.Thread(target=runtime.execute, args=('web_search', {'query': str(i)})) for i in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=4)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(len(runtime.tool_calls), 2)
        intervals = sorted((c['start_ms'], c['end_ms']) for c in runtime.tool_calls)
        self.assertLess(intervals[1][0], intervals[0][1])
        union = max(e for _, e in intervals) - min(s for s, _ in intervals)
        self.assertAlmostEqual(runtime.tool_wall_ms, union, delta=.02)
        self.assertLess(runtime.tool_wall_ms, sum(e-s for s, e in intervals))

    def test_legacy_tool_records_do_not_invent_wall_timing(self):
        runtime = ToolRuntime()
        runtime.tool_calls = [{'name': 'legacy', 'ms': 50}, {'start_ms': 2, 'end_ms': 7},
                              {'start_ms': 20, 'end_ms': 10}, {'start_ms': float('nan'), 'end_ms': 30}]
        self.assertEqual(runtime.tool_wall_ms, 5)
        self.assertFalse(runtime.tool_timeline_complete)
        runtime.tool_calls = [{'start_ms': 2, 'end_ms': 7}]
        self.assertTrue(runtime.tool_timeline_complete)

    def test_fastmcp_worker_context_isolated_and_schema_unchanged(self):
        try:
            from mcp.server.fastmcp import FastMCP, Context
            from starlette.testclient import TestClient
        except ImportError:
            self.skipTest("Optional MCP test dependencies not installed")
        parent = contextvars.ContextVar("test_parent", default=None)
        @contextmanager
        def capture(headers):
            token = parent.set(headers.get("langsmith-trace"))
            try:
                yield
            finally:
                parent.reset(token)
        both_entered = asyncio.Event()
        entered = []
        mcp = FastMCP("offline-parent-test", json_response=True)
        @mcp.tool()
        async def isolated(label: str, ctx: Context) -> dict:
            with tool_request_trace(ctx):
                if label in {"one", "two"}:
                    entered.append(label)
                    if len(entered) == 2:
                        both_entered.set()
                    await asyncio.wait_for(both_entered.wait(), 3)
                return {"parent": parent.get()}
        install_loopback_tracing(mcp)
        with patch('app.local_react.langsmith_trace.propagated_trace', capture), TestClient(
            mcp.streamable_http_app(), base_url='http://127.0.0.1:7080', client=('127.0.0.1', 55000)
        ) as client:
            common = {'accept': 'application/json, text/event-stream'}
            response = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
                'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'offline', 'version': '0'}}})
            self.assertEqual(response.status_code, 200)
            common['mcp-session-id'] = response.headers['mcp-session-id']
            client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'method': 'notifications/initialized'})
            listing = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}).json()
            self.assertEqual(set(listing['result']['tools'][0]['inputSchema']['properties']), {'label'})
            def call(label):
                response = client.post('/mcp', headers={**common, 'langsmith-trace': label}, json={
                    'jsonrpc': '2.0', 'id': label, 'method': 'tools/call', 'params': {'name': 'isolated', 'arguments': {'label': label}}})
                return json.loads(response.json()['result']['content'][0]['text'])['parent']
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(call, label) for label in ('one', 'two')]
                self.assertEqual([f.result(timeout=5) for f in futures], ['one', 'two'])
            # Forged body metadata has no authority without a trusted parent header.
            response = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {
                'name': 'isolated', 'arguments': {'label': 'forged'}, '_meta': {'ci_parent_trace': {'headers': {'langsmith-trace': 'evil'}, 'signature': 'forged'}}}})
            self.assertIsNone(json.loads(response.json()['result']['content'][0]['text'])['parent'])

    def test_middleware_preserves_malformed_frames_without_trust(self):
        captured = []
        async def app(scope, receive, send):
            captured.append((await receive()).get('body'))
        wrapper = LoopbackTraceMiddleware(app)
        async def check():
            for body in [b'not json', b'[]', b'{"method":"tools/call","params":null}', b'{"method":"tools/call","params":{"_meta":[]}}']:
                messages = [{'type': 'http.request', 'body': body, 'more_body': False}]
                async def receive():
                    return messages.pop(0)
                await wrapper({'type': 'http', 'method': 'POST', 'client': ('10.0.0.4', 20), 'headers': []}, receive, None)
        asyncio.run(check())
        self.assertEqual(captured, [b'not json', b'[]', b'{"method":"tools/call","params":null}', b'{"method":"tools/call","params":{"_meta":[]}}'])
        with tool_request_trace(SimpleNamespace(request_context=SimpleNamespace(meta={'ci_parent_trace': ['malformed']}))):
            pass

    def test_real_wren_mcp_factory_schema_and_server_metadata(self):
        try:
            from mcp.server.transport_security import TransportSecuritySettings
            from starlette.testclient import TestClient
        except ImportError:
            self.skipTest('Requires the repository MCP >=1.19 dependency')
        mcp = wren.build_mcp()
        mcp.settings.json_response = True
        current = contextvars.ContextVar('wren_parent', default=None)
        observed = []
        @contextmanager
        def capture(headers):
            token = current.set(headers.get('langsmith-trace'))
            try:
                yield
            finally:
                current.reset(token)
        connection = Mock()
        cursor = Mock(description=[('total',)], fetchall=Mock(return_value=[(10,)]))
        cursor.__enter__ = Mock(return_value=cursor)
        cursor.__exit__ = Mock(return_value=False)
        connection.cursor.return_value = cursor
        def connect():
            observed.append(current.get())
            return connection
        with patch.object(wren, '_connect', side_effect=connect), patch('app.local_react.langsmith_trace.propagated_trace', capture), TestClient(
            mcp.streamable_http_app(), base_url='http://127.0.0.1:7080', client=('127.0.0.1', 55000)
        ) as client:
            common = {'accept': 'application/json, text/event-stream', 'langsmith-trace': 'real-wren-parent'}
            response = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
                'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'offline', 'version': '0'}}})
            self.assertEqual(response.status_code, 200)
            common['mcp-session-id'] = response.headers['mcp-session-id']
            client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'method': 'notifications/initialized'})
            listing = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}).json()
            for tool in listing['result']['tools']:
                self.assertNotIn('ctx', tool['inputSchema'].get('properties', {}))
            response = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
                'name': 'run_sql', 'arguments': {'sql': 'SELECT * FROM genie_monthly_traffic', 'limit': 2}}}).json()
            result = json.loads(response['result']['content'][0]['text'])
            self.assertEqual(result['rows'], [{'total': 10}])
            self.assertIn('sql.execute', [s['stage'] for s in result['_ci_observability']['stages']])
        connection.close.assert_called_once()
        self.assertEqual(observed, ['real-wren-parent'])

    def test_real_tavily_mcp_factory_parent_and_server_metadata(self):
        try:
            from mcp.server.transport_security import TransportSecuritySettings
            from starlette.testclient import TestClient
        except ImportError:
            self.skipTest('Requires the repository MCP >=1.19 dependency')
        mcp = tavily.build_mcp()
        mcp.settings.json_response = True
        current = contextvars.ContextVar('tavily_parent', default=None)
        observed = []
        @contextmanager
        def capture(headers):
            token = current.set(headers.get('langsmith-trace'))
            try:
                yield
            finally:
                current.reset(token)
        def search(*args, **kwargs):
            observed.append(current.get())
            return {'ok': True, 'text': 'offline answer', 'sources': []}
        with patch.object(tavily, '_search_impl', side_effect=search), patch('app.local_react.langsmith_trace.propagated_trace', capture), TestClient(
            mcp.streamable_http_app(), base_url='http://127.0.0.1:7081', client=('127.0.0.1', 55000)
        ) as client:
            common = {'accept': 'application/json, text/event-stream', 'langsmith-trace': 'real-tavily-parent'}
            response = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
                'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'offline', 'version': '0'}}})
            self.assertEqual(response.status_code, 200)
            common['mcp-session-id'] = response.headers['mcp-session-id']
            client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'method': 'notifications/initialized'})
            response = client.post('/mcp', headers=common, json={'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
                'name': 'tavily_search', 'arguments': {'query': 'airport 2026'}}}).json()
            result = json.loads(response['result']['content'][0]['text'])
            self.assertEqual(result['text'], 'offline answer')
            self.assertEqual(result['_ci_observability']['stages'][0]['stage'], 'tavily.search')
        self.assertEqual(observed, ['real-tavily-parent'])

    def test_metadata_filter_discards_nonfinite_and_arbitrary_fields(self):
        value = bounded_server_observability({'component': 'evil', 'stages': [{'stage': 'sql.execute', 'ms': float('nan'), 'sql': 'do not expose'}, {'stage': 'raw SQL here', 'ms': 2}]})
        self.assertEqual(value, {'component': 'mcp_server', 'stages': [{'stage': 'sql.execute', 'ok': False}]})


if __name__ == '__main__':
    unittest.main()
