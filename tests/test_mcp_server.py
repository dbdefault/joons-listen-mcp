import json
import unittest
from unittest import mock

from joons_listen import mcp_server as m


class McpProtocol(unittest.TestCase):
    def test_initialize_and_list(self):
        r = m.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", r["result"]["capabilities"])
        r = m.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in r["result"]["tools"]]
        self.assertEqual(names, ["listen_transcribe", "listen_diarize", "listen_summarize", "listen_crosscheck", "listen_status", "listen_result", "listen_models"])
        ro = {t["name"]: t["annotations"]["readOnlyHint"] for t in r["result"]["tools"]}
        self.assertTrue(ro["listen_status"] and ro["listen_result"] and ro["listen_models"])
        self.assertFalse(ro["listen_transcribe"])

    def test_unknown_protocol_falls_back(self):
        r = m.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
        self.assertEqual(r["result"]["protocolVersion"], m.PROTOCOL)

    def test_notification_has_no_reply(self):
        self.assertIsNone(m.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_error(self):
        r = m.handle({"jsonrpc": "2.0", "id": 9, "method": "nope"})
        self.assertEqual(r["error"]["code"], -32601)

    def test_tool_call_posts_job(self):
        with mock.patch.object(m, "call_api", return_value={"ok": True, "job_id": "lst_1"}) as ca:
            r = m.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "listen_transcribe", "arguments": {"audio_path": "a.wav", "minutes": True, "junk": 1}}})
        ca.assert_called_once_with("POST", "/jobs", {"kind": "transcribe", "audio_path": "a.wav", "minutes": True})
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(json.loads(r["result"]["content"][0]["text"])["job_id"], "lst_1")

    def test_service_error_marked(self):
        with mock.patch.object(m, "call_api", return_value={"ok": False, "code": "NOT_READY", "message": "x"}):
            r = m.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "listen_result", "arguments": {"job_id": "j"}}})
        self.assertTrue(r["result"]["isError"])

    def test_job_id_is_quoted_in_path(self):
        with mock.patch.object(m, "call_api", return_value={"ok": True}) as ca:
            m.tool_call("listen_status", {"job_id": "../health"})
        self.assertEqual(ca.call_args[0][1], "/jobs/..%2Fhealth")

    def test_unknown_tool(self):
        self.assertTrue(m.tool_call("nope", {})["isError"])

    def test_non_loopback_url_refused(self):
        with mock.patch.object(m, "BASE_URL", "http://evil.example.com:1"):
            with self.assertRaises(RuntimeError):
                m.call_api("GET", "/health")


if __name__ == "__main__":
    unittest.main()
