import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sapphirus_lab.cli import main


class CliTests(unittest.TestCase):
    def test_demo_completes_without_http_and_preserves_unreviewed_status(self):
        output = io.StringIO()
        with patch("http.client.HTTPConnection") as connection:
            with contextlib.redirect_stdout(output):
                code = main(["demo"])
        connection.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["counts"], {"actor_calls": 5, "tool_calls": 4})
        self.assertEqual(len(result["sources_read"]), 2)
        self.assertTrue(all(item["ok"] for item in result["observations"]))
        self.assertEqual(result["answer_status"], "unreviewed")

    def test_model_requests_require_explicit_flag_before_loading_corpus(self):
        with patch("http.client.HTTPConnection") as connection:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["run", "--goal", "test", "--corpus", "absent.json", "--model", "test"])
        self.assertEqual(caught.exception.code, 2)
        connection.assert_not_called()
