import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sapphirus_lab.core import ToolCatalog, run_task
from sapphirus_lab.local_actor import LocalActor


class MalformedResponseTests(unittest.TestCase):
    def test_invalid_model_json_is_preserved_as_contract_failure(self):
        for content in ('{not JSON', '{"action":"reply","action":"silence"}'):
            with self.subTest(content=content):
                response = MagicMock(status=200)
                response.read.return_value = json.dumps({
                    "model": "test", "choices": [{"message": {"content": content}}]
                }).encode()
                with patch("http.client.HTTPConnection") as connection:
                    connection.return_value.getresponse.return_value = response
                    result = run_task("test", LocalActor("http://127.0.0.1:8001/v1", "test"), ToolCatalog([]))
                self.assertEqual(result["status"], "invalid_action")
                self.assertEqual(result["trace"][0]["raw"], content)
                self.assertEqual(result["counts"]["actor_calls"], 1)
