from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sapphirus_lab.local_actor import LocalActor  # noqa: E402


class LocalActorTests(unittest.TestCase):
    def test_remote_or_credentialed_endpoint_rejected(self):
        for url in ("https://example.com/v1", "http://10.0.0.1/v1", "http://user:secret@127.0.0.1:8001/v1", "http://127.0.0.1:8001/v1?key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalActor(url, "candidate")

    def test_valid_constructor_never_opens_connection(self):
        with patch("http.client.HTTPConnection") as connection:
            LocalActor("http://127.0.0.1:8001/v1", "candidate")
            connection.assert_not_called()

    def test_http_error_has_no_retry_or_secret_leak(self):
        with patch("http.client.HTTPConnection") as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 401
            response.read.return_value = b'secret should not appear in errors'
            actor = LocalActor("http://127.0.0.1:8001/v1", "candidate", api_key="private")
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                actor({"goal": "read", "observations": [], "remaining_budget": {}})
            self.assertEqual(connection.return_value.request.call_count, 1)
            connection.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
