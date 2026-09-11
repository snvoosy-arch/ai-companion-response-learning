import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sapphirus_lab.drafts import DraftStore
from sapphirus_lab.core import ToolCatalog, run_task
from sapphirus_lab.cli import WritingDemoActor, load_catalog


class DraftTests(unittest.TestCase):
    def test_full_writing_loop_and_source_binding(self):
        with tempfile.TemporaryDirectory() as root:
            store = DraftStore(Path(root))
            catalog = load_catalog(Path(__file__).resolve().parents[1] / "examples/paper_star.json")
            result = run_task("write", WritingDemoActor(), catalog, max_actor_calls=12,
                              max_tool_calls=10, draft_store=store)
            self.assertEqual(result["status"], "reply")
            self.assertEqual(result["counts"], {"actor_calls": 9, "tool_calls": 8})
            self.assertTrue(all(o["ok"] for o in result["observations"]))
            saved = store.read_draft("paper-star", 2)
            self.assertEqual(len(saved["sources"]), 2)
            self.assertTrue(all(len(s["sha256"]) == 64 for s in saved["sources"]))

    def test_unread_source_disabled_tools_and_budget_cannot_write(self):
        def actor(state):
            return {"action": "use_tool", "speech": "", "memory_candidates": [],
                    "tool_calls": [{"name": "write_draft", "arguments": {
                        "draft_id": "test", "content": "body", "source_ids": ["invented"]}}]}
        with tempfile.TemporaryDirectory() as root:
            store = DraftStore(Path(root))
            result = run_task("test", actor, ToolCatalog([]), max_actor_calls=1, draft_store=store)
            self.assertEqual(result["observations"][0]["error"], "source_not_read")
            self.assertEqual(list(Path(root).iterdir()), [])
            result = run_task("test", actor, ToolCatalog([]), max_actor_calls=1)
            self.assertEqual(result["observations"][0]["error"], "draft_tools_disabled")

    def test_versions_preserve_body_and_sources(self):
        with tempfile.TemporaryDirectory() as root:
            store = DraftStore(Path(root))
            first = store.write_draft("note", "first", [{"source_id": "a"}])
            second = store.revise_draft("note", "second", [])
            self.assertEqual(first["version"], 1)
            self.assertEqual(second["version"], 2)
            self.assertEqual(store.read_draft("note", 1)["content"], "first")
            self.assertEqual(store.read_draft("note", 1)["sources"], [{"source_id": "a"}])
            self.assertEqual(store.read_draft("note", 2)["content"], "second")
            self.assertFalse(store.write_draft("note", "overwrite", [])["ok"])

    def test_paths_and_missing_drafts_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            store = DraftStore(Path(root))
            for name in ("../escape", "a/b", "C:\\escape", "CON", ""):
                with self.subTest(name=name):
                    self.assertFalse(store.write_draft(name, "body", [])["ok"])
            self.assertFalse(store.revise_draft("absent", "body", [])["ok"])

    def test_redirected_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "real").mkdir()
            try:
                (base / "link").symlink_to(base / "real", target_is_directory=True)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                DraftStore(base / "link")
