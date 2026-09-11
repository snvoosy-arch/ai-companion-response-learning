import unittest

from sapphirus_lab import core


def action(kind, speech="", name=None, **arguments):
    return {"action": kind, "speech": speech,
            "tool_calls": [] if name is None else [{"name": name, "arguments": arguments}],
            "memory_candidates": []}


def scripted(*actions):
    sequence = iter(actions)
    return lambda state: next(sequence)


class CatalogTests(unittest.TestCase):
    def test_search_snippets_are_distinct_from_authorized_body_read(self):
        catalog = core.ToolCatalog([core.Document("a", "Harbor", "fixture:a", "abc"),
                                    core.Document("b", "Forest", "fixture:b", "trees")])
        self.assertFalse(catalog.read("a")["ok"])
        search = catalog.search("HARBOR unknown")
        self.assertEqual([r["source_id"] for r in search["results"]], ["a"])
        self.assertNotIn("text", search["results"][0])
        body = catalog.read("a")
        self.assertEqual(body["text"], "abc")
        self.assertEqual(body["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertFalse(catalog.read("b")["ok"])
        self.assertFalse(catalog.read("missing")["ok"])

    def test_search_and_body_are_bounded_and_truncation_is_visible(self):
        catalog = core.ToolCatalog([core.Document(str(i), "long", "fixture:x", "x" * 13000)
                                    for i in range(40)])
        search = catalog.search("long")
        self.assertLessEqual(len(search["results"]), 10)
        self.assertLessEqual(len(search["results"][0]["snippet"]), 400)
        read = catalog.read("0")
        self.assertEqual(len(read["text"]), 12000)
        self.assertTrue(read["truncated"])
        self.assertFalse(catalog.read("39")["ok"])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            core.ToolCatalog([core.Document("a", "A", "fixture:a", "x"),
                              core.Document("a", "B", "fixture:b", "y")])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.catalog = core.ToolCatalog([core.Document("a", "Harbor", "fixture:a", "abc"),
                                         core.Document("b", "Forest", "fixture:b", "trees")])

    def test_same_actor_searches_reads_searches_again_and_replies(self):
        seen = []
        sequence = iter([action("use_tool", name="search", query="Harbor"),
                         action("use_tool", name="read", source_id="a"),
                         action("use_tool", name="search", query="Forest"),
                         action("use_tool", name="read", source_id="b"),
                         action("reply", "Manual answer")])
        def actor(state):
            seen.append(state)
            return next(sequence)
        result = core.run_task("Compare fixtures", actor, self.catalog)
        self.assertEqual(result["status"], "reply")
        self.assertTrue(result["complete"])
        self.assertEqual(result["answer_status"], "unreviewed")
        self.assertEqual(result["speech"], "Manual answer")
        self.assertEqual(result["counts"], {"actor_calls": 5, "tool_calls": 4})
        self.assertEqual([s["source_id"] for s in result["sources_read"]], ["a", "b"])
        self.assertEqual(seen[0]["remaining_budget"], {"actor_calls": 7, "tool_calls": 6})
        self.assertEqual(len(seen[4]["observations"]), 4)
        self.assertTrue(all(o["untrusted"] for o in seen[4]["observations"]))

    def test_task_does_not_inherit_previous_search_authorization(self):
        self.catalog.search("Harbor")
        result = core.run_task("Read", scripted(action("use_tool", name="read", source_id="a"),
                                               action("silence")), self.catalog)
        self.assertFalse(result["observations"][0]["ok"])
        self.assertEqual(result["sources_read"], [])
        self.assertEqual(result["status"], "silence")

    def test_repeated_action_consumes_budget_without_reexecuting(self):
        request = action("use_tool", name="search", query="Harbor")
        result = core.run_task("Search", scripted(request, request, request), self.catalog,
                               max_actor_calls=3, max_tool_calls=3)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["complete"])
        self.assertEqual(result["speech"], "")
        self.assertEqual(result["counts"], {"actor_calls": 3, "tool_calls": 3})
        self.assertEqual(result["observations"][1]["error"], "repeated_action")

    def test_tool_budget_does_not_prevent_final_actor_reply(self):
        result = core.run_task("Search", scripted(action("use_tool", name="search", query="Harbor"),
                                                 action("reply", "Done")), self.catalog,
                               max_actor_calls=2, max_tool_calls=1)
        self.assertEqual(result["status"], "reply")

    def test_exhausted_tool_budget_refuses_next_tool(self):
        result = core.run_task("Search", scripted(action("use_tool", name="search", query="Harbor"),
                                                 action("use_tool", name="read", source_id="a")),
                               self.catalog, max_actor_calls=3, max_tool_calls=1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["counts"], {"actor_calls": 2, "tool_calls": 1})
        self.assertEqual(result["sources_read"], [])

    def test_malformed_actor_actions_are_terminal_and_preserved(self):
        invalids = [None, {}, action("reply"), action("silence", "oops"),
                    action("use_tool", name="delete", source_id="a"),
                    action("use_tool", name="read", source_id="a", extra=True),
                    action("reply", "hi") | {"memory_candidates": ["save"]},
                    action("reply", "hi") | {"extra": "field"},
                    action("use_tool", name="search", query="Harbor") | {"speech": "doing"}]
        for raw in invalids:
            with self.subTest(raw=raw):
                result = core.run_task("Test", scripted(raw), self.catalog)
                self.assertEqual(result["status"], "invalid_action")
                self.assertEqual(result["trace"][0]["raw"], raw)
                self.assertEqual(result["counts"], {"actor_calls": 1, "tool_calls": 0})

    def test_actor_failure_consumes_reservation_and_never_retries(self):
        def actor(state):
            raise RuntimeError("offline")
        result = core.run_task("Test", actor, self.catalog)
        self.assertEqual(result["status"], "actor_error")
        self.assertEqual(result["counts"], {"actor_calls": 1, "tool_calls": 0})

    def test_mutating_actor_input_cannot_change_evidence(self):
        calls = 0
        def actor(state):
            nonlocal calls
            calls += 1
            if calls == 1:
                return action("use_tool", name="search", query="Harbor")
            state["observations"][0]["results"][0]["source_id"] = "forged"
            return action("reply", "Done")
        result = core.run_task("Test", actor, self.catalog)
        self.assertEqual(result["observations"][0]["results"][0]["source_id"], "a")

    def test_invalid_budget_values_rejected_before_actor_call(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.run_task("Test", scripted(), self.catalog, max_actor_calls=value)
                with self.assertRaises(ValueError):
                    core.run_task("Test", scripted(), self.catalog, max_tool_calls=value)

    def test_failed_tool_observation_allows_actor_to_finish(self):
        result = core.run_task("Test", scripted(action("use_tool", name="read", source_id="missing"),
                                               action("reply", "Could not read that source.")), self.catalog)
        self.assertFalse(result["observations"][0]["ok"])
        self.assertEqual(result["counts"], {"actor_calls": 2, "tool_calls": 1})
        self.assertEqual(result["status"], "reply")

    def test_tool_exception_is_observed_without_refunding_budget(self):
        catalog = core.ToolCatalog([core.Document("a", "Harbor", "fixture:a", "\ud800")])
        result = core.run_task("Test", scripted(action("use_tool", name="search", query="Harbor"),
                                               action("use_tool", name="read", source_id="a"),
                                               action("silence")), catalog)
        self.assertEqual(result["observations"][1]["error"], "tool_error")
        self.assertEqual(result["counts"], {"actor_calls": 3, "tool_calls": 2})
        self.assertEqual(result["sources_read"], [])
        self.assertEqual(result["status"], "silence")

    def test_actor_context_window_is_bounded_without_losing_audit(self):
        contexts = []
        def actor(state):
            contexts.append(state)
            return action("use_tool", name="search", query=f"Harbor q{len(contexts)}")
        result = core.run_task("Test", actor, self.catalog, max_actor_calls=10, max_tool_calls=10)
        self.assertLessEqual(max(len(s["observations"]) for s in contexts), 6)
        self.assertEqual(len(result["observations"]), 10)

    def test_invalid_or_oversized_input_is_rejected(self):
        for goal in ("", " " , None, "x" * 8001):
            with self.subTest(goal_type=type(goal).__name__):
                with self.assertRaises(ValueError):
                    core.run_task(goal, scripted(), self.catalog)
        result = core.run_task("Test", scripted(action("reply", "x" * 50000)), self.catalog)
        self.assertEqual(result["status"], "invalid_action")
        self.assertTrue(result["trace"][0]["raw"]["raw_truncated"])

    def test_multiple_tool_calls_cannot_bypass_single_call_limit(self):
        raw = action("use_tool", name="search", query="Harbor")
        raw["tool_calls"].append({"name": "read", "arguments": {"source_id": "a"}})
        result = core.run_task("Test", scripted(raw), self.catalog)
        self.assertEqual(result["status"], "invalid_action")
        self.assertEqual(result["counts"]["tool_calls"], 0)


if __name__ == "__main__":
    unittest.main()
