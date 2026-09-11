"""Small, runnable lab; actual model requests require an explicit option."""
import argparse
import json
import os
from uuid import uuid4
from pathlib import Path

from .core import Document, ToolCatalog, run_task
from .local_actor import LocalActor
from .drafts import DraftStore


def load_catalog(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ToolCatalog([Document(**row) for row in data])


class DemoActor:
    """Handwritten mechanical demonstration, not candidate performance."""
    def __init__(self):
        self.steps = iter([
            {"action": "use_tool", "speech": "", "tool_calls": [{"name": "search", "arguments": {"query": "조립"}}], "memory_candidates": []},
            {"action": "use_tool", "speech": "", "tool_calls": [{"name": "read", "arguments": {"source_id": "assembly"}}], "memory_candidates": []},
            {"action": "use_tool", "speech": "", "tool_calls": [{"name": "search", "arguments": {"query": "보관"}}], "memory_candidates": []},
            {"action": "use_tool", "speech": "", "tool_calls": [{"name": "read", "arguments": {"source_id": "storage"}}], "memory_candidates": []},
            {"action": "reply", "speech": "날개를 먼저 접고 홈을 끼워 조립해. 완성품은 펼친 채 빈 상자에 보관하고, 위에 다른 물건을 쌓지 않아. [assembly] [storage]", "tool_calls": [], "memory_candidates": []},
        ])

    def __call__(self, observation):
        return next(self.steps)


class WritingDemoActor:
    """Manually authored writing exercise; no model invocation."""
    def __init__(self):
        self.steps = iter([
            ("search", {"query": "조립"}),
            ("read", {"source_id": "assembly"}),
            ("write_draft", {"draft_id": "paper-star", "content": "# 종이별\n날개를 접고 홈을 끼워 조립한다.", "source_ids": ["assembly"]}),
            ("read_draft", {"draft_id": "paper-star", "version": 1}),
            ("search", {"query": "보관"}),
            ("read", {"source_id": "storage"}),
            ("revise_draft", {"draft_id": "paper-star", "content": "# 종이별\n날개를 접고 홈을 끼워 조립한다.\n완성품은 펼친 채 빈 상자에 넣고 위에 물건을 쌓지 않는다.", "source_ids": ["assembly", "storage"]}),
            ("read_draft", {"draft_id": "paper-star", "version": 2}),
        ])

    def __call__(self, observation):
        step = next(self.steps, None)
        if step is None:
            return {"action": "reply", "speech": "초안과 수정본을 저장했어. 내용은 검토 전이야.", "tool_calls": [], "memory_candidates": []}
        name, args = step
        return {"action": "use_tool", "speech": "", "tool_calls": [{"name": name, "arguments": args}], "memory_candidates": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sapphirus search/read loop lab")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("demo", help="scripted offline demonstration; not model evaluation")
    sub.add_parser("demo-write", help="offline demo that saves versioned Markdown drafts")
    run = sub.add_parser("run", help="use an already-running local model on a supplied corpus")
    run.add_argument("--goal", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--allow-model-requests", action="store_true")
    run.add_argument("--allow-draft-writes", action="store_true")
    run.add_argument("--max-actor-calls", type=int, default=8)
    run.add_argument("--max-tool-calls", type=int, default=6)
    args = parser.parse_args(argv)
    draft_store = None
    if args.mode == "demo-write" or (args.mode == "run" and args.allow_draft_writes and args.allow_model_requests):
        draft_store = DraftStore(Path(__file__).resolve().parents[2] / "drafts" / uuid4().hex)
    if args.mode in ("demo", "demo-write"):
        catalog = load_catalog(Path(__file__).resolve().parents[2] / "examples/paper_star.json")
        actor = WritingDemoActor() if args.mode == "demo-write" else DemoActor()
        result = run_task("종이별 조립과 보관 방법을 알아봐.", actor, catalog,
                          max_actor_calls=12, max_tool_calls=10, draft_store=draft_store)
        result["execution_kind"] = "scripted_offline_demo"
    else:
        if not args.allow_model_requests:
            parser.error("actual model requests require --allow-model-requests")
        catalog = load_catalog(args.corpus)
        actor = LocalActor(args.endpoint, args.model, api_key=os.getenv("SAPPHIRUS_LAB_API_KEY", ""))
        result = run_task(args.goal, actor, catalog,
                          max_actor_calls=args.max_actor_calls, max_tool_calls=args.max_tool_calls,
                          draft_store=draft_store)
        result["execution_kind"] = "local_model_requests"
    if draft_store is not None:
        result["draft_directory"] = str(draft_store.root)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] in ("reply", "silence") else 2


if __name__ == "__main__":
    raise SystemExit(main())
