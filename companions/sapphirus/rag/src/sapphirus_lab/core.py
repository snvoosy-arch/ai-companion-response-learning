"""Bounded Actor loop over a manual, read-only document catalog.

Tool output is evidence, never instruction authority. Completion describes the
protocol's terminal action and does not assert factual or semantic correctness.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Sequence


@dataclass(frozen=True)
class Document:
    source_id: str
    title: str
    url: str
    text: str


def _error(tool: str, error: str) -> dict:
    return {"tool": tool, "ok": False, "untrusted": True, "error": error}


class ToolCatalog:
    """Search returns at most ten literal-token OR matches, in input order.

Standalone read authorization lasts for this instance. run_task creates its
own catalog session, so earlier searches cannot authorize later tasks.
"""

    def __init__(self, documents: Sequence[Document]):
        self.documents = tuple(documents)
        self._documents = {}
        self._discovered = set()
        for doc in self.documents:
            if not isinstance(doc, Document):
                raise ValueError("documents must contain Document instances")
            for field, limit in (("source_id", 128), ("title", 512), ("url", 2048)):
                value = getattr(doc, field)
                if not isinstance(value, str) or not value.strip() or len(value) > limit:
                    raise ValueError(f"invalid document {field}")
            if not isinstance(doc.text, str):
                raise ValueError("document text must be a string")
            if doc.source_id in self._documents:
                raise ValueError("duplicate source_id")
            self._documents[doc.source_id] = doc

    def search(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            return _error("search", "invalid_query")
        tokens = set(re.findall(r"\w+", query.casefold()))
        results = []
        for doc in self.documents:
            words = set(re.findall(r"\w+", (doc.title + " " + doc.text).casefold()))
            if tokens & words:
                results.append({"source_id": doc.source_id, "title": doc.title,
                                "url": doc.url, "snippet": doc.text[:400]})
                self._discovered.add(doc.source_id)
                if len(results) == 10:
                    break
        return {"tool": "search", "ok": True, "untrusted": True,
                "query": query, "results": results}

    def read(self, source_id: str) -> dict:
        if not isinstance(source_id, str) or source_id not in self._discovered:
            return _error("read", "source_not_discovered")
        doc = self._documents[source_id]
        return {"tool": "read", "ok": True, "untrusted": True,
                "source_id": doc.source_id, "title": doc.title, "url": doc.url,
                "text": doc.text[:12000],
                "sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                "truncated": len(doc.text) > 12000}


def _validate_action(raw: object) -> str | None:
    if not isinstance(raw, dict) or set(raw) != {"action", "speech", "tool_calls", "memory_candidates"}:
        return "expected_exact_action_fields"
    if not isinstance(raw["speech"], str) or len(raw["speech"]) > 12000:
        return "invalid_speech"
    if not isinstance(raw["memory_candidates"], list) or raw["memory_candidates"]:
        return "memory_disabled"
    if not isinstance(raw["tool_calls"], list):
        return "invalid_tool_calls"
    if raw["action"] == "reply":
        return None if raw["speech"].strip() and not raw["tool_calls"] else "invalid_reply"
    if raw["action"] == "silence":
        return None if raw["speech"] == "" and not raw["tool_calls"] else "invalid_silence"
    if raw["action"] != "use_tool" or raw["speech"] != "" or len(raw["tool_calls"]) != 1:
        return "invalid_tool_action"
    call = raw["tool_calls"][0]
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return "invalid_tool_call_fields"
    name = call["name"]
    if name not in ("search", "read", "write_draft", "read_draft", "revise_draft"):
        return "unknown_tool"
    arguments = call["arguments"]
    if name in ("write_draft", "revise_draft", "read_draft"):
        fields = {"draft_id", "version"} if name == "read_draft" else {"draft_id", "content", "source_ids"}
        if not isinstance(arguments, dict) or set(arguments) != fields:
            return "invalid_tool_arguments"
        if not isinstance(arguments["draft_id"], str):
            return "invalid_draft_id"
        if name == "read_draft":
            return None if type(arguments["version"]) is int and arguments["version"] > 0 else "invalid_version"
        if not isinstance(arguments["content"], str) or not 1 <= len(arguments["content"]) <= 12000:
            return "invalid_content"
        ids = arguments["source_ids"]
        return None if isinstance(ids, list) and len(ids) <= 20 and all(isinstance(i, str) for i in ids) else "invalid_sources"
    key, limit = ("query", 512) if name == "search" else ("source_id", 128)
    if not isinstance(arguments, dict) or set(arguments) != {key}:
        return "invalid_tool_arguments"
    value = arguments[key]
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        return "invalid_tool_argument_value"
    return None


def _raw_record(raw: object) -> object:
    """Preserve ordinary JSON actions exactly; cap oversized/invalid evidence."""
    try:
        encoded = json.dumps(raw, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return {"unserializable_type": type(raw).__name__}
    if len(encoded) > 32000:
        return {"raw_truncated": True, "preview": encoded[:32000]}
    return json.loads(encoded)


def run_task(goal: str, actor: Callable[[dict], dict], catalog: ToolCatalog, *,
             max_actor_calls: int = 8, max_tool_calls: int = 6, draft_store=None) -> dict:
    """Reserve calls before execution, never retry, and never invent a reply.

    Actor context includes the last six observations. Full bounded individual
    observations remain in the audit result. The Actor callable is trusted code;
    this protocol does not sandbox it or preempt a callable that never returns.
    """
    for budget in (max_actor_calls, max_tool_calls):
        if type(budget) is not int or budget <= 0:
            raise ValueError("budgets must be positive integers")
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 8000:
        raise ValueError("goal must contain 1 to 8000 characters")
    session = ToolCatalog(catalog.documents)
    counts = {"actor_calls": 0, "tool_calls": 0}
    trace, observations, sources_read = [], [], []
    seen_actions, read_ids = set(), set()

    def finish(status: str, speech: str = "") -> dict:
        return {"status": status, "complete": status in ("reply", "silence"),
                "speech": speech, "answer_status": "unreviewed", "counts": counts,
                "trace": trace, "observations": observations, "sources_read": sources_read}

    while counts["actor_calls"] < max_actor_calls:
        counts["actor_calls"] += 1
        state = {"goal": goal, "remaining_budget": {
            "actor_calls": max_actor_calls - counts["actor_calls"],
            "tool_calls": max_tool_calls - counts["tool_calls"]},
            "observations": deepcopy(observations[-6:]),
            "draft_tools_enabled": draft_store is not None}
        try:
            raw = actor(state)
        except Exception as exc:
            trace.append({"kind": "actor", "error": type(exc).__name__,
                          "message": str(exc)[:1000]})
            return finish("actor_error")
        record = {"kind": "actor", "raw": _raw_record(raw)}
        trace.append(record)
        error = _validate_action(raw)
        if error:
            record["error"] = error
            return finish("invalid_action")
        if raw["action"] in ("reply", "silence"):
            return finish(raw["action"], raw["speech"])
        if counts["tool_calls"] >= max_tool_calls:
            return finish("budget_exhausted")
        counts["tool_calls"] += 1
        call = raw["tool_calls"][0]
        key = json.dumps(call, sort_keys=True)
        if key in seen_actions:
            observation = _error(call["name"], "repeated_action")
        else:
            seen_actions.add(key)
            try:
                name, args = call["name"], call["arguments"]
                if name in ("write_draft", "read_draft", "revise_draft"):
                    if draft_store is None:
                        observation = _error(name, "draft_tools_disabled")
                    elif name == "read_draft":
                        observation = draft_store.read_draft(**args)
                    elif any(i not in read_ids for i in args["source_ids"]):
                        observation = _error(name, "source_not_read")
                    else:
                        sources = [s for s in sources_read if s["source_id"] in args["source_ids"]]
                        observation = getattr(draft_store, name)(args["draft_id"], args["content"], sources)
                else:
                    observation = getattr(session, name)(**args)
            except Exception as exc:
                observation = _error(call["name"], "tool_error")
                observation["detail"] = type(exc).__name__
        observations.append(observation)
        trace.append({"kind": "tool", "call": deepcopy(call), "observation": deepcopy(observation)})
        if observation["ok"] and observation["tool"] == "read":
            source_id = observation["source_id"]
            if source_id not in read_ids:
                read_ids.add(source_id)
                sources_read.append({k: observation[k] for k in
                                     ("source_id", "title", "url", "sha256", "truncated")})
    return finish("budget_exhausted")
