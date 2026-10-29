"""Small reference contracts, not a model, training loop or game executor.

Inputs called receipts/proofs must come from a trusted external verifier. This
module checks structure and consistency; it cannot authenticate arbitrary input
or prove that generated natural language is true.
"""
import copy
import re


def validate_proposal(proposal: dict, available: set[str]) -> dict:
    """Return a validated copy. Availability is not permission to execute."""
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be an object")
    if proposal.get("decision") == "no_action" and set(proposal) == {"decision"}:
        return dict(proposal)
    if (set(proposal) != {"decision", "tool", "arguments"}
            or proposal["decision"] != "tool_call"):
        raise ValueError("unsupported proposal shape")
    tool, args = proposal["tool"], proposal["arguments"]
    if not isinstance(tool, str) or tool not in available or not isinstance(args, dict):
        raise ValueError("tool unavailable or arguments invalid")
    if tool == "approach_user":
        if args:
            raise ValueError("approach_user takes no model-supplied target")
    elif tool == "collect_logs":
        if (not {"count"} <= set(args) <= {"count", "kind"}
                or type(args["count"]) is not int or not 1 <= args["count"] <= 3):
            raise ValueError("count must be an integer in 1..3")
        if "kind" in args and (not isinstance(args["kind"], str) or args["kind"] not in {
            "oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove"}):
            raise ValueError("unsupported log kind")
    else:
        raise ValueError("outside this reference slice")
    return copy.deepcopy(proposal)


def completion_allowed(task_id: str, requested: int, receipt: dict) -> bool:
    """Gate a completion event using a trusted task-bound external receipt."""
    return (
        isinstance(task_id, str) and bool(task_id)
        and type(requested) is int and requested > 0
        and isinstance(receipt, dict) and receipt.get("task_id") == task_id
        and receipt.get("status") == "completed"
        and receipt.get("completion_verified") is True
        and type(receipt.get("verified_count")) is int
        and receipt["verified_count"] >= requested
        and isinstance(receipt.get("evidence_id"), str)
        and bool(receipt["evidence_id"].strip())
    )


def evidence_card(receipt: dict) -> dict:
    """A confirmed count is a lower bound, not the physical total."""
    count = receipt.get("verified_count")
    if type(count) is not int or count < 0:
        raise ValueError("invalid confirmed count")
    return {"confirmed_minimum": count, "physical_total": None,
            "completion_verified": receipt.get("completion_verified") is True,
            "status": receipt.get("status")}


def advance_chat(current: dict, finished: dict, expected_parent: str) -> dict:
    """Reference compare-parent transaction over externally verified metadata.

    Returns a new in-memory selection. Real deployment additionally needs file
    hashes, tensor comparisons, a writer lock and an atomic durable registry write.
    """
    def valid_hash(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    if (not valid_hash(expected_parent) or current.get("sha256") != expected_parent
            or finished.get("parent_sha256") != expected_parent
            or finished.get("role") != "chat" or finished.get("status") != "complete"
            or finished.get("weights_verified") is not True
            or finished.get("weights_changed") is not True
            or not valid_hash(finished.get("sha256"))
            or finished["sha256"] == expected_parent
            or not isinstance(finished.get("run"), str) or not finished["run"]
            or type(current.get("revision")) is not int or current["revision"] < 1):
        raise ValueError("unverified Chat training or stale parent")
    return {"run": finished["run"], "sha256": finished["sha256"],
            "parent_sha256": expected_parent, "revision": current["revision"] + 1}
