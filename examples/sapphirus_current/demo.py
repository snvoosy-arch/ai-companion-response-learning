"""New hand-authored fixtures; no model, network, persistence or game execution."""
import json

from .contracts import advance_chat, completion_allowed, evidence_card, validate_proposal


def main():
    proposal = validate_proposal(
        {"decision": "tool_call", "tool": "collect_logs", "arguments": {"count": 2}},
        {"collect_logs"},
    )
    receipt = {"task_id": "demo-task", "status": "partial", "verified_count": 1,
               "completion_verified": False, "evidence_id": "demo-observation"}
    current = {"run": "demo-chat-old", "sha256": "a" * 64, "revision": 1}
    finished = {"run": "demo-chat-next", "sha256": "b" * 64,
                "parent_sha256": "a" * 64, "role": "chat", "status": "complete",
                "weights_verified": True, "weights_changed": True}
    print(json.dumps({"mode": "synthetic_cpu_fixture_not_model_output",
                      "proposal": proposal, "executed": False,
                      "receipt_card": evidence_card(receipt),
                      "completion_allowed": completion_allowed("demo-task", 2, receipt),
                      "next_selection": advance_chat(current, finished, "a" * 64),
                      "model_loaded": False, "registry_written": False}, indent=2))


if __name__ == "__main__":
    main()
