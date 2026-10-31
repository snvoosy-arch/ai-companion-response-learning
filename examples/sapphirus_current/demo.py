"""Hand-authored CPU fixtures; no model, network, persistence or game execution."""
import argparse
import json

from .contracts import advance_chat, completion_allowed, evidence_card, validate_proposal
from .context import select_context


def chat_example():
    rows = [
        dict(id='expired-u', role='user', content='오래된 문맥', created=10),
        dict(id='expired-a', role='assistant', content='오래된 앞 답변', created=11, reply_to='expired-u'),
        dict(id='u1', role='user', content='책을 창가에 놓았어.', created=1900),
        dict(id='a1', role='assistant', content='빛이 들어오는 자리네.', created=1901, reply_to='u1'),
        dict(id='u2', role='user', content='다른 얘기인데 수첩을 새로 샀어.', created=1902),
        dict(id='a2', role='assistant', content='어디에 쓸 생각이야?', created=1903, reply_to='u2'),
    ]
    current = dict(role='user', content='그건 왜 그래?')

    def scenario(source, reply_to=None):
        selected = select_context(source, now=2000, reply_to=reply_to)
        return dict(selected_ids=[row['id'] for row in selected], messages=[
            dict(role=row['role'], content=row['content']) for row in selected]+[dict(current)])

    deleted = [dict(row, deleted=True) if row['id'] == 'u1' else dict(row) for row in rows]
    edited = [dict(row) for row in rows]
    # The caller owns revision/dependency tracking; the selector does not infer edits.
    for row in edited:
        if row['id'] == 'u1':
            row['content'] = '책을 책장으로 옮겼어.'
        elif row['id'] == 'a1':
            row['deleted'] = True  # caller invalidated the reply based on the old input
    return dict(mode='synthetic_cpu_context_selection_not_model_output',
        caller_scope='one synthetic user/channel; edit dependencies invalidated by caller',
        scenarios=dict(recent=scenario(rows), explicit_reply=scenario(rows, 'a1'),
            missing_reply=scenario(rows, 'outside-scope-id'), deleted_reply=scenario(deleted, 'a1'),
            edited_reply=scenario(edited, 'a1')),
        model_loaded=False, model_response_generated=False, database_written=False)


def contract_example():
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
    return {"mode": "synthetic_cpu_fixture_not_model_output",
            "proposal": proposal, "executed": False,
            "receipt_card": evidence_card(receipt),
            "completion_allowed": completion_allowed("demo-task", 2, receipt),
            "next_selection": advance_chat(current, finished, "a" * 64),
            "model_loaded": False, "registry_written": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sapphirus standard-library CPU reference')
    parser.add_argument('--contracts', action='store_true', help='show the previous contract-only fixture')
    args = parser.parse_args(argv)
    result = contract_example() if args.contracts else chat_example()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
