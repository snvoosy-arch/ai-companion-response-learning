"""Bounded complete-pair selection; conversational meaning belongs to the model."""

WINDOW_SECONDS = 1800
MAX_CONTEXT_MESSAGES = 4


def select_context(rows: list, *, now: float, reply_to=None) -> list:
    """Select fresh complete pairs from chronological rows scoped by the caller.

    Explicit missing/orphan references never fall back to another conversation.
    Returned records retain IDs for deletion/edit dependency checks.
    """
    rows = [r for r in rows if r.get('role') in ('user', 'assistant')
            and isinstance(r.get('content'), str) and r['content'].strip()
            and not r.get('deleted', False) and now - WINDOW_SECONDS <= r['created'] <= now]
    by_id = {str(r['id']): r for r in rows}
    if len(by_id) != len(rows):
        return []
    pairs = []
    for answer in rows:
        user = by_id.get(str(answer.get('reply_to')))
        if (answer['role'] == 'assistant' and user and user['role'] == 'user'
                and answer['created'] >= user['created']):
            pairs.append((user, answer))
    pairs.sort(key=lambda p: (p[1]['created'], str(p[1]['id'])))
    if reply_to is not None:
        pairs = [p for p in pairs if str(reply_to) in (str(p[0]['id']), str(p[1]['id']))][-1:]
    elif pairs:
        latest = max(rows, key=lambda r: (r['created'], r['role'] == 'assistant', str(r['id'])))
        if latest['id'] != pairs[-1][1]['id']:
            return []  # Do not skip a newer unanswered input to revive an old topic.
        pairs = pairs[-MAX_CONTEXT_MESSAGES // 2:]
    return [dict(row) for pair in pairs for row in pair]
