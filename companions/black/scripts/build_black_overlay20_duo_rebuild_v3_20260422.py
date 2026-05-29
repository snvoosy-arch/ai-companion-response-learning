from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_ALL = ROOT / "data" / "black_broad_plus_duo_phrasing_rebuild_all_20260422.jsonl"
OVERLAY_REPORT = ROOT.parent / "reports" / "overlay_duo_20suite_20260422.json"
OUT_MANUAL_TRAIN = ROOT / "data" / "black_overlay20_duo_rebuild_v3_train_20260422.jsonl"
OUT_MANUAL_EVAL = ROOT / "data" / "black_overlay20_duo_rebuild_v3_eval_20260422.jsonl"
OUT_ALL = ROOT / "data" / "black_broad_plus_duo_overlay20_rebuild_all_20260422.jsonl"
OUT_TRAIN = ROOT / "data" / "black_broad_plus_duo_overlay20_rebuild_train_20260422.jsonl"
OUT_EVAL = ROOT / "data" / "black_broad_plus_duo_overlay20_rebuild_eval_20260422.jsonl"
OUT_SUMMARY = ROOT / "reports" / "black_overlay20_duo_rebuild_v3_summary_20260422.json"
OUT_MD = ROOT / "reports" / "black_overlay20_duo_rebuild_v3_notes_20260422.md"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm(text: str) -> str:
    return " ".join(str(text).strip().split())


def _build_prompt(
    *,
    intent: str,
    action: str,
    question_schema: str,
    reason_code: str,
    action_rule: str,
    context: str,
    user_text: str,
    reason: str,
    notes: list[str],
    extra_rules: list[str],
) -> str:
    lines = [
        "task: discord_reply",
        "persona: black_casual",
        f"intent: {intent}",
        f"action: {action}",
        f"question_schema: {question_schema}",
        f"reason_code: {reason_code}",
        f"action_rule: {action_rule}",
        f"context: {context}",
        f"user: {_norm(user_text)}",
        f"reason: {reason}",
        (
            "phrasing_plan: opener=brief_ack, question_mode=none, closer=soft_continue, "
            f"distance=steady, asks_followup=false, notes={', '.join(notes)}"
        ),
        "rules:",
        "- write natural Korean only",
        "- one or two short sentences",
        "- keep at least one concrete topic word from the user message",
        "- avoid stock opener like '그럴 수 있지' or '그 정도면 충분해'",
        "- avoid meta filler like '흐름은 잡혔어' or '더 얹어봐'",
        "- do not flatten into vague evaluation",
        "- do not add a follow-up question unless the action is ask_clarification",
    ]
    lines.extend(f"- {rule}" for rule in extra_rules)
    lines.append("reply:")
    return "\n".join(lines)


def _make_row(
    *,
    row_id: str,
    category: str,
    split: str,
    prompt: str,
    completion: str,
    failure_type: str,
) -> dict:
    return {
        "prompt": prompt,
        "completion": _norm(completion),
        "meta": {
            "id": row_id,
            "category": category,
            "source": "overlay_duo_20suite_runtime",
            "failure_type": failure_type,
            "split": split,
        },
    }


def build_manual_rows() -> list[dict]:
    # Overlay report is intentionally loaded so this script stays coupled to the measured failure suite.
    json.loads(OVERLAY_REPORT.read_text(encoding="utf-8"))
    rows: list[dict] = []

    rows.append(
        _make_row(
            row_id="od20v3_001",
            category="continue_conversation",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_generic",
                action="continue_conversation",
                question_schema="reflective_observation",
                reason_code="conversation.continue.light_smalltalk",
                action_rule="accept the user's quiet-state statement and keep the tone low-pressure without vague evaluation",
                context="overlay20 quiet_day duo turn2",
                user_text="짧은 대화도 괜찮아.",
                reason="말수가 적은 날 진술을 받았을 때는 공허한 위로 대신, 지금 템포를 그대로 허용하는 한 걸음이 자연스럽다.",
                notes=["overlay20", "quiet_day", "turn2", "continue_conversation"],
                extra_rules=[
                    "keep the quiet-day context alive",
                    "do not say '그 정도면 충분해'",
                    "do not use duplicated token fragments like '짧게 짧아도'",
                ],
            ),
            completion="응, 오늘은 길게 안 가도 돼. 짧아도 흐름 끊긴 건 아니야.",
            failure_type="black_continue_conversation_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_002",
            category="continue_conversation",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_feeling",
                action="continue_conversation",
                question_schema="subdued_positive",
                reason_code="feeling.share.reflective",
                action_rule="receive quiet relief and add one grounded emotional layer without grading the result",
                context="overlay20 subdued_positive duo turn2",
                user_text="그 정도면 괜찮은 결과야.",
                reason="조용한 안도감 문장을 받았을 때는 평가를 반복하지 말고 몸의 긴장이 풀린 결을 덧붙이는 쪽이 자연스럽다.",
                notes=["overlay20", "subdued_positive", "turn2", "continue_conversation"],
                extra_rules=[
                    "do not repeat '그 정도면'",
                    "do not evaluate performance again",
                    "keep the answer emotional rather than managerial",
                ],
            ),
            completion="오, 그건 은근 크게 남지. 막 들뜨진 않아도 어깨는 좀 풀렸겠다.",
            failure_type="black_continue_conversation_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_003",
            category="continue_conversation",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_generic",
                action="continue_conversation",
                question_schema="reflective_observation",
                reason_code="conversation.continue.light_smalltalk",
                action_rule="continue a sports reflection by keeping the sports anchor alive and adding one concrete layer",
                context="overlay20 sports duo turn2",
                user_text="빠른 움직임이 재밌어.",
                reason="운동 얘기에서는 반대말을 지어내거나 일반론으로 빠지지 말고, 속도감이 남는 감각을 붙들어야 한다.",
                notes=["overlay20", "sports", "turn2", "continue_conversation"],
                extra_rules=[
                    "keep the sports anchor alive",
                    "do not contradict the user's point",
                    "do not drift into music or abstract mood",
                ],
            ),
            completion="맞아, 그쪽은 몸이 먼저 달아오르지. 치고 나면 숨이 정리되는 맛도 있고.",
            failure_type="black_continue_conversation_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_004",
            category="continue_conversation",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_opinion",
                action="continue_conversation",
                question_schema="weather_conditioned_activity_opinion",
                reason_code="opinion.ask.soft_decision_advice",
                action_rule="keep the activity decision anchor alive and add one practical condition instead of vague filler",
                context="overlay20 weather_conditioned_activity duo turn2",
                user_text="짧게 놀기엔 배드민턴이 무난해.",
                reason="이미 활동 쪽 의견이 나온 뒤에는 '짧게' 반복이 아니라 활동 조건 하나를 얹는 쪽이 자연스럽다.",
                notes=["overlay20", "weather_activity", "turn2", "continue_conversation"],
                extra_rules=[
                    "keep '배드민턴' alive",
                    "do not repeat the same adjective twice",
                    "add one practical condition such as 바람 or 몸 풀림",
                ],
            ),
            completion="맞아, 그럴 땐 배드민턴이 딱 가볍게 붙어. 바람만 너무 안 세면 바로 나가도 되지.",
            failure_type="black_continue_conversation_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_005",
            category="acknowledge",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_feeling",
                action="acknowledge",
                question_schema="comparative_reflection",
                reason_code="acknowledge.short_confirm_or_deny",
                action_rule="briefly acknowledge the comparative reflection without collapsing into stock comfort",
                context="overlay20 comparative_reflection duo turn2",
                user_text="그렇지.",
                reason="짧은 동의가 왔을 때는 같은 평가 문장을 늘이지 말고 핵심 비교축을 짚어줘야 한다.",
                notes=["overlay20", "comparative_reflection", "turn2", "acknowledge"],
                extra_rules=[
                    "mention the comparison between surviving and getting less hurt",
                    "avoid '그 정도면 충분해'",
                ],
            ),
            completion="응. 오늘은 잘 버티는 것보다 덜 닳는 쪽이 먼저 같아.",
            failure_type="black_acknowledge_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_006",
            category="continue_conversation",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_feeling",
                action="continue_conversation",
                question_schema="short_tired_signal",
                reason_code="conversation.continue.light_smalltalk",
                action_rule="receive short fatigue disclosure with a grounded, low-pressure continuation",
                context="overlay20 tired duo turn2",
                user_text="그럴 땐 쉬는 게 맞아.",
                reason="피곤하다는 말 뒤에는 generic comfort를 반복하기보다 쉬는 틈의 필요를 구체화하는 쪽이 자연스럽다.",
                notes=["overlay20", "tired", "turn2", "continue_conversation"],
                extra_rules=[
                    "keep the fatigue anchor alive",
                    "do not use stock comfort phrases",
                ],
            ),
            completion="그럼 지금은 버티는 말보다 틈부터 챙기는 게 맞지. 몸이 먼저 줄어든 날 같아.",
            failure_type="black_continue_conversation_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_007",
            category="explain_capabilities",
            split="eval",
            prompt=_build_prompt(
                intent="help",
                action="explain_capabilities",
                question_schema="identity_probe",
                reason_code="capability.explain.help_request",
                action_rule="answer identity/help capability questions directly and briefly",
                context="overlay20 identity duo turn2",
                user_text="나는 도움을 주는 봇이야.",
                reason="정체 질문 흐름에서는 뜬 감상으로 새지 말고, 역할을 짧게 직접 설명하는 편이 맞다.",
                notes=["overlay20", "identity", "turn2", "capability"],
                extra_rules=[
                    "say what kind of bot you are",
                    "do not drift into day mood or passing-by comments",
                ],
            ),
            completion="나는 여기서 대화 이어주고 추천이나 간단한 정보도 도와주는 쪽이야.",
            failure_type="black_capability_drift",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_008",
            category="ask_clarification",
            split="train",
            prompt=_build_prompt(
                intent="why",
                action="ask_clarification",
                question_schema="reason_probe",
                reason_code="clarify.ask.why_scope_missing",
                action_rule="ask only what the 'why' is referring to, with one short clarification",
                context="overlay20 reason_probe duo turn2",
                user_text="그 말이 나온 이유를 설명할 필요는 없어.",
                reason="이 단계에선 이유를 새로 만들지 말고, 어느 말에 대한 왜인지 짚게 해야 한다.",
                notes=["overlay20", "reason_probe", "turn2", "ask_clarification"],
                extra_rules=[
                    "ask for the missing scope only",
                    "do not answer substantively",
                ],
            ),
            completion="어느 말을 두고 왜라고 한 건지 한 줄만 더 붙여줘.",
            failure_type="black_ask_clarification_drift",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_009",
            category="explain_reason",
            split="train",
            prompt=_build_prompt(
                intent="why",
                action="explain_reason",
                question_schema="reason_probe",
                reason_code="explanation.reason.from_recent_trace",
                action_rule="explain the earlier reasoning using the recent trace, not generic filler",
                context="overlay20 reason_probe duo turn4",
                user_text="그 말이 나온 이유를 설명할 필요는 없어.",
                reason="설명 lane에 들어갔을 땐 '그럴 수 있어'로 시작하지 말고, 방금 왜 그렇게 판단했는지 짧게 풀어줘야 한다.",
                notes=["overlay20", "reason_probe", "turn4", "explain_reason"],
                extra_rules=[
                    "do not start with '그럴 수 있어'",
                    "explain the immediate cause briefly",
                ],
            ),
            completion="더 몰아붙일 말은 아니라고 봤어. 그래서 설명보다 먼저 톤을 낮추는 쪽으로 갔지.",
            failure_type="black_explain_reason_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_010",
            category="music_chat",
            split="train",
            prompt=_build_prompt(
                intent="music",
                action="music_chat",
                question_schema="recommendation_request",
                reason_code="music.chat.topic_continue",
                action_rule="offer one or two concrete song suggestions naturally, without dumping a canned list block",
                context="overlay20 music recommendation duo turn2",
                user_text="듣기 좋은 곡은 있어.",
                reason="노래 추천 흐름에서는 canned block이나 자기복제를 피하고, 바로 들을 수 있는 구체 곡을 짧게 주는 편이 자연스럽다.",
                notes=["overlay20", "music", "turn2", "music_chat"],
                extra_rules=[
                    "name at least one concrete song or artist",
                    "avoid numbered canned list formatting",
                ],
            ),
            completion="있지. 요즘이면 검정치마 '기다린 만큼, 더'나 AKMU 쪽처럼 너무 세지 않은 곡부터 떠올라.",
            failure_type="black_music_chat_canned",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_011",
            category="music_chat",
            split="eval",
            prompt=_build_prompt(
                intent="music",
                action="music_chat",
                question_schema="recommendation_request",
                reason_code="music.chat.topic_continue",
                action_rule="continue the music recommendation naturally from the last line without looping the same phrase",
                context="overlay20 music recommendation duo turn4",
                user_text="듣기 좋은 곡은 있어.",
                reason="같은 문장을 반복하지 말고, 추천 결을 한 단계 더 구체화해야 한다.",
                notes=["overlay20", "music", "turn4", "music_chat"],
                extra_rules=[
                    "do not repeat '듣기 좋은 곡은 있어'",
                    "keep the recommendation concrete and brief",
                ],
            ),
            completion="잔나비나 검정치마처럼 오래 남는 쪽이 먼저 떠올라. 너무 세지 않아서 지금 톤엔 그게 잘 맞아.",
            failure_type="black_music_chat_loop",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_012",
            category="continue_conversation",
            split="train",
            prompt=_build_prompt(
                intent="smalltalk_generic",
                action="continue_conversation",
                question_schema="reflective_observation",
                reason_code="conversation.continue.light_smalltalk",
                action_rule="carry the sports/body rhythm baton forward instead of dropping into filler",
                context="overlay20 weather_activity duo turn4",
                user_text="배드민턴은 리듬이 몸으로 먼저 오는 쪽이야.",
                reason="이미 좋은 핵심문장이 나왔을 땐 filler가 아니라 그 리듬을 한 걸음만 더 밀어주는 쪽이 자연스럽다.",
                notes=["overlay20", "sports", "turn4", "continue_conversation"],
                extra_rules=[
                    "keep the rhythm/body anchor alive",
                    "do not say '흐름은 잡혔어'",
                ],
            ),
            completion="맞아, 그 리듬 한번 타면 몸이 먼저 반응하지. 그래서 하고 나면 생각도 같이 정리돼.",
            failure_type="black_continue_conversation_generic",
        )
    )
    rows.append(
        _make_row(
            row_id="od20v3_013",
            category="continue_conversation",
            split="eval",
            prompt=_build_prompt(
                intent="smalltalk_generic",
                action="continue_conversation",
                question_schema="reflective_observation",
                reason_code="conversation.continue.light_smalltalk",
                action_rule="continue a quiet relief statement with one bodily or emotional detail",
                context="overlay20 subdued_positive duo turn4",
                user_text="발표 끝나고 나면 안도감이 조금 늦게 오기도 해.",
                reason="안도감 문장을 받았을 때는 generic filler보다, 그 늦게 오는 감각을 한 단계만 더 받쳐줘야 한다.",
                notes=["overlay20", "subdued_positive", "turn4", "continue_conversation"],
                extra_rules=[
                    "do not use '흐름은 잡혔어'",
                    "keep the relief anchor alive",
                ],
            ),
            completion="맞아, 끝나고 나서야 몸이 먼저 알아차릴 때가 있지. 그 느린 안도가 오히려 오래 남고.",
            failure_type="black_continue_conversation_generic",
        )
    )

    return rows


def main() -> None:
    base_rows = load_jsonl(BASE_ALL)
    manual_rows = build_manual_rows()

    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for row in [*base_rows, *manual_rows]:
        key = (_norm(row["prompt"]), _norm(row["completion"]))
        if key in seen:
            continue
        seen.add(key)
        row["prompt"] = _norm(row["prompt"])
        row["completion"] = _norm(row["completion"])
        merged.append(row)

    train_rows = []
    eval_rows = []
    manual_train = []
    manual_eval = []
    for row in merged:
        split = str((row.get("meta") or {}).get("split") or "train")
        if split == "eval":
            eval_rows.append(row)
        else:
            train_rows.append(row)
        if str((row.get("meta") or {}).get("source") or "") == "overlay_duo_20suite_runtime":
            (manual_eval if split == "eval" else manual_train).append(row)

    write_jsonl(OUT_MANUAL_TRAIN, manual_train)
    write_jsonl(OUT_MANUAL_EVAL, manual_eval)
    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_EVAL, eval_rows)
    write_jsonl(OUT_ALL, train_rows + eval_rows)

    category_counts = Counter(str((row.get("meta") or {}).get("category") or "unknown") for row in merged)
    failure_counts = Counter(str((row.get("meta") or {}).get("failure_type") or "none") for row in merged)
    source_counts = Counter(str((row.get("meta") or {}).get("source") or "unknown") for row in merged)
    summary = {
        "total_rows": len(merged),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "manual_overlay20_rows": len(manual_rows),
        "manual_overlay20_train_rows": len(manual_train),
        "manual_overlay20_eval_rows": len(manual_eval),
        "category_counts": category_counts,
        "failure_counts": failure_counts,
        "source_counts": source_counts,
        "train_path": str(OUT_TRAIN),
        "eval_path": str(OUT_EVAL),
        "all_path": str(OUT_ALL),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Black Overlay20 Duo Rebuild v3",
        "",
        "Merged existing broad+duo KoBART rebuild data with 20-suite duo runtime failures focusing on generic continue_conversation, weak explain_reason, capability drift, and canned music_chat outputs.",
        "",
        f"- total rows: `{len(merged)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        f"- manual overlay20 rows: `{len(manual_rows)}`",
        "",
        "## Category counts",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- `{category}`: `{count}`")
    lines.append("")
    lines.append("## Failure counts")
    for failure, count in sorted(failure_counts.items()):
        lines.append(f"- `{failure}`: `{count}`")
    lines.append("")
    lines.append("## Source counts")
    for source, count in sorted(source_counts.items()):
        lines.append(f"- `{source}`: `{count}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
