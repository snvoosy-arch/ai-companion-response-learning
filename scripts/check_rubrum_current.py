"""Validate public Rubrum snapshot structure and its visible numeric claims.

Source artifact hashes identify private versions; this checker cannot read or
independently verify unpublished training/evaluation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[a-f0-9]{64}")


def plain(text):
    return text.replace("*", "").replace("`", "")


def validate_repository(root: Path) -> list[str]:
    errors = []

    def need(condition, label):
        if not condition:
            errors.append(label)

    def integer(value, label, upper=None):
        valid = type(value) is int and value >= 0
        need(valid, label + " must be a nonnegative integer")
        if valid and upper is not None:
            need(value <= upper, label + " exceeds denominator")
        return value if valid else 0

    def ratio(row, label, count="count", correct="correct"):
        total = integer(row[count], label + "." + count)
        hits = integer(row[correct], label + "." + correct, total)
        if "false_accepts" in row:
            integer(row["false_accepts"], label + ".false_accepts", total - hits)
        return total, hits

    def match(text, pattern, expected, label, convert=int):
        found = re.search(pattern, plain(text))
        if found is None:
            errors.append("document claim missing: " + label)
            return
        try:
            actual = tuple(convert(value) for value in found.groups())
        except ValueError:
            errors.append("document claim invalid: " + label)
            return
        need(actual == tuple(expected), "document claim mismatch: " + label)

    try:
        root = Path(root)
        directory = root / "evidence/model-training"
        aggregate = json.loads((directory / "current-summary.json").read_text(encoding="utf-8"))
        ref = aggregate["rubrum"]["latest_dialogue_transition"]
        name = ref["summary"]
        if not isinstance(name, str) or re.fullmatch(r"rubrum-dialogue-\d{8}\.json", name) is None:
            return ["Unsafe or unsupported Rubrum summary reference"]
        path = directory / name
        if path.resolve().parent != directory.resolve():
            return ["Rubrum summary reference escapes evidence directory"]
        raw = path.read_bytes()
        data = json.loads(raw)
        need(
            set(data)
            == {
                "schema_version",
                "as_of",
                "evidence_level",
                "scope",
                "architecture",
                "chat_adaptation",
                "dialogue_heads",
                "record_relevance",
                "response_grounding",
                "actual_model_offline_integration",
                "response_audit",
                "related_private_cpu_checks",
                "actual_discord_replay",
                "known_limits",
                "source_artifacts",
            },
            "Unexpected public snapshot fields; raw conversation rows are not allowed",
        )
        need(
            isinstance(ref.get("summary_sha256"), str)
            and ref.get("summary_sha256")
            == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
            "Rubrum summary_sha256 binding mismatch",
        )
        need(
            data["schema_version"] == "rubrum.dialogue_research_summary.v1",
            "Unknown Rubrum snapshot schema",
        )
        need(
            aggregate["rubrum"]["as_of"] == ref["as_of"] == data["as_of"],
            "Rubrum snapshot date binding mismatch",
        )
        chat = data["chat_adaptation"]
        unique = integer(chat["unique_training_rows"], "unique_training_rows")
        need(
            unique
            == chat["structured_response_train_rows"] + chat["existing_daily_replay_train_rows"],
            "Unique training split mismatch",
        )
        for key in (
            "structured_response_train_rows",
            "existing_daily_replay_train_rows",
            "structured_response_eval_rows",
            "independently_corrupted_training_canvases",
            "optimizer_updates",
            "validation_questions",
            "fixed_validation_canvases",
        ):
            integer(chat[key], "chat_adaptation." + key)
        need(
            chat["validation_questions"] == chat["structured_response_eval_rows"],
            "Validation split mismatch",
        )
        need(
            chat["fixed_validation_canvases"]
            == chat["validation_questions"] * len(chat["validation_noise_levels"]),
            "Validation canvas mismatch",
        )
        need(
            chat["independently_corrupted_training_canvases"]
            == unique * chat["training_corruptions_per_row"],
            "Training canvas mismatch",
        )
        need(ref["chat_child_run_id"] == chat["run_id"], "Chat child binding mismatch")
        for key in ("adapter_sha256", "sc_sha256"):
            need(
                isinstance(chat[key], str) and SHA.fullmatch(chat[key]) is not None,
                "Invalid " + key,
            )
        for group in ("validation_loss_before", "validation_loss_after"):
            for label, value in chat[group].items():
                need(
                    type(value) in (float, int) and math.isfinite(value) and value >= 0,
                    group + "." + label + " must be finite",
                )

        heads = data["dialogue_heads"]
        fresh = heads["new_combination"]
        total = integer(fresh["questions"], "new_combination.questions")
        need(
            total == heads["new_combination_evaluation_rows"],
            "Dialogue evaluation denominator mismatch",
        )
        for key in (
            "route_exact",
            "speaker_exact",
            "time_kind_exact",
            "context_set_exact",
            "query_char_span_exact",
            "time_char_span_exact",
            "positive_query_questions",
        ):
            integer(fresh[key], "new_combination." + key, total)
        integer(
            fresh["positive_query_span_exact"],
            "positive_query_span_exact",
            fresh["positive_query_questions"],
        )
        ratio(
            heads["context_available_subset"],
            "context_available_subset",
            "questions",
            "exact",
        )
        ratio(
            heads["explicit_reply_subset"],
            "explicit_reply_subset",
            "questions",
            "exact",
        )
        need(
            heads["context_available_subset"]["questions"] <= total,
            "Context subset exceeds full evaluation",
        )
        need(
            heads["explicit_reply_subset"]["questions"]
            <= heads["context_available_subset"]["questions"],
            "Reply subset exceeds context subset",
        )
        relevance = data["record_relevance"]
        ratio(relevance["evaluation"], "record_relevance.evaluation")
        ratio(relevance["historical"], "record_relevance.historical")
        guard = data["response_grounding"]["evaluation"]
        ratio(guard, "response_grounding.evaluation")
        checks = data["related_private_cpu_checks"]
        integer(checks["tests"], "related_private_cpu_checks.tests")
        need(
            ref["related_private_tests"] == checks["tests"],
            "Private CPU count binding mismatch",
        )
        need(
            checks.get("passed") is True
            and checks["failures"] == 0
            and checks["errors"] == 0
            and checks["compileall"] is True,
            "Private verification is not passing",
        )
        live = data["actual_discord_replay"]
        need(
            set(live)
            <= {
                "cases",
                "delivered",
                "candidate_audit_matches",
                "delivery_outcome_matches",
                "inputs",
                "route",
                "first_candidate_passes",
                "grounding_retry_observed",
                "native_user_gateway_ingress_tested",
                "production_memory_unchanged",
                "response_quality",
                "service_running_after_test",
                "private_text_or_ids_published",
                "test_mode",
                "revision_scope",
            },
            "Unexpected private replay details in public snapshot",
        )
        cases = integer(live["cases"], "actual_discord_replay.cases")
        for key in (
            "delivered",
            "candidate_audit_matches",
            "delivery_outcome_matches",
            "first_candidate_passes",
        ):
            integer(live[key], "actual_discord_replay." + key, cases)
        need(
            ref["private_channel_replays"] == cases,
            "Live replay count binding mismatch",
        )
        need(
            ref.get("native_user_ingress_tested") is live["native_user_gateway_ingress_tested"],
            "Aggregate native user ingress scope mismatch",
        )
        need(
            ref.get("broad_dialogue_quality_proven") is False,
            "Aggregate cannot claim broad dialogue quality from these diagnostics",
        )
        need(
            live["delivered"]
            == live["candidate_audit_matches"]
            == live["delivery_outcome_matches"],
            "document combined replay audit claim mismatch",
        )
        need(live["first_candidate_passes"] == cases, "document all-first-candidate claim mismatch")
        need(
            live["test_mode"] == "historical_replay"
            and live["native_user_gateway_ingress_tested"] is False,
            "Historical replay cannot claim native user ingress",
        )
        need(
            live["route"] == "CHAT" and live["grounding_retry_observed"] is False,
            "CHAT-only replay cannot claim grounding retries",
        )
        need(
            live["private_text_or_ids_published"] is False,
            "Private replay publication scope changed",
        )
        architecture = data["architecture"]
        need(
            architecture["grounding_routes"] == ["MEMORY_QUERY", "CLARIFY"]
            and architecture["grounding_general_chat"] is False,
            "Response grounding scope does not match the documented path",
        )
        need(
            architecture["shared_backbone"] is True and architecture["extra_bert_model"] is False,
            "Shared backbone scope mismatch",
        )
        need(
            data["actual_model_offline_integration"]["meaning_correct_count"] is None,
            "Fixture delivery cannot become meaning accuracy",
        )
        required = {
            "chat_training",
            "validation_before",
            "validation_after",
            "dialogue_heads",
            "record_relevance",
            "response_grounding",
            "offline_integration",
            "private_cpu_checks",
            "private_channel_readback",
        }
        need(set(data["source_artifacts"]) == required, "source_artifacts roles mismatch")
        for key, source in data["source_artifacts"].items():
            need(
                set(source) == {"artifact", "sha256", "scope"},
                "Unexpected source_artifacts fields",
            )
            need(
                isinstance(source["sha256"], str) and SHA.fullmatch(source["sha256"]) is not None,
                "Invalid source_artifacts." + key + " hash",
            )
            need(
                isinstance(source["artifact"], str)
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.json", source["artifact"]) is not None,
                "Unsafe source_artifacts." + key + " identifier",
            )

        document = plain((root / "docs/rubrum-model-led-dialogue.md").read_text(encoding="utf-8"))
        rows = {}
        for line in document.splitlines():
            cells = [re.sub(r"\s+", " ", cell.strip()) for cell in line.split("|")]
            if len(cells) >= 4 and cells[0] == "":
                rows[cells[1]] = cells[2:-1]
        expected = {
            "경로 전체 일치": (fresh["route_exact"], total),
            "발화 출처": (fresh["speaker_exact"], total),
            "기간 종류": (fresh["time_kind_exact"], total),
            "문맥 집합 전체 일치": (fresh["context_set_exact"], total),
            "문맥 후보가 주어진 부분의 전체 일치": (
                heads["context_available_subset"]["exact"],
                heads["context_available_subset"]["questions"],
            ),
            "명시적 답글 대상 대조 부분": (
                heads["explicit_reply_subset"]["exact"],
                heads["explicit_reply_subset"]["questions"],
            ),
            "검색 문자 범위 전체 일치": (fresh["query_char_span_exact"], total),
            "검색 범위가 있는 부분의 전체 일치": (
                fresh["positive_query_span_exact"],
                fresh["positive_query_questions"],
            ),
        }
        for label, (hits, count) in expected.items():
            need(
                rows.get(label) == [f"{hits}/{count}"],
                "document table mismatch: " + label,
            )
        for label, key in [
            ("본문", "content"),
            ("실제 턴 종료", "terminator"),
            ("EOS 채움 tail", "tail"),
        ]:
            values = rows.get(label, [])
            need(
                len(values) == 2
                and [float(x) for x in values]
                == [
                    chat["validation_loss_before"][key],
                    chat["validation_loss_after"][key],
                ],
                "document loss mismatch: " + label,
            )
        match(
            document,
            r"생성 근거 대조는\s*(\d+)/(\d+),\s*오수락\s*(\d+)건",
            [guard["correct"], guard["count"], guard["false_accepts"]],
            "grounding score",
        )
        match(
            document,
            r"관련 비공개 CPU 테스트\s*(\d+)개",
            [checks["tests"]],
            "private CPU tests",
        )
        match(
            document,
            r"기록 관련성 보완 후 보류 평가가\s*(\d+)/(\d+)\s*→\s*(\d+)/(\d+)"
            r"[\s\S]*?이후 오수락은\s*(\d+)건",
            [
                relevance["before_evaluation_correct"],
                relevance["before_evaluation_questions"],
                relevance["evaluation"]["correct"],
                relevance["evaluation"]["count"],
                relevance["evaluation"]["false_accepts"],
            ],
            "record relevance",
        )
        match(
            document,
            r"(\d+)건 모두 일반 CHAT의 첫 후보였습니다",
            [live["first_candidate_passes"]],
            "first candidate scope",
        )
        match(
            document,
            r"(\d+)/(\d+) 본문·답글 대상·후보 원문 로그·전송 결과 해시 일치",
            [live["delivered"], cases],
            "Discord replay delivery",
        )
        readme = plain((root / "README.md").read_text(encoding="utf-8").split("<details>", 1)[0])
        match(
            readme,
            r"고유 학습 대화\s*(\d+)건·오염 canvas\s*(\d+)개·(\d+) updates",
            [
                unique,
                chat["independently_corrupted_training_canvases"],
                chat["optimizer_updates"],
            ],
            "README training",
        )
        match(
            readme,
            r"경로 판정은\s*(\d+)/(\d+)",
            [fresh["route_exact"], total],
            "README route",
        )
        match(
            readme,
            r"생성 근거 대조는\s*(\d+)/(\d+)\(오수락\s*(\d+)건\)",
            [guard["correct"], guard["count"], guard["false_accepts"]],
            "README grounding",
        )
        training = (
            (root / "docs/current-model-training.md")
            .read_text(encoding="utf-8")
            .split("### 후속 대화 판단·근거 응답 적응", 1)[1]
            .split("\n## Sapphirus", 1)[0]
        )
        match(training, r"고유\s*(\d+)건", [unique], "training document unique rows")
        match(
            training,
            r"(\d+) canvas·(\d+) updates",
            [
                chat["independently_corrupted_training_canvases"],
                chat["optimizer_updates"],
            ],
            "training document presentations",
        )
        match(
            training,
            r"근거 대조\s*(\d+)쌍의\s*(\d+)/(\d+)",
            [guard["count"], guard["correct"], guard["count"]],
            "training document grounding",
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append("Invalid public Rubrum schema/document: " + type(exc).__name__)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    errors = validate_repository(parser.parse_args().root)
    for error in errors:
        print("ERROR: " + error)
    if not errors:
        print(
            "Rubrum current checks passed: schema, denominators, document claims, "
            "snapshot binding and replay scope"
        )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
