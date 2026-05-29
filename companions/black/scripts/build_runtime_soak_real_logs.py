from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.config import DEFAULT_STATE_DB_PATH
from predictive_bot.evaluation.runtime_logs import (
    DEFAULT_ANONYMIZATION_SALT,
    DEFAULT_MAX_TURNS_PER_SESSION,
    DEFAULT_MIN_TURNS_PER_SESSION,
    DEFAULT_SESSION_GAP_MINUTES,
    build_runtime_soak_sessions,
    build_runtime_soak_summary,
    load_decision_trace_rows,
    write_sessions_jsonl,
)


DEFAULT_OUTPUT = ROOT / "data" / "runtime_soak_real_logs_eval.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "runtime_soak_real_logs_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실제 decision_trace 로그를 익명화해 runtime soak JSONL로 내보냅니다.")
    parser.add_argument("--source-db", type=Path, default=DEFAULT_STATE_DB_PATH, help="입력 SQLite DB 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="익명화된 soak JSONL 경로")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY, help="요약 JSON 경로")
    parser.add_argument("--session-gap-minutes", type=int, default=DEFAULT_SESSION_GAP_MINUTES)
    parser.add_argument("--max-turns-per-session", type=int, default=DEFAULT_MAX_TURNS_PER_SESSION)
    parser.add_argument("--min-turns-per-session", type=int, default=DEFAULT_MIN_TURNS_PER_SESSION)
    parser.add_argument("--anonymization-salt", default=DEFAULT_ANONYMIZATION_SALT)
    parser.add_argument("--no-redact-text", action="store_true", help="텍스트 redaction 을 끕니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_decision_trace_rows(args.source_db)
    sessions, stats = build_runtime_soak_sessions(
        rows,
        source_db=str(args.source_db),
        anonymization_salt=args.anonymization_salt,
        session_gap_minutes=args.session_gap_minutes,
        max_turns_per_session=args.max_turns_per_session,
        min_turns_per_session=args.min_turns_per_session,
        redact_text=not args.no_redact_text,
    )
    write_sessions_jsonl(args.output, sessions)
    summary = build_runtime_soak_summary(sessions, stats)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
