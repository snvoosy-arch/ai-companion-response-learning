from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.config import DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES, AppConfig  # noqa: E402
from predictive_bot.core.models import WeatherReport  # noqa: E402
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline  # noqa: E402
from predictive_bot.factory import build_engine  # noqa: E402


DATE_STEM = "20260523"
DEFAULT_MODEL_DIR = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_frame_bootstrap_v1"
DEFAULT_OUT_JSON = PROJECT_ROOT / "reports" / f"black_frame_shadow_probe50_{DATE_STEM}.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "reports" / f"black_frame_shadow_probe50_{DATE_STEM}.md"
DEFAULT_PROBE_NAME = f"black_frame_shadow_probe50_{DATE_STEM}"
TRUSTED_AXES = DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES


PROBE_ITEMS: list[dict[str, str]] = [
    {"id": "hc001", "category": "money_practical", "text": "요즘 가스비 너무 올라서 보일러 켜기 무서운데, 춥게 버티는 게 맞는지 현실적으로 말해봐."},
    {"id": "hc002", "category": "money_practical", "text": "월급은 그대로인데 카드값이 자꾸 불어나서 숨 막혀, 오늘 당장 뭘 끊어야 해?"},
    {"id": "hc003", "category": "money_practical", "text": "충동구매 참아야 하는 건 아는데 스트레스 받으면 장바구니부터 보게 돼, 이거 어떻게 막아?"},
    {"id": "hc004", "category": "money_practical", "text": "친구 생일 선물 비싸게 해야 할지 모르겠어, 마음은 있는데 이번 달 진짜 빠듯해."},
    {"id": "hc005", "category": "money_practical", "text": "배달 줄이면 돈 아끼는 건 알겠는데 퇴근하면 요리할 힘이 없어, 현실적인 타협안 줘."},
    {"id": "hc006", "category": "work_action", "text": "새 프로젝트 맡았는데 막막하고 팀장 눈치도 보여, 오늘 첫 단계 딱 하나만 정해줘."},
    {"id": "hc007", "category": "work_action", "text": "회의에서 말실수한 것 같아서 계속 곱씹고 있어, 내일 어떻게 수습하는 게 나아?"},
    {"id": "hc008", "category": "work_action", "text": "일은 쌓였는데 머리가 멈춘 느낌이야, 쉬어야 하는지 밀어붙여야 하는지 판단해줘."},
    {"id": "hc009", "category": "work_action", "text": "상사가 애매하게 던진 일을 내가 다 떠안는 분위기야, 선 긋는 말투 좀 잡아줘."},
    {"id": "hc010", "category": "work_action", "text": "퇴근 후 공부하려고 했는데 매번 뻗어, 의지가 약한 건지 계획이 틀린 건지 봐줘."},
    {"id": "hc011", "category": "emotion_grounding", "text": "괜찮은 척했는데 사실 오늘 하루 종일 서운해서 집중이 안 됐어."},
    {"id": "hc012", "category": "emotion_grounding", "text": "아무 일 아닌 말에 기분이 확 꺼졌는데, 내가 예민한 건지 그냥 지친 건지 모르겠어."},
    {"id": "hc013", "category": "emotion_grounding", "text": "누가 위로해줘도 안 들어오고 그냥 멍해, 지금 뭘 해야 좀 돌아올까?"},
    {"id": "hc014", "category": "emotion_grounding", "text": "오늘은 진짜 별거 아닌 일에도 울컥해서 내가 나한테 질려."},
    {"id": "hc015", "category": "emotion_grounding", "text": "사람 만나는 건 좋은데 끝나고 나면 기가 다 빨려, 내가 이상한 건 아니지?"},
    {"id": "hc016", "category": "relationship_boundary", "text": "친구가 계속 늦게 답장하면서 필요할 때만 찾아, 나도 거리 둬야 하나?"},
    {"id": "hc017", "category": "relationship_boundary", "text": "상대가 미안하다고는 하는데 같은 행동을 반복해, 믿어도 되는지 모르겠어."},
    {"id": "hc018", "category": "relationship_boundary", "text": "내가 서운하다고 말하면 분위기 망칠까 봐 계속 참게 돼, 어떻게 꺼내야 해?"},
    {"id": "hc019", "category": "relationship_boundary", "text": "단톡에서 나만 빼고 약속 잡은 것 같아서 찝찝해, 바로 물어보는 게 나아?"},
    {"id": "hc020", "category": "relationship_boundary", "text": "연락 빈도로 마음을 판단하면 안 되는 거 아는데 자꾸 불안해져."},
    {"id": "hc021", "category": "choice_judgment", "text": "오늘 운동 가는 게 맞아, 아니면 집에서 쉬는 게 맞아? 몸은 무겁고 죄책감은 있어."},
    {"id": "hc022", "category": "choice_judgment", "text": "지금 사과하는 게 맞을까, 감정 가라앉히고 내일 말하는 게 맞을까?"},
    {"id": "hc023", "category": "choice_judgment", "text": "이직 준비를 시작해야 할지 그냥 지금 회사에서 버텨야 할지 감이 안 와."},
    {"id": "hc024", "category": "choice_judgment", "text": "중고로 싸게 살지 새 제품으로 오래 쓸지 고민돼, 돈이랑 안정성 중 뭐가 먼저야?"},
    {"id": "hc025", "category": "choice_judgment", "text": "오늘 약속 나가면 재밌긴 할 텐데 내일 일정이 망가질 것 같아, 뭐가 낫지?"},
    {"id": "hc026", "category": "ai_meta", "text": "너는 생성모델 없이 분류랑 템플릿만으로도 고맥락 대화가 가능하다고 봐? 단점까지 말해봐."},
    {"id": "hc027", "category": "ai_meta", "text": "ModernBERT가 답을 쓰는 게 아니라 프레임을 맞히는 거면, 지금 우리 구조에서 제일 위험한 축이 뭐야?"},
    {"id": "hc028", "category": "ai_meta", "text": "규칙이 너무 많아지면 모델 학습 데이터로는 좋아도 운영 엔진으로는 지저분해지는 거 아니야?"},
    {"id": "hc029", "category": "ai_meta", "text": "draft_frame 정확도가 낮은데 family랑 tone만 믿고 답변을 고르는 게 말이 돼?"},
    {"id": "hc030", "category": "ai_meta", "text": "고맥락을 하려면 raw 문장, compact, 단어 의미, 최근 대화 중 뭐를 제일 먼저 봐야 해?"},
    {"id": "hc031", "category": "logic_compound", "text": "논리적으로는 쉬는 게 맞는데 감정적으로는 뒤처질까 봐 불안해, 오늘 계획 어떻게 잡아야 해?"},
    {"id": "hc032", "category": "logic_compound", "text": "내가 화난 이유를 설명하면 정당화처럼 들릴까 봐 겁나는데, 그래도 말해야 하나?"},
    {"id": "hc033", "category": "logic_compound", "text": "효율만 보면 포기하는 게 맞는데 아쉬움이 커, 이럴 땐 판단 기준을 뭘로 둬?"},
    {"id": "hc034", "category": "logic_compound", "text": "친구 말이 틀린 건 아닌데 방식이 너무 무례했어, 내용과 태도를 분리해서 봐야 해?"},
    {"id": "hc035", "category": "logic_compound", "text": "사실 증거는 없는데 촉이 안 좋아, 이걸 무시하는 게 합리적인지 모르겠어."},
    {"id": "hc036", "category": "daily_preference", "text": "비 오는 날엔 집에 박혀 있는 게 좋아, 아니면 일부러 나가서 기분 전환하는 게 좋아?"},
    {"id": "hc037", "category": "daily_preference", "text": "커피를 줄이고 싶은데 아침 루틴에서 커피가 빠지면 하루가 안 켜지는 느낌이야."},
    {"id": "hc038", "category": "daily_preference", "text": "요즘 플레이리스트가 다 질렸어, 기분 전환용으로 어떤 분위기부터 바꾸면 좋을까?"},
    {"id": "hc039", "category": "daily_preference", "text": "주말에 집콕하면 아깝고 나가면 피곤해, 이런 애매한 날엔 뭐가 맞아?"},
    {"id": "hc040", "category": "daily_preference", "text": "오늘 하루를 한 단어로 고르면 버팀인데, 이게 잘 산 건지 그냥 버틴 건지 모르겠어."},
    {"id": "hc041", "category": "health_body", "text": "머리가 계속 무겁고 눈도 뻐근한데 그냥 피곤한 건지 쉬어야 할 신호인지 봐줘."},
    {"id": "hc042", "category": "health_body", "text": "밤마다 잠이 안 와서 다음 날이 망가져, 오늘 밤엔 뭘 먼저 바꿔야 해?"},
    {"id": "hc043", "category": "health_body", "text": "속이 답답한 게 소화 문제인지 스트레스인지 모르겠어, 일단 어떻게 구분해?"},
    {"id": "hc044", "category": "health_body", "text": "운동 쉬면 불안한데 몸은 회복이 덜 된 느낌이야, 쉬는 것도 훈련으로 봐도 돼?"},
    {"id": "hc045", "category": "health_body", "text": "하루 종일 앉아 있었더니 허리랑 목이 굳었어, 지금 당장 할 수 있는 것부터 말해줘."},
    {"id": "hc046", "category": "playful_reaction", "text": "나 오늘 진짜 생산성 바닥 쳤는데, 혼내지 말고 정신 차리게 한마디 해줘."},
    {"id": "hc047", "category": "playful_reaction", "text": "치킨 먹고 싶은 나와 통장 지키려는 내가 싸우고 있어, 판결 내려줘."},
    {"id": "hc048", "category": "playful_reaction", "text": "내 집중력 지금 3초짜리 광고보다 짧아, 그래도 할 일 시작하게 만들어봐."},
    {"id": "hc049", "category": "playful_reaction", "text": "오늘의 나는 침대와 한 몸이 됐어, 그래도 인간으로 복귀하는 루트 있어?"},
    {"id": "hc050", "category": "playful_reaction", "text": "나 지금 답정너인 거 아는데 그래도 네가 직설적으로 말해줘, 이거 사지 말까?"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 50-prompt shadow probe comparing deterministic DraftNLG silver frames with gated ModernBERT frame heads."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--probe-json", type=Path, default=None)
    parser.add_argument("--probe-name", default=DEFAULT_PROBE_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trusted-axes", default=",".join(TRUSTED_AXES))
    parser.add_argument("--sequential", action="store_true")
    return parser.parse_args()


def parse_trusted_axes(raw: str | None) -> tuple[str, ...] | None:
    value = str(raw or "").strip()
    if not value:
        return TRUSTED_AXES
    if value.lower() in {"*", "all", "open", "ungated"}:
        return None
    if value.lower() in {"none", "empty", "blocked"}:
        return ()
    parts = [part.strip() for chunk in value.split("|") for part in chunk.replace(";", ",").split(",")]
    return tuple(dict.fromkeys(part for part in parts if part))


def load_probe_items(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return list(PROBE_ITEMS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"probe json must contain a list: {path}")
    items: list[dict[str, str]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("prompt") or "").strip()
        if not text:
            continue
        items.append(
            {
                "id": str(item.get("id") or f"probe-{index:03d}"),
                "category": str(item.get("category") or "custom"),
                "text": text,
            }
        )
    return items


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(location=location or "서울", temperature_c=18.0, description="맑음", wind_kph=7.0)


class FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="12:30",
            formatted_date="2026-05-23",
            timezone_name="Asia/Seoul",
            source="black_frame_shadow_probe50_fake_clock",
        )


class FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        return [NewsHeadline(title="로컬 shadow probe 뉴스 픽스처", source="local-test")][:limit]


def configure_runtime(model_dir: Path, *, device: str, trusted_axes: tuple[str, ...] | None) -> AppConfig:
    if trusted_axes is None:
        trusted_value = "*"
    else:
        trusted_value = ",".join(trusted_axes)
    os.environ.update(
        {
            "BOT_PERSONA": "black",
            "GENERATION_BACKEND": "template",
            "STRICT_LLM_ONLY": "false",
            "BLACK_DRAFT_ONLY": "true",
            "STATE_BACKEND": "memory",
            "KNOWLEDGE_BACKEND": "builtin",
            "TTS_ENABLED": "false",
            "DEFAULT_LOCATION": "서울",
            "INTENT_MODEL_TYPE": "modernbert_meaning",
            "KCBERT_MODEL_PATH": str(model_dir),
            "KCBERT_DEVICE": device,
            "INTENT_MEANING_TRUSTED_AXES": trusted_value,
            "BLACK_MODEL_ALIAS": "",
        }
    )
    return AppConfig.from_env()


def signal_axis_map(packet: Any, *, source: str = "meaning_model") -> dict[str, dict[str, Any]]:
    axes: dict[str, dict[str, Any]] = {}
    for signal in list(getattr(packet, "signals", []) or []):
        if str(getattr(signal, "source", "") or "") != source:
            continue
        axis = str(getattr(signal, "axis", "") or "").strip()
        label = str(getattr(signal, "label", "") or "").strip()
        if not axis or not label:
            continue
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
        if axis not in axes or confidence > float(axes[axis].get("confidence", 0.0)):
            axes[axis] = {"label": label, "confidence": confidence, "evidence": list(getattr(signal, "evidence", []) or [])}
    return axes


def draft_targets_from_draft(draft: dict[str, Any]) -> dict[str, Any]:
    semantic_frame = draft.get("semantic_frame") if isinstance(draft.get("semantic_frame"), dict) else {}
    targets = semantic_frame.get("targets") if isinstance(semantic_frame.get("targets"), dict) else {}
    return dict(targets)


def compare_frame_axes(
    *,
    draft_targets: dict[str, Any],
    model_axes: dict[str, dict[str, Any]],
    trusted_axes: tuple[str, ...] | None,
) -> dict[str, dict[str, Any]]:
    axes = tuple(model_axes.keys()) if trusted_axes is None else trusted_axes
    comparison: dict[str, dict[str, Any]] = {}
    for axis in axes:
        draft_label = draft_targets.get(axis)
        model_payload = model_axes.get(axis) or {}
        model_label = model_payload.get("label")
        comparison[axis] = {
            "draft": draft_label,
            "model": model_label,
            "confidence": model_payload.get("confidence"),
            "match": bool(draft_label is not None and model_label is not None and str(draft_label) == str(model_label)),
            "missing_draft": draft_label is None,
            "missing_model": model_label is None,
        }
    return comparison


def untrusted_model_axes(model_axes: dict[str, dict[str, Any]], trusted_axes: tuple[str, ...] | None) -> list[str]:
    if trusted_axes is None:
        return []
    trusted = set(trusted_axes)
    return sorted(axis for axis in model_axes if axis not in trusted)


def summarize_results(rows: list[dict[str, Any]], trusted_axes: tuple[str, ...] | None) -> dict[str, Any]:
    axes = sorted({axis for row in rows for axis in row["axis_comparison"]})
    axis_summary: dict[str, dict[str, int | float]] = {}
    for axis in axes:
        compared = 0
        matches = 0
        missing_model = 0
        missing_draft = 0
        for row in rows:
            item = row["axis_comparison"].get(axis)
            if not item:
                continue
            if item["missing_model"]:
                missing_model += 1
            if item["missing_draft"]:
                missing_draft += 1
            if not item["missing_model"] and not item["missing_draft"]:
                compared += 1
                if item["match"]:
                    matches += 1
        axis_summary[axis] = {
            "compared": compared,
            "matches": matches,
            "mismatches": compared - matches,
            "missing_model": missing_model,
            "missing_draft": missing_draft,
            "match_rate": round(matches / compared, 4) if compared else 0.0,
        }
    leak_rows = [row for row in rows if row["untrusted_model_axes"]]
    mismatch_rows = [
        row
        for row in rows
        if any(
            not item["match"] and not item["missing_model"] and not item["missing_draft"]
            for item in row["axis_comparison"].values()
        )
    ]
    suspected_silver_underlabels = [
        row
        for row in rows
        if row.get("draft_targets", {}).get("schema") == "direct_reply"
        and row.get("model_axes", {}).get("schema", {}).get("label") not in {None, "", "direct_reply"}
    ]
    generic_draft_family_rows = [
        row
        for row in rows
        if row.get("draft_targets", {}).get("draft_frame_family") == "social_acknowledgement"
    ]
    return {
        "total": len(rows),
        "trusted_axes": list(trusted_axes) if trusted_axes is not None else None,
        "categories": dict(Counter(row["category"] for row in rows)),
        "classifier_sources": dict(Counter(str(row.get("classifier_source")) for row in rows)),
        "draft_reasons": dict(Counter(str(row.get("draft_reason")) for row in rows)),
        "axis_summary": axis_summary,
        "rows_with_axis_mismatch": len(mismatch_rows),
        "rows_with_untrusted_model_axis": len(leak_rows),
        "suspected_silver_underlabel_rows": len(suspected_silver_underlabels),
        "generic_draft_family_rows": len(generic_draft_family_rows),
        "nonempty_replies": sum(1 for row in rows if str(row.get("reply") or "").strip()),
        "llm_used": sum(1 for row in rows if row.get("llm_used")),
    }


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    trusted_axes = parse_trusted_axes(args.trusted_axes)
    items = load_probe_items(args.probe_json)
    config = configure_runtime(args.model_dir, device=args.device, trusted_axes=trusted_axes)
    engine = build_engine(config)
    engine.weather_service = FakeWeatherService()
    engine.time_service = FakeTimeService()
    engine.news_service = FakeNewsService()

    rows: list[dict[str, Any]] = []
    shared_user_id = "black-frame-shadow-probe50" if args.sequential else ""
    try:
        for index, item in enumerate(items, start=1):
            user_id = shared_user_id or f"black-frame-shadow-probe50-{index:03d}"
            result = await engine.respond(user_id, item["text"])
            packet = result.features.meaning_packet
            draft = result.draft_utterance if isinstance(result.draft_utterance, dict) else {}
            draft_targets = draft_targets_from_draft(draft)
            model_axes = signal_axis_map(packet)
            comparison = compare_frame_axes(
                draft_targets=draft_targets,
                model_axes=model_axes,
                trusted_axes=trusted_axes,
            )
            evidence = result.features.classifier_evidence
            row = {
                "index": index,
                "id": item["id"],
                "category": item["category"],
                "text": item["text"],
                "intent": result.features.intent.value,
                "schema": result.features.question_schema,
                "speech_act": result.features.speech_act,
                "action": result.decision.action.value,
                "classifier_source": evidence.source if evidence else None,
                "draft_reason": str(draft.get("direct_surface_reason") or draft.get("output_shape") or ""),
                "draft_targets": draft_targets,
                "model_axes": model_axes,
                "axis_comparison": comparison,
                "untrusted_model_axes": untrusted_model_axes(model_axes, trusted_axes),
                "reply": result.reply,
                "llm_used": result.llm_used,
            }
            rows.append(row)
            mismatched = [
                axis
                for axis, axis_item in comparison.items()
                if not axis_item["match"] and not axis_item["missing_model"] and not axis_item["missing_draft"]
            ]
            print(
                f"[{index:02d}/{len(items)}] {item['id']} action={row['action']} "
                f"schema={row['schema']} mismatch={','.join(mismatched) or '-'} "
                f"leak={','.join(row['untrusted_model_axes']) or '-'}",
                flush=True,
            )
    finally:
        close = getattr(engine.state_store, "close", None)
        if callable(close):
            close()

    summary = summarize_results(rows, trusted_axes)
    return {
        "metadata": {
            "name": args.probe_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_dir": str(args.model_dir),
            "device": args.device,
            "mode": "sequential" if args.sequential else "independent",
            "generation_backend": config.generation_backend,
            "intent_model_type": config.intent_model_type,
            "servers_started": False,
            "rewrite": "disabled",
        },
        "summary": summary,
        "results": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Black Frame Shadow Probe 50",
        "",
        f"- model: `{payload['metadata']['model_dir']}`",
        f"- total: `{summary['total']}`",
        f"- trusted axes: `{summary['trusted_axes']}`",
        f"- rows with axis mismatch: `{summary['rows_with_axis_mismatch']}`",
        f"- rows with untrusted model axis: `{summary['rows_with_untrusted_model_axis']}`",
        f"- suspected silver underlabel rows: `{summary['suspected_silver_underlabel_rows']}`",
        f"- generic draft family rows: `{summary['generic_draft_family_rows']}`",
        f"- llm used: `{summary['llm_used']}`",
        "",
        "## Axis Summary",
        "",
        "| axis | compared | match | mismatch | missing model | match rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for axis, item in sorted(summary["axis_summary"].items()):
        lines.append(
            f"| `{axis}` | {item['compared']} | {item['matches']} | {item['mismatches']} | "
            f"{item['missing_model']} | {item['match_rate']:.2%} |"
        )
    lines.extend(["", "## Mismatch Samples", ""])
    sample_count = 0
    for row in payload["results"]:
        mismatches = {
            axis: item
            for axis, item in row["axis_comparison"].items()
            if not item["match"] and not item["missing_model"] and not item["missing_draft"]
        }
        if not mismatches:
            continue
        sample_count += 1
        lines.append(f"- `{row['id']}` {row['category']}: {row['text']}")
        lines.append(f"  - mismatches: `{mismatches}`")
        if sample_count >= 12:
            break
    if sample_count == 0:
        lines.append("- no direct trusted-axis mismatches")
    lines.extend(["", "## Silver Underlabel Samples", ""])
    underlabel_count = 0
    for row in payload["results"]:
        if row.get("draft_targets", {}).get("schema") != "direct_reply":
            continue
        model_schema = row.get("model_axes", {}).get("schema", {}).get("label")
        if not model_schema or model_schema == "direct_reply":
            continue
        underlabel_count += 1
        lines.append(
            f"- `{row['id']}` draft=`direct_reply` model=`{model_schema}` "
            f"family_model=`{row.get('model_axes', {}).get('draft_frame_family', {}).get('label')}`: {row['text']}"
        )
        if underlabel_count >= 12:
            break
    if underlabel_count == 0:
        lines.append("- no suspected silver underlabel rows")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise SystemExit(f"model dir not found: {args.model_dir}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(run_probe(args))
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved frame shadow probe to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
