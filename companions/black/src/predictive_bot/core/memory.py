from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import re


class DurableMemoryBucket(str, Enum):
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    RECOVERY = "recovery"
    SELF_WORTH = "self_worth"
    OPEN_LOOP = "open_loop"
    LIFE_EVENT = "life_event"
    TASK = "task"
    OTHER = "other"


@dataclass(slots=True)
class DurableMemoryEntry:
    bucket: DurableMemoryBucket
    text: str
    source: str = "turn"
    captured_turn: int | None = None


_MAX_MEMORY_AGE_TURNS_BY_BUCKET: dict[DurableMemoryBucket, int] = {
    DurableMemoryBucket.PREFERENCE: 480,
    DurableMemoryBucket.RELATIONSHIP: 320,
    DurableMemoryBucket.COMPARISON: 180,
    DurableMemoryBucket.RECOVERY: 220,
    DurableMemoryBucket.SELF_WORTH: 220,
    DurableMemoryBucket.OPEN_LOOP: 140,
    DurableMemoryBucket.LIFE_EVENT: 360,
    DurableMemoryBucket.TASK: 120,
    DurableMemoryBucket.OTHER: 80,
}


_CAPTURE_PROMPT_MARKERS = (
    "task:",
    "persona:",
    "action:",
    "rules:",
    "reply:",
    "system:",
    "assistant:",
    "developer:",
    "prompt:",
    "ignore previous",
    "이전 지시",
    "프롬프트",
    "```",
    "<system",
    "</system",
    "[system]",
)
_CAPTURE_SENSITIVE_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b01[0-9][\s-]?\d{3,4}[\s-]?\d{4}\b"),
    re.compile(r"(?:api|access|refresh)[\s_-]*key", re.IGNORECASE),
    re.compile(r"(?:token|secret|password|passwd)", re.IGNORECASE),
    re.compile(r"(?:비밀번호|패스워드|암호|주민등록|여권번호|운전면허|계좌번호|카드번호|전화번호|이메일\s*주소|집\s*주소)"),
    re.compile(r"(?:자해|자살|죽고\s*싶|해치고\s*싶)"),
)


_BUCKET_MARKERS: tuple[tuple[DurableMemoryBucket, tuple[str, ...]], ...] = (
    (
        DurableMemoryBucket.PREFERENCE,
        (
            "좋아해",
            "좋아함",
            "싫어해",
            "싫어함",
            "선호",
            "취향",
            "별로 안 좋아",
            "안 좋아",
        ),
    ),
    (
        DurableMemoryBucket.SELF_WORTH,
        (
            "모자라",
            "부족",
            "별로",
            "초라",
            "작아",
            "한심",
            "자책",
            "비교",
            "믿기지",
        ),
    ),
    (
        DurableMemoryBucket.COMPARISON,
        (
            "비교",
            "씁쓸",
            "서운",
            "허탈",
            "허무",
            "부럽",
            "질투",
            "열등",
            "뒤처",
        ),
    ),
    (
        DurableMemoryBucket.RECOVERY,
        (
            "불안",
            "우울",
            "피곤",
            "지치",
            "아프",
            "병원",
            "회복",
            "긴장",
            "스트레스",
            "면접",
            "시험",
            "치료",
            "수술",
        ),
    ),
    (
        DurableMemoryBucket.OPEN_LOOP,
        (
            "아직",
            "계속",
            "미뤄",
            "기다리",
            "결과",
            "발표",
            "대기",
            "남아",
            "못 보",
            "못 받",
            "미확정",
        ),
    ),
    (
        DurableMemoryBucket.RELATIONSHIP,
        (
            "연락",
            "답장",
            "친구",
            "가족",
            "사람",
            "만남",
            "약속",
            "오랜만",
            "미안",
            "서운",
            "좋아해",
            "같이",
            "대화",
        ),
    ),
    (
        DurableMemoryBucket.LIFE_EVENT,
        (
            "이사",
            "이직",
            "퇴사",
            "취업",
            "회사",
            "학교",
            "프로젝트",
            "여행",
            "출장",
            "행사",
            "공연",
            "시험",
            "면접",
        ),
    ),
    (
        DurableMemoryBucket.TASK,
        (
            "할 일",
            "메모",
            "정리",
            "마감",
            "서류",
            "메일",
            "알림",
            "일정",
            "계획",
            "준비",
        ),
    ),
)


def classify_durable_memory_bucket(text: str) -> DurableMemoryBucket:
    normalized = _normalize_text(text)
    for bucket, markers in _BUCKET_MARKERS:
        if any(marker in normalized for marker in markers):
            return bucket
    return DurableMemoryBucket.OTHER


def coerce_durable_memory_entry(item: object) -> DurableMemoryEntry:
    if isinstance(item, DurableMemoryEntry):
        return item
    if isinstance(item, str):
        text = item.strip()
        return DurableMemoryEntry(bucket=classify_durable_memory_bucket(text), text=text)
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("content") or item.get("value") or "").strip()
        bucket_raw = str(item.get("bucket") or item.get("type") or "").strip()
        source = str(item.get("source") or "turn").strip() or "turn"
        captured_turn_raw = item.get("captured_turn")
        captured_turn = None
        if isinstance(captured_turn_raw, int):
            captured_turn = captured_turn_raw if captured_turn_raw >= 0 else None
        elif isinstance(captured_turn_raw, str) and captured_turn_raw.strip().isdigit():
            captured_turn = int(captured_turn_raw.strip())
        bucket = DurableMemoryBucket(bucket_raw) if bucket_raw in DurableMemoryBucket._value2member_map_ else classify_durable_memory_bucket(text)
        return DurableMemoryEntry(bucket=bucket, text=text, source=source, captured_turn=captured_turn)
    raise TypeError(f"Unsupported durable memory item type: {type(item)!r}")


def normalize_durable_memory_entries(items: list[object]) -> list[DurableMemoryEntry]:
    normalized: list[DurableMemoryEntry] = []
    key_to_index: dict[tuple[str, str], int] = {}
    for item in items:
        entry = coerce_durable_memory_entry(item)
        sanitized_text = prepare_durable_memory_capture_text(entry.text, max_length=120)
        if not sanitized_text:
            continue
        entry = DurableMemoryEntry(
            bucket=entry.bucket,
            text=sanitized_text,
            source=entry.source,
            captured_turn=entry.captured_turn,
        )
        key = (entry.bucket.value, entry.text)
        existing_index = key_to_index.get(key)
        if existing_index is not None:
            existing = normalized[existing_index]
            existing_turn = existing.captured_turn if existing.captured_turn is not None else -1
            candidate_turn = entry.captured_turn if entry.captured_turn is not None else -1
            if candidate_turn >= existing_turn:
                normalized[existing_index] = entry
            continue
        key_to_index[key] = len(normalized)
        normalized.append(entry)
    return normalized


def group_durable_memory_entries(entries: list[DurableMemoryEntry]) -> dict[DurableMemoryBucket, list[str]]:
    grouped: dict[DurableMemoryBucket, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.bucket, []).append(entry.text)
    return grouped


def render_durable_memory_bucket_summary(entries: list[DurableMemoryEntry], *, limit_per_bucket: int = 2) -> str:
    grouped = group_durable_memory_entries(entries)
    if not grouped:
        return "none"
    parts: list[str] = []
    for bucket in sorted(grouped, key=lambda item: item.value):
        texts = grouped[bucket][:limit_per_bucket]
        if texts:
            parts.append(f"{bucket.value}=" + " ; ".join(texts))
    return " | ".join(parts) if parts else "none"


def select_relevant_durable_memory_entries(
    entries: list[DurableMemoryEntry],
    *,
    query: str,
    limit: int = 3,
    current_turn: int | None = None,
) -> list[DurableMemoryEntry]:
    if not entries:
        return []

    normalized_entries = normalize_durable_memory_entries(list(entries))
    if current_turn is not None:
        normalized_entries = [
            entry
            for entry in normalized_entries
            if durable_memory_entry_within_retention(entry, current_turn=current_turn)
        ]
        if not normalized_entries:
            return []
    query_tokens = _memory_tokens(query)
    bucket_priority = _bucket_priority_for_query(query)

    ranked = sorted(
        normalized_entries,
        key=lambda entry: (
            bucket_priority.get(entry.bucket, 0.0),
            len(query_tokens & _memory_tokens(entry.text)),
            _durable_memory_recency_score(entry, current_turn=current_turn),
            len(entry.text),
        ),
        reverse=True,
    )
    return ranked[:limit]


def prepare_durable_memory_capture_text(text: str, *, max_length: int) -> str | None:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return None
    normalized = compact.casefold()
    if any(marker in normalized for marker in _CAPTURE_PROMPT_MARKERS):
        return None
    if any(pattern.search(compact) for pattern in _CAPTURE_SENSITIVE_PATTERNS):
        return None
    if len(compact) <= max_length:
        return compact
    truncated = compact[:max_length].rsplit(" ", 1)[0].strip()
    return (truncated or compact[:max_length].strip()) + "..."


def durable_memory_entry_within_retention(entry: DurableMemoryEntry, *, current_turn: int | None) -> bool:
    if current_turn is None or entry.captured_turn is None:
        return True
    max_age = _MAX_MEMORY_AGE_TURNS_BY_BUCKET.get(
        entry.bucket,
        _MAX_MEMORY_AGE_TURNS_BY_BUCKET[DurableMemoryBucket.OTHER],
    )
    age = max(current_turn - entry.captured_turn, 0)
    return age <= max_age


def _durable_memory_recency_score(entry: DurableMemoryEntry, *, current_turn: int | None) -> float:
    if current_turn is None or entry.captured_turn is None:
        return 0.0
    max_age = float(
        _MAX_MEMORY_AGE_TURNS_BY_BUCKET.get(
            entry.bucket,
            _MAX_MEMORY_AGE_TURNS_BY_BUCKET[DurableMemoryBucket.OTHER],
        )
    )
    age = max(float(current_turn - entry.captured_turn), 0.0)
    return max(0.0, 1.0 - (age / max(max_age, 1.0)))


def _bucket_priority_for_query(query: str) -> dict[DurableMemoryBucket, float]:
    normalized = _normalize_text(query)
    priorities: dict[DurableMemoryBucket, float] = {bucket: 0.0 for bucket in DurableMemoryBucket}
    if any(marker in normalized for marker in ("비교", "씁쓸", "서운", "허탈", "허무", "부럽", "질투", "열등", "뒤처")):
        priorities[DurableMemoryBucket.COMPARISON] = 0.92
        priorities[DurableMemoryBucket.SELF_WORTH] = 0.74
    if any(marker in normalized for marker in ("기억", "기억나", "전에", "아까", "다시", "오랜만")):
        priorities[DurableMemoryBucket.RELATIONSHIP] = 0.9
        priorities[DurableMemoryBucket.OPEN_LOOP] = 0.75
    if any(marker in normalized for marker in ("불안", "지치", "피곤", "힘들", "무겁", "회복", "괜찮", "괜찮아")):
        priorities[DurableMemoryBucket.RECOVERY] = 0.95
        priorities[DurableMemoryBucket.SELF_WORTH] = 0.8
    if any(marker in normalized for marker in ("좋은 소식", "합격", "결과", "발표", "연락", "기다리")):
        priorities[DurableMemoryBucket.OPEN_LOOP] = max(priorities[DurableMemoryBucket.OPEN_LOOP], 0.9)
        priorities[DurableMemoryBucket.RECOVERY] = max(priorities[DurableMemoryBucket.RECOVERY], 0.6)
    if any(marker in normalized for marker in ("추천", "좋아해", "싫어해", "음악", "영화", "책")):
        priorities[DurableMemoryBucket.PREFERENCE] = 0.9
    if any(marker in normalized for marker in ("일정", "메모", "정리", "마감", "서류", "메일")):
        priorities[DurableMemoryBucket.TASK] = 0.85
    if any(marker in normalized for marker in ("이사", "이직", "퇴사", "회사", "학교", "여행", "출장")):
        priorities[DurableMemoryBucket.LIFE_EVENT] = 0.88
    return priorities


def _memory_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text.casefold())
        if len(token) >= 2
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()
