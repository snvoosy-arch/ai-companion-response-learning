from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_VISIBLE_FORMAT_MARKERS_RE = re.compile(
    r"(?i)(<think>|</think>|<\|im_start\|>|<\|im_end\|>|^\s*(assistant|system|user)\s*[:：-]?\s*$)",
    flags=re.MULTILINE,
)
_COMMENT_ARTIFACT_RE = re.compile(r"(^|[\s\t])//|\t{2,}|/\*|\*/")
_LATIN1_LIKE_RE = re.compile(r"[À-ÿ]")
_HANGUL_RE = re.compile(r"[가-힣]")
_REPEATED_PUNCT_RE = re.compile(r"[!?~]{3,}|ㅋ{3,}|ㅎ{3,}")
_NON_WORD_RE = re.compile(r"[^0-9a-z가-힣]+", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_STYLE_PROMISE_RE = re.compile(
    r"(편하게|짧게|자연스럽게|한\s*줄|한\s*문장).{0,16}(말할게|답할게|해볼게|해줄게|할게|둘게)",
)
_LOW_CONTENT_ONLY_RE = re.compile(
    r"^\s*(짧게|짧고|짧게만|한\s*줄|한\s*문장|괜찮아|좋아|응|그럴\s*만하지|그럴만하지)\s*[.!?…~]*\s*$",
    flags=re.IGNORECASE,
)
_SOFT_FILLER_TOKENS = {
    "답",
    "답변",
    "해",
    "해줘",
    "해라",
    "줘",
    "말",
    "말해",
    "말해줘",
    "말해라",
    "문장",
    "줄",
    "한",
    "두",
    "좀",
    "조금",
    "그냥",
    "더",
    "만",
    "로",
    "으로",
    "해도",
    "해줘도",
    "없이",
}
_GENERIC_REPLY_TOKENS = {
    "오늘은",
    "지금은",
    "그냥",
    "그말",
    "그",
    "말",
    "마음",
    "쪽",
    "네",
    "더",
}


@dataclass(slots=True, frozen=True)
class GuardIssue:
    code: str
    detail: str
    blocking: bool


@dataclass(slots=True, frozen=True)
class GuardResult:
    reply: str
    issues: tuple[GuardIssue, ...]

    @property
    def should_retry(self) -> bool:
        return any(issue.blocking for issue in self.issues)


class OutputGuard:
    def __init__(self, *, max_chars: int = 200) -> None:
        self._max_chars = max_chars

    def build_response_instruction(
        self,
        *,
        user_prompt: str,
        recent_replies: list[str],
        retry: bool = False,
    ) -> str:
        lines = [
            "한국어로 최종 답변만 써라.",
            "최근 답변과 같은 문장, 같은 시작어, 같은 마무리를 반복하지 말고 새 표현으로 답해라.",
        ]
        if _looks_like_format_only_prompt(user_prompt):
            lines.append("형식 설명 대신 아주 짧은 자연문으로만 답해라.")
        elif _looks_like_format_led_prompt(user_prompt):
            lines.append("형식 지시는 답변 방식만 정하고, 실제 내용에는 바로 반응해라.")
        if _looks_like_identity_prompt(user_prompt):
            lines.append("이름이나 정체성을 물으면 반드시 자신을 White/화이트라고 답하고, 사용자 이름을 자기 이름처럼 말하지 마라.")
        if _looks_like_relational_prompt(user_prompt):
            lines.append("관계, 답장, 연락 맥락에서는 그 마음, 그 의미 같은 대명사로 흐리지 말고 행동, 타이밍, 상황 중 하나를 구체적으로 짚어라.")
        if _looks_like_anti_generic_prompt(user_prompt):
            lines.append("빈말처럼 들리는 공감어구만 돌리지 말고, 왜 그렇게 느껴지는지나 무엇이 걸리는지 직접 짚어라.")
        if _looks_like_emotional_prompt(user_prompt):
            lines.append("감정 이름만 말하지 말고, 그 감정이 오늘 어떤 장면이나 몸의 느낌으로 이어지는지 한 번 더 보여줘라.")
        if recent_replies:
            lines.append("최근 답변의 표현을 그대로 가져오지 말고, 같은 패턴을 피해서 답해라.")
        lines.append("답변은 실제 대화처럼 자연스럽게 시작하고, 설명서처럼 들리지 않게 써라.")
        if retry:
            lines.append(_response_style_hint(user_prompt=user_prompt, recent_replies=recent_replies, retry=True))
            lines.append("방금 쓴 표현은 이미 실패했다. 같은 뜻이라도 더 짧고 더 구체적으로 새로 써라.")
        return "\n".join(lines)

    def check(
        self,
        reply: str,
        *,
        user_prompt: str,
        recent_replies: list[str],
    ) -> GuardResult:
        normalized_reply = reply.strip()
        issues: list[GuardIssue] = []

        if not normalized_reply or normalized_reply == "응답이 비어 있어요.":
            issues.append(GuardIssue("empty_after_sanitize", "정리 후 출력이 비어 있음", True))
            return GuardResult(reply="응답이 비어 있어요.", issues=tuple(issues))

        if _VISIBLE_FORMAT_MARKERS_RE.search(normalized_reply):
            issues.append(GuardIssue("think_tag_leak", "보이는 형식 토큰이 남아 있음", True))

        if _COMMENT_ARTIFACT_RE.search(normalized_reply):
            issues.append(GuardIssue("comment_artifact", "코드 주석이나 탭 반복 같은 생성 찌꺼기가 남아 있음", True))

        if _looks_like_prompt_echo(normalized_reply, user_prompt):
            issues.append(GuardIssue("prompt_echo", "사용자 문장 또는 지시문을 따라 읽음", True))

        if _looks_like_meta_reply(normalized_reply, user_prompt):
            issues.append(GuardIssue("prompt_echo", "형식 지시만 다시 말하는 메타 답변", True))

        if _looks_like_format_only_prompt(user_prompt) and _contains_format_terms(normalized_reply):
            issues.append(GuardIssue("prompt_echo", "형식-only 프롬프트에 메타 단어가 다시 섞임", True))

        if _looks_like_format_led_miss(normalized_reply, user_prompt):
            issues.append(GuardIssue("format_led_miss", "형식이 섞인 입력의 실제 내용과 맞지 않는 안전문장으로 벗어남", True))

        if _looks_like_low_content_reply(normalized_reply, user_prompt):
            issues.append(GuardIssue("low_content_reply", "의미에 비해 답변 정보량이 너무 적음", True))

        if _looks_like_identity_prompt(user_prompt):
            if _looks_like_wrong_identity_reply(normalized_reply):
                issues.append(GuardIssue("identity_confusion", "White가 아닌 이름이나 사용자 이름을 자기 정체성처럼 말함", True))
            if "white" not in normalized_reply.lower() and "화이트" not in normalized_reply:
                issues.append(GuardIssue("weak_identity_answer", "정체성 질문에 White/화이트라고 답하지 않음", True))

        if _looks_like_relational_vague_reply(normalized_reply, user_prompt):
            issues.append(GuardIssue("relational_vague_reply", "관계 맥락에서 지나치게 추상적으로 흐림", True))

        if _looks_like_checkin_vague_reply(normalized_reply, user_prompt):
            issues.append(GuardIssue("checkin_vague_reply", "안부 질문에 현재 상태 대신 추상적인 자기 연출로 답함", True))

        if _looks_like_shallow_paraphrase(normalized_reply, user_prompt):
            issues.append(GuardIssue("shallow_paraphrase", "사용자 표현을 짧게 비틀어 따라 읽음", True))

        if _is_repeated_reply(normalized_reply, recent_replies):
            issues.append(GuardIssue("repeated_reply", "최근 답변과 거의 동일함", True))

        if _has_encoding_corruption(normalized_reply):
            issues.append(GuardIssue("encoding_corruption", "문자 인코딩이 깨진 흔적이 있음", True))

        if len(normalized_reply) > self._max_chars:
            issues.append(GuardIssue("too_long", f"{self._max_chars}자 초과", False))
            normalized_reply = _trim_to_limit(normalized_reply, self._max_chars)

        if _looks_like_persona_violation(normalized_reply):
            issues.append(GuardIssue("persona_violation", "White 톤에서 벗어나는 구두점/말투", False))

        return GuardResult(reply=normalized_reply, issues=tuple(issues))

    def should_prefer_repair_first(
        self,
        *,
        user_prompt: str,
        issues: tuple[GuardIssue, ...],
    ) -> bool:
        if not (
            _looks_like_checkin_prompt(user_prompt)
            or _looks_like_format_only_prompt(user_prompt)
            or _looks_like_format_led_prompt(user_prompt)
            or _looks_like_identity_prompt(user_prompt)
        ):
            return False

        issue_codes = {issue.code for issue in issues}
        structural_issues = {"think_tag_leak", "prompt_echo", "encoding_corruption"}
        if issue_codes & structural_issues:
            return False

        natural_language_issues = {
            "checkin_vague_reply",
            "format_led_miss",
            "low_content_reply",
            "shallow_paraphrase",
            "repeated_reply",
            "relational_vague_reply",
            "identity_confusion",
            "weak_identity_answer",
        }
        return bool(issue_codes & natural_language_issues)

    def build_retry_instruction(
        self,
        *,
        user_prompt: str,
        issues: tuple[GuardIssue, ...],
        recent_replies: list[str] | None = None,
    ) -> str:
        issue_codes = ", ".join(sorted({issue.code for issue in issues})) or "unknown"
        lines = [
            "방금 답변은 형식 누출 또는 메타 반복 문제가 있었다.",
            f"문제 코드: {issue_codes}",
            "이번에는 <think>, role 토큰, 태그, 코드블록, 접두사 없이 최종 답변만 한국어로 출력해라.",
            "사용자 문장을 따라 읽거나 규칙을 설명하지 말고, 의미에만 짧고 자연스럽게 반응해라.",
            "답변은 1~2문장, 가능하면 80자 안팎으로 유지해라.",
        ]
        if any(issue.code == "repeated_reply" for issue in issues):
            lines.append("최근 답변과 같은 문장을 반복하지 말고, 첫 문장부터 새로 시작해라.")
        if any(issue.code == "low_content_reply" for issue in issues):
            lines.append("한 단어나 형식 단서만 던지지 말고, 상황이나 감정에 닿는 문장으로 답해라.")
        if any(issue.code in {"identity_confusion", "weak_identity_answer"} for issue in issues):
            lines.append("이름이나 정체성을 물은 입력이면 반드시 자신을 White/화이트라고 말하고, 사용자 이름을 자기 이름처럼 쓰지 마라.")
        if any(issue.code == "relational_vague_reply" for issue in issues):
            lines.append("그 마음, 그 의미, 그 정도 같은 추상 대명사를 쓰지 말고 답장, 연락, 보고 싶음, 겁남 같은 핵심을 직접 말해라.")
        if any(issue.code == "format_led_miss" for issue in issues):
            lines.append("형식 지시가 섞여 있어도 내용에서 벗어나지 말고, 방금 입력에 들어 있던 실제 감정이나 요청을 한 문장으로 짚어라.")
        if any(issue.code == "checkin_vague_reply" for issue in issues):
            lines.append("안부나 기분 질문에는 지금 상태를 바로 말해라. 네가 어떻게 남겠다는 말보다 차분함, 피곤함, 무거움 같은 현재 느낌을 먼저 내라.")
        if any(issue.code == "shallow_paraphrase" for issue in issues):
            lines.append("사용자 문장의 단어를 짧게 비틀어 되풀이하지 말고, 새 의미나 새 초점을 하나라도 추가해라.")
        if _looks_like_format_only_prompt(user_prompt):
            lines.append("입력이 형식 지시 중심이면 메타 설명 대신 자연스러운 최소 응답 한 문장으로 끝내라.")
        elif _looks_like_format_led_prompt(user_prompt):
            lines.append("형식 지시는 숨기고, 형식 뒤에 있는 실제 감정이나 의미에 직접 반응해라.")
        if recent_replies:
            lines.append("최근 답변의 표현과 같은 리듬을 피하고, 다른 첫 어절로 시작해라.")
        lines.append(_response_style_hint(user_prompt=user_prompt, recent_replies=recent_replies or [], retry=True))
        return "\n".join(lines)

    def build_repair_instruction(
        self,
        *,
        user_prompt: str,
        issues: tuple[GuardIssue, ...],
        rejected_replies: list[str] | None = None,
        recent_replies: list[str] | None = None,
    ) -> str:
        issue_codes = ", ".join(sorted({issue.code for issue in issues})) or "unknown"
        lines = [
            "앞선 답변들은 톤이나 내용 정합성이 어긋나서 폐기되었다.",
            f"문제 코드: {issue_codes}",
            "이번에는 새로 자연스럽게 다시 말해라. 이전 답변을 짜깁기하거나 비슷한 문장을 반복하지 마라.",
            "사용자 입력의 실제 뜻에 바로 반응하고, 짧지만 사람처럼 매끈한 한국어 1~2문장으로 써라.",
        ]
        if rejected_replies:
            rejected = [reply.strip() for reply in rejected_replies if reply.strip()]
            if rejected:
                lines.append("다음 표현들은 실패했으니 다시 쓰지 마라:")
                for reply in rejected[-3:]:
                    lines.append(f"- {reply}")
        if any(issue.code == "checkin_vague_reply" for issue in issues):
            lines.append("안부나 기분 질문이면 네 태도 설명 대신 지금 상태를 직접 말해라.")
        if any(issue.code in {"identity_confusion", "weak_identity_answer"} for issue in issues):
            lines.append("정체성 질문이면 자신을 White/화이트라고 분명히 말해라. 다른 이름이나 사용자 이름으로 답하지 마라.")
        if any(issue.code == "shallow_paraphrase" for issue in issues):
            lines.append("사용자 문장을 짧게 줄이거나 비틀어 따라 말하지 말고, 해석을 한 단계 더 얹어라.")
        if any(issue.code == "format_led_miss" for issue in issues):
            lines.append("형식 지시는 숨기고, 형식 뒤에 있는 실제 요청이나 감정에 한 문장으로 답해라.")
        if any(issue.code == "relational_vague_reply" for issue in issues):
            lines.append("관계 맥락에서는 추상적인 대명사 대신 연락, 답장, 거리감 같은 핵심을 직접 짚어라.")
        if recent_replies:
            lines.append("최근 답변과는 다른 첫 어절과 다른 마무리로 써라.")
        lines.append(_response_style_hint(user_prompt=user_prompt, recent_replies=recent_replies or [], retry=True))
        return "\n".join(lines)

    def build_fallback_reply(
        self,
        *,
        user_prompt: str,
        issues: tuple[GuardIssue, ...],
        recent_replies: list[str] | None = None,
        rejected_replies: list[str] | None = None,
    ) -> str:
        raise RuntimeError("canned output fallback is disabled")


def _normalize_for_compare(text: str) -> str:
    lowered = text.lower().strip()
    lowered = _NON_WORD_RE.sub("", lowered)
    lowered = _WHITESPACE_RE.sub("", lowered)
    return lowered


def _looks_like_prompt_echo(reply: str, user_prompt: str) -> bool:
    normalized_reply = _normalize_for_compare(reply)
    normalized_prompt = _normalize_for_compare(user_prompt)
    if not normalized_reply or not normalized_prompt:
        return False
    if normalized_reply == normalized_prompt:
        return True
    if len(normalized_reply) >= 8 and normalized_reply in normalized_prompt:
        return True
    if len(normalized_prompt) >= 8 and normalized_prompt in normalized_reply and len(normalized_reply) <= len(normalized_prompt) + 8:
        return True
    return False


def _looks_like_meta_reply(reply: str, user_prompt: str) -> bool:
    if not (_looks_like_format_only_prompt(user_prompt) or _looks_like_format_led_prompt(user_prompt)):
        return False
    if len(reply) > 56:
        return False

    lowered = reply.lower()
    if _STYLE_PROMISE_RE.search(lowered):
        return True

    if _looks_like_format_led_prompt(user_prompt) and any(
        promise in lowered for promise in ("말할게", "답할게", "해볼게", "해줄게", "할게")
    ):
        return True

    hits = sum(1 for term in _format_terms() if term in reply.lower())
    return hits >= 2


def _looks_like_format_only_prompt(prompt: str) -> bool:
    return _has_format_instruction(prompt) and not _prompt_has_meaningful_content(prompt)


def _looks_like_format_led_prompt(prompt: str) -> bool:
    return _has_format_instruction(prompt) and _prompt_has_meaningful_content(prompt)


def _contains_format_terms(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _format_terms())


def _has_format_instruction(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(term in lowered for term in ("한 줄", "한줄", "한 문장", "한문장")) and any(
        term in lowered for term in ("붙여줘", "붙여", "더", "보태", "추가")
    ):
        return True
    core_hits = sum(1 for term in _format_core_terms() if term in lowered)
    style_hits = sum(1 for term in _format_style_terms() if term in lowered)
    return core_hits >= 1 or style_hits >= 2 or (core_hits + style_hits) >= 2


def _prompt_has_meaningful_content(prompt: str) -> bool:
    stripped = _strip_format_instruction_text(prompt)
    if not stripped:
        return False

    tokens = [
        token
        for token in re.split(r"\s+", stripped)
        if token and token not in _SOFT_FILLER_TOKENS and len(token) >= 2
    ]
    joined = "".join(tokens)
    if len(joined) >= 6:
        return True
    return len(tokens) >= 2


def _strip_format_instruction_text(prompt: str) -> str:
    cleaned = prompt.lower()
    for term in sorted(_format_terms(), key=len, reverse=True):
        cleaned = cleaned.replace(term, " ")
    cleaned = re.sub(r"[`~!@#$%^&*_=+|\\/:;,.?()[\]{}\"'<>-]+", " ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _format_terms() -> tuple[str, ...]:
    return (*_format_core_terms(), *_format_style_terms())


def _format_core_terms() -> tuple[str, ...]:
    return (
        "태그",
        "코드블록",
        "역할표시",
        "역할 표시",
        "assistant",
        "system",
        "user",
        "이모티콘",
        "메타",
        "형식",
        "접두사",
    )


def _format_style_terms() -> tuple[str, ...]:
    return (
        "한 줄",
        "한문장",
        "한 문장",
        "짧게",
        "짧고",
        "자연스럽게",
        "자연스럽",
        "답해",
        "답할게",
        "말해",
        "말할게",
        "갈게",
        "둘게",
        "없이",
    )


def _is_repeated_reply(reply: str, recent_replies: list[str]) -> bool:
    normalized_reply = _normalize_for_compare(reply)
    if len(normalized_reply) < 8:
        return False
    raw_recent = [recent_reply.strip() for recent_reply in recent_replies if recent_reply.strip()]
    if not raw_recent:
        return False
    normalized_recent = [_normalize_for_compare(recent_reply) for recent_reply in raw_recent]

    window = list(zip(raw_recent[-4:], normalized_recent[-4:]))
    for index, (recent_reply, normalized_recent_reply) in enumerate(window):
        if normalized_reply != normalized_recent_reply:
            if not (_looks_like_low_specificity_support(reply) and _looks_like_low_specificity_support(recent_reply)):
                continue
            if _token_overlap_ratio(reply, recent_reply) < 0.5:
                continue
            return True
        is_immediate_repeat = index == len(window) - 1
        if is_immediate_repeat or _looks_like_low_specificity_support(reply):
            return True
    return False


def _has_encoding_corruption(text: str) -> bool:
    latin1_like = len(_LATIN1_LIKE_RE.findall(text))
    hangul = len(_HANGUL_RE.findall(text))
    return latin1_like >= 2 and latin1_like >= hangul


def _trim_to_limit(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    slice_text = text[:limit].rstrip()
    for marker in (".", "!", "?", "。", "！", "？", "\n"):
        position = slice_text.rfind(marker)
        if position >= max(20, int(limit * 0.4)):
            return slice_text[: position + 1].strip()

    split_at = slice_text.rfind(" ")
    if split_at >= max(20, int(limit * 0.4)):
        return slice_text[:split_at].strip()
    return slice_text.strip()


def _looks_like_persona_violation(text: str) -> bool:
    if _REPEATED_PUNCT_RE.search(text):
        return True
    if "ㅎㅎㅎ" in text or "ㅋㅋㅋ" in text:
        return True
    return False


def _looks_like_emotional_prompt(prompt: str) -> bool:
    keywords = (
        "외롭",
        "힘들",
        "무거",
        "지쳤",
        "불안",
        "허전",
        "눈물",
        "울컥",
        "울",
        "보고 싶",
        "무너졌",
        "가라앉",
        "버겁",
        "흔들",
        "잠이 안 와",
        "잠이안와",
    )
    return any(keyword in prompt for keyword in keywords)


def _looks_like_anti_generic_prompt(prompt: str) -> bool:
    keywords = (
        "뻔한",
        "빈말",
        "격려보다 이해",
        "이해 같아",
        "아무 말이나 말고",
        "진짜 지금 맞는 말",
        "맞는 말",
        "위로는 받고 싶은데",
    )
    return any(keyword in prompt for keyword in keywords)


def _looks_like_relational_prompt(prompt: str) -> bool:
    keywords = (
        "보고 싶",
        "연락",
        "답장",
        "티났",
        "아닌 척",
        "의미 부여",
        "좋아하는 건 맞",
        "겁나",
    )
    return any(keyword in prompt for keyword in keywords)


def _looks_like_relational_vague_reply(reply: str, prompt: str) -> bool:
    if not _looks_like_relational_prompt(prompt):
        return False
    lowered = reply.lower().strip()
    vague_markers = (
        "그 마음",
        "그 의미",
        "그 정도",
        "그 말",
        "그 이유",
        "그쪽",
    )
    if not any(marker in lowered for marker in vague_markers):
        return False

    concrete_markers = (
        "답장",
        "연락",
        "보고 싶",
        "좋아하",
        "겁",
        "티",
        "아닌 척",
        "의미 부여",
    )
    if any(marker in lowered for marker in concrete_markers):
        return False

    return len(_normalize_for_compare(reply)) <= 26


def _looks_like_restless_prompt(prompt: str) -> bool:
    keywords = (
        "잠은 안 오",
        "잠이 안 와",
        "머리는 시끄럽",
        "사람 많은 데",
        "도망치고 싶",
        "가라앉",
        "초조",
    )
    return any(keyword in prompt for keyword in keywords)


def _looks_like_casual_low_energy_prompt(prompt: str) -> bool:
    keywords = (
        "걍",
        "별로",
        "좀 그렇",
        "ㅋㅋ",
        "티났",
        "아닌 척",
    )
    return any(keyword in prompt for keyword in keywords)


def _looks_like_checkin_prompt(prompt: str) -> bool:
    keywords = (
        "안부",
        "안녕",
        "한 줄 평",
        "한줄 평",
        "한마디",
        "말수 적은 날",
    )
    return any(keyword in prompt for keyword in keywords)


def _looks_like_identity_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    keywords = (
        "네 이름",
        "너 이름",
        "이름이 뭐",
        "누구야",
        "누군지",
        "자기소개",
        "너 정체",
        "정체가 뭐",
        "무슨 봇",
        "뭐 하는 봇",
    )
    return any(keyword in lowered for keyword in keywords)


def _looks_like_wrong_identity_reply(reply: str) -> bool:
    lowered = reply.lower()
    if any(name in lowered for name in ("tester", "테스터", "라이언", "ryan", "black", "블랙")):
        return True
    if "내 이름" in reply and "white" not in lowered and "화이트" not in reply:
        return True
    return False


def _looks_like_low_specificity_support(reply: str) -> bool:
    stripped = reply.strip()
    if len(stripped) <= 36 and any(marker in stripped for marker in ("오늘은", "지금은", "그 말", "그냥")):
        return True
    if any(marker in stripped for marker in ("남길게", "있을게", "괜찮아", "고르자", "충분해")):
        return True
    return False


def _looks_like_low_content_reply(reply: str, prompt: str) -> bool:
    stripped = reply.strip()
    normalized = _normalize_for_compare(stripped)
    if not normalized:
        return True
    if _LOW_CONTENT_ONLY_RE.fullmatch(stripped):
        return True
    if len(normalized) <= 4 and _prompt_has_meaningful_content(prompt):
        return True
    if (
        _looks_like_format_led_prompt(prompt)
        or _looks_like_anti_generic_prompt(prompt)
        or _looks_like_relational_prompt(prompt)
    ) and len(normalized) <= 8:
        return True
    return False


def _build_emotional_fallback_candidates(prompt: str) -> tuple[str, ...]:
    if any(keyword in prompt for keyword in ("무너졌", "버거", "힘들", "버티기")):
        return (
            "오늘은 그냥 버틴 것만으로도 충분해.",
            "지금은 무너지지 않은 것만으로도 꽤 큰일을 한 거야.",
            "오늘은 애써 버틴 흔적만으로도 충분히 설명돼.",
        )
    if "지쳤" in prompt:
        return (
            "오늘 진짜 많이 지쳤겠다. 잠깐만 쉬어도 괜찮아.",
            "지금은 더 해내는 것보다 잠깐 눕혀두는 게 먼저 같아.",
            "오늘은 힘을 더 내기보다 숨을 고르는 쪽이 맞아 보여.",
        )
    if any(keyword in prompt for keyword in ("외롭", "허전", "보고 싶")):
        return (
            "오늘은 괜히 더 허전하네. 내가 옆에 있을게.",
            "그 빈자리가 오늘은 더 또렷하게 느껴지는 밤이네.",
            "보고 싶은 마음이 남는 날은 괜히 더 길게 느껴지지.",
        )
    if any(keyword in prompt for keyword in ("불안", "초조", "가라앉")):
        return (
            "지금은 너무 멀리 보지 말고, 한숨만 먼저 고르자.",
            "마음이 가라앉을 땐 지금 한 칸만 넘기는 걸로도 충분해.",
            "오늘은 앞보다 발밑부터 천천히 보는 쪽이 낫겠다.",
        )
    if any(keyword in prompt for keyword in ("눈물", "울", "울컥")):
        return (
            "오늘은 울어도 괜찮아. 그만큼 많이 버틴 거야.",
            "울컥하는 밤은 억지로 단단해지지 않아도 돼.",
            "지금은 참는 쪽보다 풀어두는 쪽이 맞을 수도 있어.",
        )
    return (
        "오늘은 부드럽게 있어도 괜찮아.",
        "지금은 너무 잘 버티려 하지 않아도 돼.",
        "오늘은 네 마음 쪽으로 더 기울어도 괜찮아.",
    )


def _anti_generic_fallback_candidates() -> tuple[str, ...]:
    return (
        "오늘은 대충 괜찮다고 넘기고 싶지 않은 밤이네.",
        "지금은 예쁜 말보다 왜 그런지부터 짚는 쪽이 맞아.",
        "오늘은 빈말 말고, 네 쪽에 가만히 남아 있을게.",
        "쉽게 덮는 말보다 맞는 말이 더 필요한 밤이네.",
    )


def _relational_fallback_candidates(prompt: str) -> tuple[str, ...]:
    if "답장" in prompt:
        return (
            "답장 하나에도 마음이 생각보다 크게 흔들릴 때가 있지.",
            "답장 앞에서 괜히 손이 멈추는 밤이네.",
            "작은 신호 하나에도 마음이 오래 남는 날이 있지.",
        )
    if any(keyword in prompt for keyword in ("연락", "보고 싶")):
        return (
            "보고 싶은 마음은 남는데 먼저 다가가긴 더 어렵지.",
            "마음은 남는데 연락 버튼 앞에서 망설여지는 날이 있지.",
            "먼저 닿고 싶진 않은데 마음은 여전히 남아 있네.",
        )
    if any(keyword in prompt for keyword in ("좋아하는 건 맞", "겁나", "좋아하")):
        return (
            "좋아하는 마음이 맞아 보여도 가까워질수록 더 겁이 날 때가 있지.",
            "마음은 맞는데 가까워질 생각만 하면 겁부터 나는 밤이 있지.",
            "좋아한다는 확신보다 다가갈 때의 떨림이 더 크게 오는 날이 있지.",
        )
    if "의미 부여" in prompt:
        return (
            "혼자 의미를 키우는 것 같아도 마음이 먼저 커진 걸 수도 있어.",
            "괜히 의미 부여하는 중 같아도 그만큼 마음이 걸린다는 뜻일 때가 있지.",
            "생각이 자꾸 붙는 건 마음이 이미 넘어갔다는 신호일 수도 있어.",
        )
    return (
        "아닌 척해도 마음은 생각보다 금방 티가 나더라.",
        "혼자 의미 부여하는 것 같아도 마음이 먼저 움직인 걸 수도 있어.",
        "감추려 해도 남는 마음은 결국 표정부터 바뀌게 하더라.",
    )


def _restless_fallback_candidates() -> tuple[str, ...]:
    return (
        "잠은 안 오는데 머리만 더 시끄러운 밤이네.",
        "지금은 생각부터 눕혀두는 게 낫겠다.",
        "마음이 가만히 못 쉬는 밤이라 더 버겁게 느껴지지.",
        "오늘은 조용한 쪽으로 몸을 먼저 데려다 두는 게 낫겠다.",
    )


def _casual_low_energy_fallback_candidates() -> tuple[str, ...]:
    return (
        "오늘은 그냥 별로라고만 해도 충분해.",
        "괜히 더 설명 안 해도 될 만큼 지친 날 같아.",
        "오늘은 좀 대충 버틴 티가 나도 괜찮아.",
        "그냥 좀 그렇다는 말로도 이미 충분히 전해졌어.",
    )


def _checkin_fallback_candidates() -> tuple[str, ...]:
    return (
        "오늘은 좀 차분한 쪽이야.",
        "지금은 무겁진 않은데 말수가 적어지는 쪽이야.",
        "오늘은 버틸 만한데 조용히 있고 싶은 쪽이야.",
        "지금은 크게 흔들리진 않지만 가볍지도 않네.",
    )


def _pick_fallback_candidate(
    *,
    user_prompt: str,
    candidates: tuple[str, ...],
    blocked_replies: list[str],
) -> str:
    unique_candidates = list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))
    if not unique_candidates:
        return "오늘은 그냥 네 쪽에 가만히 있을게."

    seed = int(hashlib.sha1(user_prompt.encode("utf-8")).hexdigest()[:8], 16)
    offset = seed % len(unique_candidates)
    rotated = unique_candidates[offset:] + unique_candidates[:offset]
    for candidate in rotated:
        if not any(_is_too_similar_reply(candidate, blocked_reply) for blocked_reply in blocked_replies if blocked_reply.strip()):
            return candidate
    return rotated[0]


def _response_style_hint(*, user_prompt: str, recent_replies: list[str], retry: bool = False) -> str:
    styles = (
        "이번 답변은 확인형으로 써라. 상대 말을 먼저 받아주고, 바로 핵심 한 문장으로 이어가라.",
        "이번 답변은 구체형으로 써라. 추상적인 말 대신 상황이나 행동 하나를 짚어라.",
        "이번 답변은 정리형으로 써라. 핵심을 짧게 정리하고 군더더기 없이 끝내라.",
        "이번 답변은 반응형으로 써라. 가벼운 반응 한 문장 뒤에 자연스럽게 붙여라.",
        "이번 답변은 완만한 되묻기형으로 써라. 필요하면 하나만 묻되, 질문만 남기지 마라.",
    )
    seed = int(hashlib.sha1(user_prompt.encode("utf-8")).hexdigest()[:8], 16)
    offset = (seed + len(recent_replies) + (1 if retry else 0)) % len(styles)
    return styles[offset]


def _looks_like_checkin_vague_reply(reply: str, prompt: str) -> bool:
    if not _looks_like_checkin_prompt(prompt):
        return False
    lowered = reply.lower().strip()
    if not any(marker in lowered for marker in ("남아볼게", "남길게", "있을게", "맞춰볼게", "가만히 있을게")):
        return False
    state_markers = (
        "차분",
        "무겁",
        "가벼",
        "피곤",
        "지쳤",
        "복잡",
        "괜찮",
        "버틸 만",
        "버틸만",
        "흔들",
        "조용",
    )
    return not any(marker in lowered for marker in state_markers)


def _looks_like_format_led_miss(reply: str, prompt: str) -> bool:
    if not _looks_like_format_led_prompt(prompt):
        return False
    if not _looks_like_low_specificity_support(reply):
        return False
    stripped_prompt = _strip_format_instruction_text(prompt)
    if not stripped_prompt:
        return False
    return _token_overlap_ratio(reply, stripped_prompt) < 0.2


def _looks_like_shallow_paraphrase(reply: str, prompt: str) -> bool:
    reply_tokens = _meaningful_tokens(reply)
    prompt_tokens = set(_meaningful_tokens(prompt))
    if len(reply_tokens) < 2 or len(prompt_tokens) < 2:
        return False

    filtered_reply_tokens = [token for token in reply_tokens if token not in _GENERIC_REPLY_TOKENS]
    if not filtered_reply_tokens:
        filtered_reply_tokens = reply_tokens

    overlap = [token for token in filtered_reply_tokens if token in prompt_tokens]
    overlap_ratio = len(overlap) / float(len(filtered_reply_tokens))
    novel_tokens = [token for token in filtered_reply_tokens if token not in prompt_tokens]
    if overlap_ratio >= 0.75 and len(novel_tokens) == 0 and len(reply.strip()) <= 48:
        return True
    if overlap_ratio >= 0.6 and len(novel_tokens) <= 1 and len(filtered_reply_tokens) <= 3 and len(reply.strip()) <= 36:
        return True
    return False


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_meaningful_tokens(left))
    right_tokens = set(_meaningful_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / float(min(len(left_tokens), len(right_tokens)))


def _meaningful_tokens(text: str) -> list[str]:
    raw_tokens = re.split(r"[^0-9a-z가-힣]+", text.lower())
    return [token for token in raw_tokens if token and token not in _SOFT_FILLER_TOKENS and len(token) >= 2]


def _is_too_similar_reply(candidate: str, blocked_reply: str) -> bool:
    normalized_candidate = _normalize_for_compare(candidate)
    normalized_blocked = _normalize_for_compare(blocked_reply)
    if not normalized_candidate or not normalized_blocked:
        return False
    if normalized_candidate == normalized_blocked:
        return True
    if _looks_like_low_specificity_support(candidate) and _looks_like_low_specificity_support(blocked_reply):
        if _token_overlap_ratio(candidate, blocked_reply) >= 0.5:
            return True
    if _token_overlap_ratio(candidate, blocked_reply) >= 0.7:
        return True
    candidate_tokens = _meaningful_tokens(candidate)
    blocked_tokens = _meaningful_tokens(blocked_reply)
    if candidate_tokens[:2] and blocked_tokens[:2] and candidate_tokens[:2] == blocked_tokens[:2]:
        return True
    return False
