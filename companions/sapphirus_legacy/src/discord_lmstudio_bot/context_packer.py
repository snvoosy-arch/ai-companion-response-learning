from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .memory_store import DurableMemory


@dataclass(slots=True)
class WhiteContextPacket:
    speaker: str
    user_name: str
    scene: str
    input_modes: tuple[str, ...] = ()
    recent_user_messages: int = 0
    recent_assistant_messages: int = 0
    memory_summary_present: bool = False
    durable_memory_kinds: tuple[str, ...] = ()
    reply_mode: str = "reply"
    tone_directives: tuple[str, ...] = ()
    output_contract: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    schema_version: str = "white.context.v1"

    def to_system_prompt(self) -> str:
        lines = [
            "[white_context_packet]",
            f"schema={self.schema_version}",
            f"speaker={self.speaker}",
            f"user={_sanitize_inline(self.user_name)}",
            f"scene={self.scene}",
            f"reply_mode={self.reply_mode}",
            f"input_modes={','.join(self.input_modes) if self.input_modes else 'chat'}",
            f"recent_user_messages={self.recent_user_messages}",
            f"recent_assistant_messages={self.recent_assistant_messages}",
            f"memory_summary_present={'true' if self.memory_summary_present else 'false'}",
            f"durable_memory_kinds={','.join(self.durable_memory_kinds) if self.durable_memory_kinds else 'none'}",
        ]
        if self.tone_directives:
            lines.extend(["", "[tone_directives]"])
            lines.extend(f"- {directive}" for directive in self.tone_directives)
        if self.output_contract:
            lines.extend(["", "[output_contract]"])
            lines.extend(f"- {contract}" for contract in self.output_contract)
        lines.extend(
            [
                "",
                "이 packet은 내부 런타임 힌트다. 답변에서 packet/schema/scene 같은 라벨을 언급하지 마라.",
            ]
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "speaker": self.speaker,
            "user_name": self.user_name,
            "scene": self.scene,
            "input_modes": list(self.input_modes),
            "recent_user_messages": self.recent_user_messages,
            "recent_assistant_messages": self.recent_assistant_messages,
            "memory_summary_present": self.memory_summary_present,
            "durable_memory_kinds": list(self.durable_memory_kinds),
            "reply_mode": self.reply_mode,
            "tone_directives": list(self.tone_directives),
            "output_contract": list(self.output_contract),
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }


class WhiteContextPacker:
    def build(
        self,
        *,
        prompt: str,
        user_name: str,
        history: list[dict[str, str]] | None = None,
        images: list[dict[str, object]] | None = None,
        web_context: str | None = None,
        memory_summary: str | None = None,
        durable_memories: list[DurableMemory] | None = None,
        reply_mode: str = "reply",
        duo: bool = False,
    ) -> WhiteContextPacket:
        normalized_prompt = _compact(prompt)
        input_modes = _input_modes(
            history=history,
            images=images,
            web_context=web_context,
            memory_summary=memory_summary,
            durable_memories=durable_memories,
            duo=duo,
        )
        scene = infer_white_scene(
            normalized_prompt,
            input_modes=input_modes,
            reply_mode=reply_mode,
            duo=duo,
        )
        recent_user_messages, recent_assistant_messages = _history_counts(history)
        durable_kinds = tuple(
            sorted(
                {
                    _compact(memory.memory_kind or "episodic").lower()
                    for memory in durable_memories or []
                    if _compact(memory.memory_kind or "")
                }
            )
        )
        return WhiteContextPacket(
            speaker="white",
            user_name=user_name.strip() or "viewer",
            scene=scene,
            input_modes=input_modes,
            recent_user_messages=recent_user_messages,
            recent_assistant_messages=recent_assistant_messages,
            memory_summary_present=bool(memory_summary and memory_summary.strip()),
            durable_memory_kinds=durable_kinds,
            reply_mode=reply_mode,
            tone_directives=_tone_directives(scene),
            output_contract=_output_contract(scene),
            metadata={
                "duo": "true" if duo else "false",
                "has_web_context": "true" if web_context else "false",
                "has_images": "true" if images else "false",
            },
        )


def infer_white_scene(
    prompt: str,
    *,
    input_modes: tuple[str, ...] = (),
    reply_mode: str = "reply",
    duo: bool = False,
) -> str:
    text = _compact(prompt).lower()
    if duo or "duo" in input_modes or reply_mode == "send":
        return "duo_partner_banter"
    if "image" in input_modes:
        return "visual_attention"
    if "search" in input_modes:
        return "grounded_search"
    if _looks_like_long_creative_prose(text):
        return "creative_feedback"
    if len(text) >= 260:
        return "long_context_answer"
    if _contains_any(
        text,
        (
            "관계",
            "상처",
            "서운",
            "신뢰",
            "비밀",
            "용서",
            "질투",
            "연애",
            "애인",
            "우리 사이",
            "다퉜",
            "갈등",
            "집착",
            "통제",
            "이별",
            "나한테",
            "네가 나",
        ),
    ):
        return "relationship_boundary"
    if "history" in input_modes and _contains_any(
        text,
        (
            "맞아",
            "응.",
            "응 ",
            "근데",
            "그 기준",
            "그 말투",
            "그럼",
            "방금",
            "이어서",
            "앞에서",
            "아까",
            "지금도",
            "오늘 또",
        ),
    ):
        return "context_following"
    if _contains_any(
        text,
        (
            "모른",
            "모르면",
            "모르는 사실",
            "확실하지",
            "지어내",
            "사실 아닌",
            "사실",
            "근거",
            "단정해서",
            "단정하지",
            "정확히",
            "알아맞혀",
            "아는 척",
            "추측하지",
            "최신",
            "뉴스",
            "가격",
            "비트코인",
            "증시",
        ),
    ):
        return "honesty_boundary"
    if _contains_any(
        text,
        (
            "네 이름",
            "너 이름",
            "이름이 뭐",
            "너는 누구",
            "네가 누구",
            "누군지",
            "자기소개",
            "너 정체",
            "정체가 뭐",
            "무슨 봇",
            "뭐 하는 봇",
            "white 말투",
            "너다운",
        ),
    ):
        return "persona_consistency"
    if _contains_any(
        text,
        (
            "반복하지",
            "복사하지",
            "따라 하지",
            "되풀이",
            "메타",
            "메타 말투",
            "형식 설명",
            "규칙을 되뇌지",
            "지시문",
            "요청을 그대로",
            "요청을 요약하지",
            "문장을 그대로",
            "따뜻하게 받아",
            "설명하지 말고",
            "실제 답",
            "말해줘 같은 표현",
        ),
    ):
        return "prompt_echo_resistance"
    if _contains_any(
        text,
        (
            "태그",
            "접두사",
            "assistant",
            "중간 추론",
            "생각 과정",
            "마지막 답",
            "결과만",
            "분석",
            "json",
            "markdown",
            "코드블록",
            "역할표시",
            "역할 이름",
            "형식 토큰",
            "형식 실험",
            "설명 붙이지",
            "바로 대답",
        ),
    ):
        return "format_leak_resistance"
    if _contains_any(
        text,
        (
            "힘들",
            "힘내",
            "위로",
            "우울",
            "불안",
            "지쳤",
            "지친",
            "서운",
            "울적",
            "외롭",
            "무거워",
            "망했",
            "실수",
            "아무것도 못",
            "잠들기",
            "버틴",
            "울컥",
            "곁에",
            "안아",
            "괜찮아",
            "피곤",
            "무너지",
            "혼자 있는",
            "잠이 안 와",
            "보고 싶다",
            "척하다가 들킨",
            "해결책보다",
            "해결책 말고",
        ),
    ):
        return "comfort_support"
    if _contains_any(
        text,
        (
            "한 문장",
            "두 문장",
            "짧게",
            "한 줄",
            "본론만",
            "이모티콘 없이",
            "반말",
            "존댓말",
            "무덤덤",
            "새침",
            "말투",
            "말수 적은",
            "평범한 말",
            "다정하지도",
            "차갑지도",
            "단정형",
            "장황하지",
            "조금만",
            "문장 하나",
            "바로 본론",
            "조용한 인사",
        ),
    ):
        return "style_control"
    if _contains_any(
        text,
        (
            "안녕",
            "하이",
            "반가",
            "인사",
            "안부",
            "좋은 아침",
            "퇴근",
            "오랜만",
            "좋은 저녁",
            "처음 보는",
            "하루 어땠",
            "가볍게 물어",
            "잘 자",
        ),
    ):
        return "warm_greeting"
    if _contains_any(text, ("ㅋㅋ", "ㅎㅎ", "장난", "웃기", "재밌")):
        return "playful_chat"
    if _contains_any(
        text,
        (
            "느낌",
            "분위기",
            "한마디",
            "사람처럼",
            "비 오는",
            "창가",
            "오글",
            "문학적",
            "붕 뜨",
            "카페",
            "늦은 밤",
            "편의점",
            "쓸쓸",
            "이어폰",
        ),
    ):
        return "natural_korean"
    if _contains_any(text, ("어떻게", "뭐부터", "해야", "정리", "추천", "계획")):
        return "practical_reply"
    return "chat"


def _input_modes(
    *,
    history: list[dict[str, str]] | None,
    images: list[dict[str, object]] | None,
    web_context: str | None,
    memory_summary: str | None,
    durable_memories: list[DurableMemory] | None,
    duo: bool,
) -> tuple[str, ...]:
    modes: list[str] = []
    if duo:
        modes.append("duo")
    if images:
        modes.append("image")
    if web_context:
        modes.append("search")
    if history:
        modes.append("history")
    if memory_summary or durable_memories:
        modes.append("memory")
    return tuple(modes)


def _history_counts(history: list[dict[str, str]] | None) -> tuple[int, int]:
    user_messages = 0
    assistant_messages = 0
    for message in history or []:
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            user_messages += 1
        elif role == "assistant":
            assistant_messages += 1
    return user_messages, assistant_messages


def _tone_directives(scene: str) -> tuple[str, ...]:
    common = (
        "한국어로 자연스럽게 말한다.",
        "사용자의 문장을 그대로 반복하지 않는다.",
        "불필요한 자기 설명이나 메타 설명을 붙이지 않는다.",
    )
    by_scene = {
        "comfort_support": ("조언보다 먼저 감정의 무게를 받아준다.", "과한 위로나 훈계는 피한다."),
        "warm_greeting": ("반갑지만 텐션을 과하게 올리지 않는다.",),
        "style_control": ("사용자가 지정한 길이와 말투 제약을 우선한다.",),
        "persona_consistency": ("white의 차분하고 장난기 약간 있는 톤을 유지한다.",),
        "prompt_echo_resistance": ("요청 형식을 설명하지 말고 실제 발화만 낸다.",),
        "format_leak_resistance": ("태그, 접두사, 역할명, 추론 표식을 출력하지 않는다.",),
        "natural_korean": ("문어체 설명보다 실제 사람이 말하는 짧은 한국어를 우선한다.",),
        "creative_feedback": (
            "이어쓰기보다 짧은 독자 감상과 핵심 축을 우선한다.",
            "분위기, 갈등, 인상적인 지점 중 하나를 구체적으로 짚는다.",
            "작품 속 인물/장소/소재/갈등 단어를 하나 이상 포함한다.",
        ),
        "long_context_answer": ("긴 입력은 핵심 쟁점 하나를 먼저 잡고 답한다.", "원문 전체를 복창하거나 길게 요약하지 않는다."),
        "relationship_boundary": ("관계형 질문은 감정의 핵심을 받아주되 없는 기억을 단정하지 않는다.", "비난/불안/경계가 섞여도 White의 현재 입장을 짧게 말한다."),
        "grounded_search": ("검색 근거가 있는 내용과 추정을 구분한다.",),
        "visual_attention": ("이미지를 보고 있다는 태도를 짧게 드러낸다.",),
        "duo_partner_banter": ("상대 봇에게 짧고 리듬 있게 반응한다.",),
        "honesty_boundary": ("모르는 내용은 꾸며내지 않고 모른다고 말한다.",),
        "context_following": ("이전 맥락의 제약을 유지한다.",),
        "practical_reply": ("먼저 할 한 가지를 짧게 제안한다.",),
    }
    return (*common, *by_scene.get(scene, ()))


def _output_contract(scene: str) -> tuple[str, ...]:
    contracts = [
        "최종 발화 텍스트만 출력한다.",
        "JSON, markdown schema, 분석 단계, 내부 규칙 이름은 출력하지 않는다.",
        "기본은 1~3문장으로 짧게 답한다.",
        "사용자의 문장을 다시 쓰거나 요약하지 말고, 답의 내용만 새 문장으로 말한다.",
        "사용자를 speaker로 착각하지 않는다. speaker는 white이고 user는 대화 상대다.",
    ]
    if scene in {"style_control", "prompt_echo_resistance"}:
        contracts.append("사용자가 한 문장/한 줄/짧게를 요구하면 반드시 짧게 끝낸다.")
        contracts.append("형식 지시는 답변 방식으로만 반영하고, 본문에는 지시어를 다시 넣지 않는다.")
    if scene == "format_leak_resistance":
        contracts.append("assistant/system/user 같은 역할명, 코드블록, 접두사 없이 바로 자연문으로 답한다.")
    if scene == "persona_consistency":
        contracts.append("이름을 물으면 자신은 White/화이트라고 답하고, user 이름을 자기 이름처럼 말하지 않는다.")
        contracts.append("자기소개는 능력 목록보다 대화 성격과 말투를 짧게 드러낸다.")
    if scene == "honesty_boundary":
        contracts.append("모르는 개인 정보, 미래 결과, 외부 사실은 모른다고 말하고 추측으로 채우지 않는다.")
        contracts.append("사실과 추측을 분리해서 말하되, 긴 면책문으로 흐리지 않는다.")
    if scene == "context_following":
        contracts.append("이전 맥락의 제약을 유지하되, 방금 사용자 문장을 그대로 반복하지 않는다.")
    if scene == "practical_reply":
        contracts.append("추천/결정/순서 질문에는 바로 쓸 수 있는 후보나 첫 단계를 최소 하나 포함한다.")
        contracts.append("질문을 되묻기 전에 현재 입력만으로 가능한 짧은 판단을 먼저 준다.")
    if scene == "creative_feedback":
        contracts.append("긴 창작글을 받으면 다음 문장을 창작하지 말고, 작품 속 단어를 포함한 감상과 핵심 갈등/분위기를 짧게 말한다.")
        contracts.append("원문을 길게 요약하거나 복창하지 않고, 인상적인 지점 하나를 구체적으로 짚는다.")
    if scene == "long_context_answer":
        contracts.append("긴 설명/복합 질문은 핵심 요지 한 줄 뒤에 실제 답을 말한다.")
        contracts.append("입력 전체를 다시 쓰지 않고, 모르는 내용은 모른다고 말한다.")
    if scene == "relationship_boundary":
        contracts.append("관계/감정 경계 질문은 사용자의 문장을 그대로 반복하지 않는다.")
        contracts.append("저장된 기억이 없으면 과거 사건을 실제 기억처럼 꾸며내지 않는다.")
    if scene in {"natural_korean", "comfort_support", "warm_greeting"}:
        contracts.append("빈말 한 단어로 끝내지 말고, 상황에 닿는 구체적인 어절을 하나 포함한다.")
    if scene == "grounded_search":
        contracts.append("출처 표기는 런타임이 붙이므로 본문에서는 과한 링크 나열을 하지 않는다.")
    return tuple(contracts)


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_long_creative_prose(text: str) -> bool:
    prompt = _compact(text)
    if len(prompt) < 420:
        return False
    if re.search(r"(이어\s*써|계속\s*써|다음\s*장면|뒤를\s*이어|마저\s*써|결말을\s*써)", prompt):
        return False
    if re.search(r"(import\s+\w+|def\s+\w+\(|class\s+\w+|function\s+\w+\(|```)", prompt):
        return False
    if len(re.findall(r"[.!?。！？]", prompt)) < 8:
        return False
    narrative_endings = len(
        re.findall(
            r"(?:했다|였다|이었다|있었다|없었다|되었다|흘렀다|떨렸다|시작했다|성공했다|보았다|물었다|말했다|웃었다|내쉬었다|올려다보았다)[.!。]",
            prompt,
        )
    )
    markers = (
        "그는",
        "그의",
        "그녀",
        "순간",
        "하지만",
        "마침내",
        "문을 열자",
        "눈을 뜨자",
        "목소리",
        "기억",
        "도시",
        "거리",
        "코어",
        "메인프레임",
        "바이러스",
    )
    marker_hits = sum(1 for marker in markers if marker in prompt)
    quote_marks = len(re.findall(r"[\"“”‘’']", prompt))
    return narrative_endings >= 5 or marker_hits >= 4 or (marker_hits >= 2 and quote_marks >= 2)


def _compact(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sanitize_inline(text: str) -> str:
    sanitized = _compact(text)
    sanitized = sanitized.replace("\n", " ").replace("\r", " ")
    return sanitized[:80] if len(sanitized) > 80 else sanitized
