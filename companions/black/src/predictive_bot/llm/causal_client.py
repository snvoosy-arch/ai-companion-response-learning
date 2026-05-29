from __future__ import annotations

import asyncio
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_CAUSAL_IMPORT_ERROR: Exception | None = None

try:
    import torch
except Exception as exc:  # pragma: no cover - optional local ML stack
    torch = None
    _CAUSAL_IMPORT_ERROR = exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception as exc:  # pragma: no cover - optional local ML stack
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    if _CAUSAL_IMPORT_ERROR is None:
        _CAUSAL_IMPORT_ERROR = exc

from predictive_bot.llm.kobart_client import KoBartGenerationClient


class CausalLMGenerationClient:
    """Chat/causal LM wrapper for Black's structured response-plan pipeline."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "auto",
        max_new_tokens: int = 96,
        temperature: float = 0.35,
        top_p: float = 0.9,
        quantization: str = "",
        output_guard_enabled: bool = True,
    ) -> None:
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "Causal LM generation dependencies are unavailable in this Python environment."
            ) from _CAUSAL_IMPORT_ERROR

        self.model_name_or_path = model_name_or_path
        self.device_name = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.05, float(top_p)))
        self.quantization = (quantization or "").lower()
        self.output_guard_enabled = bool(output_guard_enabled)
        self.last_generation_issue: str | None = None
        self.last_raw_generation: str = ""

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.quantization in {"4bit", "int4", "nf4"}:
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes quantization is unavailable in this Python environment.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.device_name == "cuda" else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        elif self.quantization in {"8bit", "int8"}:
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes quantization is unavailable in this Python environment.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
        else:
            if self.device_name == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if "device_map" not in model_kwargs:
            self._model.to(self.device_name)
        self._model.eval()

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(
            self.generate_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def generate_sync(self, *, system_prompt: str, user_prompt: str) -> str:
        facts = self._extract_facts(user_prompt)
        action = str(facts.get("action", "")).strip()
        self.last_generation_issue = None
        self.last_raw_generation = ""

        messages = self._build_messages(system_prompt=system_prompt, facts=facts)
        generated = self._generate_from_messages(messages)
        self.last_raw_generation = generated
        if not self.output_guard_enabled:
            return KoBartGenerationClient._normalize_output(generated)
        try:
            reply = KoBartGenerationClient._finalize_generated(generated, action=action, facts=facts)
            self._validate_draft_preservation(reply=reply, facts=facts)
            self._validate_draft_rewrite_effort(reply=reply, facts=facts)
            return reply
        except RuntimeError as exc:
            first_issue = KoBartGenerationClient._issue_code_from_exception(exc)

        retry_messages = self._build_messages(
            system_prompt=system_prompt,
            facts=facts,
            previous_candidate=generated,
            retry_issue=first_issue,
        )
        retry_generated = self._generate_from_messages(retry_messages, sample=first_issue == "llm_draft_copy")
        self.last_raw_generation = retry_generated
        try:
            reply = KoBartGenerationClient._finalize_generated(retry_generated, action=action, facts=facts)
            self._validate_draft_preservation(reply=reply, facts=facts)
            self._validate_draft_rewrite_effort(reply=reply, facts=facts)
            self.last_generation_issue = f"{first_issue}:recovered"
            return reply
        except RuntimeError as exc:
            retry_issue = KoBartGenerationClient._issue_code_from_exception(exc)
            self.last_generation_issue = f"{first_issue};retry_{retry_issue}"
            return KoBartGenerationClient._normalize_output(retry_generated or generated)

    def _generate_from_messages(self, messages: list[dict[str, str]], *, sample: bool = False) -> str:
        input_ids, attention_mask = self._encode_messages(messages)
        input_ids = input_ids.to(self._model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        input_length = input_ids.shape[-1]
        do_sample = sample or self.temperature > 0.0
        sample_temperature = self._generation_temperature(self.temperature, do_sample=do_sample)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "temperature": sample_temperature,
            "top_p": self.top_p if do_sample else None,
            "repetition_penalty": 1.08,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with torch.no_grad():
            output_ids = self._model.generate(**generation_kwargs)
        new_tokens = output_ids[0][input_length:]
        return self._clean_chat_output(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

    def _encode_messages(self, messages: list[dict[str, str]]):
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return encoded["input_ids"], encoded.get("attention_mask")
        except Exception:
            prompt = self._messages_to_plain_prompt(messages)
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            return encoded["input_ids"], encoded.get("attention_mask")

    @staticmethod
    def _build_messages(
        *,
        system_prompt: str,
        facts: dict[str, Any],
        previous_candidate: str = "",
        retry_issue: str = "",
    ) -> list[dict[str, str]]:
        response_plan = facts.get("response_plan") if isinstance(facts.get("response_plan"), dict) else {}
        draft_utterance = facts.get("draft_utterance") if isinstance(facts.get("draft_utterance"), dict) else {}
        phrasing_plan = facts.get("phrasing_plan") if isinstance(facts.get("phrasing_plan"), dict) else {}
        world_state = facts.get("world_state") if isinstance(facts.get("world_state"), dict) else {}
        action_payload = facts.get("action_payload") if isinstance(facts.get("action_payload"), dict) else {}
        draft_reply = str(draft_utterance.get("draft_reply") or "").strip()
        draft_required_terms = _draft_required_terms(draft_utterance)
        response_plan_for_prompt = response_plan
        if draft_reply and response_plan:
            response_plan_for_prompt = dict(response_plan)
            response_plan_for_prompt["anchor"] = str(draft_utterance.get("anchor") or "")
            response_plan_for_prompt["must_include"] = draft_utterance.get("must_include") or []
        constraints = world_state.get("constraints") if isinstance(world_state, dict) else []
        if not isinstance(constraints, list):
            constraints = []

        plan_lines = [
            f"action: {facts.get('action') or 'unknown'}",
            f"intent: {world_state.get('dominant_intent') or facts.get('intent') or 'unknown'}",
            (
                "user_text: omitted; surface wording must come from draft_utterance.draft_reply"
                if draft_reply
                else f"user_text: {facts.get('user_text') or ''}"
            ),
            f"reason_code: {facts.get('reason_code') or 'none'}",
            f"reason_summary: {facts.get('reason_summary') or facts.get('reason') or 'none'}",
            f"response_plan: {json.dumps(response_plan_for_prompt, ensure_ascii=False)}",
            f"draft_utterance: {json.dumps(draft_utterance, ensure_ascii=False)}",
            f"draft_required_terms: {', '.join(draft_required_terms[:4]) if draft_required_terms else 'none'}",
            (
                "phrasing_plan: omitted; preserve draft wording"
                if draft_reply
                else f"phrasing_plan: {json.dumps(phrasing_plan, ensure_ascii=False)}"
            ),
            f"constraints: {', '.join(str(item) for item in constraints if str(item).strip()) or 'none'}",
        ]
        if action_payload and not draft_reply:
            plan_lines.append(f"action_payload: {json.dumps(action_payload, ensure_ascii=False)}")
        if retry_issue:
            plan_lines.extend(
                [
                    f"retry_issue: {retry_issue}",
                    f"previous_candidate: {previous_candidate}",
                    "retry_rule: do not reuse the failed wording; keep the same anchor and produce a clean final reply.",
                    "retry_rule: keep the semantic content, but change the surface Korean wording or sentence ending.",
                ]
            )
            if retry_issue == "llm_draft_copy":
                plan_lines.append(
                    "retry_rule: the previous candidate copied the draft; rewrite particles, ending, or word order without adding facts."
                )
            if retry_issue == "llm_draft_negation_missing":
                plan_lines.append(
                    "retry_rule: the previous candidate flipped negation; preserve the same negation while still smoothing the sentence."
                )
            if retry_issue in {"llm_draft_anchor_missing", "llm_draft_semantic_drift"}:
                plan_lines.append(
                    "retry_rule: the previous candidate dropped the draft anchor; keep the anchor visible without copying the whole draft verbatim."
                )

        if draft_reply:
            draft_must = ", ".join(draft_required_terms[:4]) if draft_required_terms else "none"
            raw_avoid = draft_utterance.get("avoid") if isinstance(draft_utterance, dict) else []
            avoid_terms = [
                str(item).strip()
                for item in (raw_avoid if isinstance(raw_avoid, list) else [])
                if str(item).strip()
            ][:6]
            compact_response_plan = {
                "action": facts.get("action") or "unknown",
                "stance": response_plan_for_prompt.get("stance") if isinstance(response_plan_for_prompt, dict) else "",
                "tone": (
                    draft_utterance.get("tone")
                    or (response_plan_for_prompt.get("tone", "") if isinstance(response_plan_for_prompt, dict) else "")
                ),
                "sentence_budget": draft_utterance.get("sentence_budget") or "one_or_two_short",
            }
            black_system = "\n".join(
                [
                    "너는 Black의 한국어 문장 다듬기 층이다.",
                    "주어진 초안의 뜻은 유지하고 말투와 표면 표현만 자연스럽게 고쳐라.",
                    "최종 대사 한두 문장만 출력해라. 라벨, 설명, 마크다운은 쓰지 마라.",
                    "원래 사용자 문장에 새로 답하지 말고 초안만 변환해라.",
                    "초안을 그대로 복사하지 마라.",
                    "조사, 어미, 축약, 어순 중 적어도 하나는 바꿔라.",
                    "반말만 사용해라. 요, 입니다, 습니다, 시다면 같은 존댓말은 금지다.",
                    "필수 단어가 있으면 유지해라.",
                    "새 사실, 새 조언, 새 질문, 감사, 칭찬을 추가하지 마라.",
                    "부정 표현은 절대 뒤집지 마라.",
                ]
            )
            user_lines = [
                "Black 문장 다듬기 작업",
                f"행동: {compact_response_plan['action']}",
                f"입장: {compact_response_plan['stance'] or 'none'}",
                f"톤: {compact_response_plan['tone'] or 'none'}",
                f"문장 길이: {compact_response_plan['sentence_budget']}",
                f"필수 단어: {draft_must}",
                f"피할 표현: {', '.join(avoid_terms) if avoid_terms else 'none'}",
                f"초안: {draft_reply}",
            ]
            if retry_issue:
                user_lines.extend(
                    [
                        f"재시도 이유: {retry_issue}",
                        f"이전 실패 답변: {previous_candidate}",
                        "재시도 규칙: 뜻은 유지하되 이전 실패 답변의 표현을 다시 쓰지 마라.",
                    ]
                )
                if retry_issue == "llm_draft_copy":
                    user_lines.append(
                        "재시도 규칙: 이전 답변이 초안을 복사했다. 조사, 어미, 어순을 바꿔라."
                    )
                if retry_issue == "llm_draft_negation_missing":
                    user_lines.append(
                        "재시도 규칙: 이전 답변이 부정을 뒤집었다. 같은 부정 의미를 유지해라."
                    )
                if retry_issue in {"llm_draft_anchor_missing", "llm_draft_semantic_drift"}:
                    user_lines.append(
                        "재시도 규칙: 이전 답변이 핵심 단어를 잃었다. 핵심 단어를 보이게 유지해라."
                    )
            user_lines.extend(
                [
                    "규칙:",
                    "- 같은 뜻을 유지해라",
                    "- Black이 자연스럽게 말하는 반말로 다듬어라",
                    "- 초안을 그대로 복사하지 마라",
                    "- 최종 대사만 출력해라",
                    "최종 대사:",
                ]
            )
            return [{"role": "system", "content": black_system}, {"role": "user", "content": "\n".join(user_lines)}]

        black_system = "\n".join(
            [
                system_prompt,
                "You are writing Black's final spoken Korean VTuber reply.",
                "Follow the structured plan, not your own new policy.",
                "Return only the final reply. No labels, no markdown, no explanations.",
                "Use one or two short natural Korean sentences.",
                "Use casual Korean 반말; do not use polite endings like 요, 입니다, or 시다면.",
                "Keep at least one concrete anchor; do not copy raw user instructions.",
                "Preserve negation exactly; never turn 않다/아니다/없다/말다 into a positive statement.",
                "Do not turn an accepted invite or recommendation into a question.",
                "Do not add weather, thanks, praise, facts, questions, or new advice.",
                "Never describe your response strategy with phrases like 나의 반응은, 단어가 적당, or 감정적으로.",
                "Do not prefix the reply with emotion labels; speak as Black, not as a classifier.",
                "Avoid stock tails like 그런 결, 한 번만 더, 지금 결이 너무.",
            ]
        )
        user = "\n".join(
            [
                "Structured Black decision:",
                *plan_lines,
                "",
                "Final reply only:",
            ]
        )
        return [{"role": "system", "content": black_system}, {"role": "user", "content": user}]

    @staticmethod
    def _extract_facts(user_prompt: str) -> dict[str, Any]:
        start = user_prompt.find("{")
        if start == -1:
            return {"user_text": user_prompt}
        try:
            return json.loads(user_prompt[start:])
        except json.JSONDecodeError:
            return {"user_text": user_prompt}

    @staticmethod
    def _messages_to_plain_prompt(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages) + "\n\nASSISTANT:\n"

    @staticmethod
    def _clean_chat_output(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"^(assistant|model|reply|답변|최종 답변)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip().strip("\"'`").strip()
        cleaned = re.split(r"\b(USER|SYSTEM|ASSISTANT)\s*:", cleaned, maxsplit=1)[0].strip()
        return KoBartGenerationClient._normalize_output(cleaned)

    @staticmethod
    def _generation_temperature(configured_temperature: float, *, do_sample: bool) -> float | None:
        if not do_sample:
            return None
        if configured_temperature > 0.0:
            return configured_temperature
        return 0.7

    @staticmethod
    def _validate_draft_preservation(*, reply: str, facts: dict[str, Any]) -> None:
        draft_utterance = facts.get("draft_utterance")
        if not isinstance(draft_utterance, dict):
            return
        draft_reply = str(draft_utterance.get("draft_reply") or "")
        if _has_draft_negation(draft_reply) and not _has_draft_negation(reply):
            raise RuntimeError("Causal LM missed draft negation.")
        required_terms = _draft_required_terms(draft_utterance)
        normalized_reply = _normalize_draft_text(reply)
        reply_tokens = _draft_content_tokens(reply)
        found_required = [
            term
            for term in required_terms
            if _draft_required_term_found(term=term, normalized_reply=normalized_reply, reply_tokens=reply_tokens)
        ]
        if required_terms and not found_required:
            raise RuntimeError("Causal LM missed draft anchor.")
        draft_tokens = _draft_content_tokens(" ".join([draft_reply, *required_terms]))
        if (
            len(draft_tokens) >= 4
            and len(reply_tokens) >= 3
            and not (draft_tokens & reply_tokens)
            and not found_required
        ):
            raise RuntimeError("Causal LM drifted away from draft meaning.")

    @staticmethod
    def _validate_draft_rewrite_effort(*, reply: str, facts: dict[str, Any]) -> None:
        draft_utterance = facts.get("draft_utterance")
        if not isinstance(draft_utterance, dict):
            return
        draft_reply = str(draft_utterance.get("draft_reply") or "").strip()
        if not draft_reply:
            return
        normalized_draft = _normalize_draft_text(draft_reply)
        normalized_reply = _normalize_draft_text(reply)
        if len(normalized_draft) < 8 or len(normalized_reply) < 8:
            return
        if _draft_copy_score(reply=reply, draft_reply=draft_reply) >= 0.96:
            raise RuntimeError("Causal LM copied draft without rewriting.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        return normalized


def _has_draft_negation(text: str) -> bool:
    normalized = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()
    return any(marker in normalized for marker in ("않", "아니", "없", "말진", "말지", "말고", "못"))


def _draft_required_terms(draft_utterance: dict[str, Any]) -> list[str]:
    draft_reply = str(draft_utterance.get("draft_reply") or "")
    normalized_draft = _normalize_draft_text(draft_reply)
    draft_tokens = _draft_content_tokens(draft_reply)
    raw_must_include = draft_utterance.get("must_include", [])
    must_include = raw_must_include if isinstance(raw_must_include, list) else []
    raw_terms = [
        str(draft_utterance.get("anchor") or "").strip(),
        *[
            str(item).strip()
            for item in must_include
            if str(item).strip()
        ],
    ]
    terms: list[str] = []
    for term in raw_terms:
        if not _draft_term_is_specific(term):
            continue
        if not _draft_required_term_found(
            term=term,
            normalized_reply=normalized_draft,
            reply_tokens=draft_tokens,
        ):
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _draft_term_is_specific(term: str) -> bool:
    normalized = _normalize_draft_text(term)
    if len(normalized) < 2:
        return False
    return normalized not in {
        "그말",
        "그쪽",
        "그마음",
        "그문제",
        "그선택",
        "지금",
        "오늘",
        "맞아",
        "응",
        "그래",
    }


def _draft_required_term_found(
    *,
    term: str,
    normalized_reply: str,
    reply_tokens: set[str],
) -> bool:
    normalized = _normalize_draft_text(term)
    if normalized and normalized in normalized_reply:
        return True
    term_tokens = _draft_content_tokens(term)
    if not term_tokens:
        return False
    if len(normalized) > 10:
        return len(term_tokens & reply_tokens) >= min(2, len(term_tokens))
    if term_tokens & reply_tokens:
        return True
    return any(
        left.startswith(right) or right.startswith(left)
        for left in term_tokens
        for right in reply_tokens
        if len(left) >= 3 and len(right) >= 3
    )


def _draft_content_tokens(text: str) -> set[str]:
    stopwords = {
        "오늘",
        "지금",
        "그냥",
        "조금",
        "너무",
        "정도",
        "같아",
        "있어",
        "없어",
        "그건",
        "그쪽",
        "그말",
        "무난해",
        "괜찮아",
        "보여",
        "해야",
        "하면",
        "좋아",
        "곡이",
        "곡들",
    }
    tokens = set()
    for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or "").casefold()):
        cleaned = _strip_korean_particle(token)
        if len(cleaned) >= 2 and cleaned not in stopwords:
            tokens.add(cleaned)
    return tokens


def _normalize_draft_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()


def _draft_copy_score(*, reply: str, draft_reply: str) -> float:
    normalized_reply = _normalize_draft_text(reply)
    normalized_draft = _normalize_draft_text(draft_reply)
    if not normalized_reply or not normalized_draft:
        return 0.0
    if normalized_reply == normalized_draft:
        return 1.0
    return SequenceMatcher(None, normalized_reply, normalized_draft).ratio()


def _strip_korean_particle(token: str) -> str:
    for suffix in ("으로는", "로는", "하지만", "처럼", "이면", "이야", "야", "입니다", "다", "은", "는", "이", "가", "을", "를", "도", "에", "의"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token
