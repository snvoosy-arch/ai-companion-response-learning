from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

_KOBART_IMPORT_ERROR: Exception | None = None

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional local ML stack
    torch = None
    _KOBART_IMPORT_ERROR = exc

try:
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast
except Exception as exc:  # pragma: no cover - depends on optional local ML stack
    class _UnavailableAutoConfig:
        @staticmethod
        def for_model(*args, **kwargs):
            raise RuntimeError("transformers AutoConfig is unavailable") from _KOBART_IMPORT_ERROR

        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("transformers AutoConfig is unavailable") from _KOBART_IMPORT_ERROR

    AutoModelForSeq2SeqLM = None
    AutoConfig = _UnavailableAutoConfig
    class _UnavailableAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("transformers AutoTokenizer is unavailable") from _KOBART_IMPORT_ERROR

    def _unavailable_fast_tokenizer(*args, **kwargs):
        raise RuntimeError("transformers PreTrainedTokenizerFast is unavailable") from _KOBART_IMPORT_ERROR

    AutoTokenizer = _UnavailableAutoTokenizer
    PreTrainedTokenizerFast = _unavailable_fast_tokenizer
    if _KOBART_IMPORT_ERROR is None:
        _KOBART_IMPORT_ERROR = exc


class KoBartGenerationClient:
    _BASE_TOKENIZER_NAME = "gogamza/kobart-base-v2"
    _TOPIC_LOCK_ACTIONS = {
        "continue_conversation",
        "share_feeling",
        "share_opinion",
        "small_talk",
    }
    _ALLOWED_REPEAT_TOKENS = {"진짜", "정말", "너무", "아주"}
    _TOPIC_STOPWORDS = {
        "그냥",
        "조금",
        "지금",
        "오늘",
        "이상하게",
        "아직",
        "같이",
        "사람",
        "정도",
        "느낌",
        "상태",
        "마음",
        "기분",
        "생각",
        "있다",
        "없다",
        "하다",
        "같다",
        "되다",
        "그럴",
        "이런",
        "저런",
        "그쪽",
        "여기",
        "저기",
        "뭔가",
        "기준",
        "잡아줘",
    }
    _ACTION_RULES = {
        "small_talk": "reply like a short casual greeting or opener, do not ask for missing context",
        "continue_conversation": "reply like ongoing small talk, ask one light follow-up at most",
        "share_feeling": "reply with light emotional support, do not evaluate choices or preferences",
        "share_opinion": "give a simple, direct opinion or judgment in one or two short sentences, do not comfort emotionally, do not echo the question, and do not answer as if you personally will do the user's action",
        "answer_identity": "explain what the bot is, do not ask the user's feeling back",
        "explain_capabilities": "describe what the bot can do, do not ask for taste or preference first",
        "explain_reason": "briefly explain the earlier reasoning using only the supplied reasoning trace, do not invent new reasons",
        "ask_clarification": "ask only for the missing topic or 기준, do not answer the question itself, do not give an opinion, and keep it short",
        "acknowledge": "briefly confirm understanding, do not ask another question",
        "react_laugh": "react with light laughter only, do not advise or ask for more",
        "react_surprise": "react to surprise only, do not mention capability or comfort",
        "deescalate": "lower the tone politely, do not joke or laugh",
        "ask_location": "ask for a location only, do not mention actual weather quality",
        "weather_lookup": "state checked weather briefly, do not ask for location again",
        "weather_unavailable": "briefly say the weather lookup failed and ask for one retry with a location only",
        "recommend": "make or frame a recommendation, at most ask one preference axis",
        "search_answer": "explain the meaning or answer briefly, do not ask broad follow-up",
        "news_answer": "summarize the provided news briefly and stay grounded to the supplied headlines only",
        "music_chat": "talk about music taste only",
        "game_chat": "talk about games only",
        "game_accept_or_decline": "react to the game invitation directly, without changing the invitation's meaning",
        "tell_time": "state the provided time or date information briefly and directly",
    }

    _ACTION_BLOCKED_SNIPPETS = {
        "share_opinion": (
            "몰아붙이지",
            "쉬는 쪽",
            "회복부터",
            "그럴 수 있지",
            "힘들겠다",
            "지치",
            "위로",
            "그냥 지나가는",
            "끝내자",
            "필요한 날",
        ),
        "small_talk": ("뜻 설명", "날씨처럼", "기능은 있다", "취향 축"),
        "answer_identity": ("너는 어땠", "너 쪽", "오늘 뭐 있었"),
        "explain_capabilities": ("취향 축", "취향 있", "싫어하는 것", "반응 고르는 봇이라고"),
        "ask_clarification": ("그쪽으로 정리", "확인했다", "설명은 가능", "나라면", "기울 것 같아", "그럴 수 있어", "괜찮아"),
        "explain_reason": ("새로 지어내면", "모르지만 아마", "그냥 느낌상", "그럴 수 있지", "괜찮아", "힘들겠다"),
        "deescalate": ("ㅋㅋ", "못 참지", "웃"),
        "ask_location": ("날씨도 꽤", "괜찮은 편", "맑", "비 온"),
        "weather_lookup": ("어느 지역", "도시 이름", "위치 좀", "다시 위치"),
        "weather_unavailable": ("맑", "비 온", "기온은", "바람은"),
        "recommend": ("기능은", "설명은 가능", "그럴 수 있어", "위로", "너는 어때"),
        "music_chat": ("기능은", "설명은 가능", "그럴 수 있어", "위로"),
        "search_answer": ("그럴 수 있어", "위로", "괜찮아", "힘들겠다"),
        "news_answer": ("그럴 수 있어", "위로", "괜찮아", "힘들겠다"),
        "tell_time": ("그럴 수 있어", "위로", "괜찮아"),
        "react_surprise": ("반응은 꽤 잘 한다", "설명은 가능", "말해봐", "그럴 수 있지", "괜찮아", "힘들겠다", "위로"),
        "react_laugh": ("그럴 수 있지", "괜찮아", "힘들겠다", "위로", "설명은 가능", "말해봐"),
    }

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "auto",
        max_new_tokens: int = 24,
        num_beams: int = 1,
        input_mode: str = "full",
        output_guard_enabled: bool = True,
    ) -> None:
        if torch is None or AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "KoBART generation dependencies are unavailable in this Python environment."
            ) from _KOBART_IMPORT_ERROR

        self.model_name_or_path = model_name_or_path
        self.device_name = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.input_mode = (input_mode or "full").lower()
        self.output_guard_enabled = bool(output_guard_enabled)
        self.last_generation_issue: str | None = None
        self.last_raw_generation: str = ""

        self._tokenizer = self._load_tokenizer(model_name_or_path)
        generation_config = self._load_generation_config(model_name_or_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=generation_config)
        self._model.to(self.device_name)
        self._model.eval()

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _generate_sync(self, *, system_prompt: str, user_prompt: str) -> str:
        facts = self._extract_facts(user_prompt)
        facts.setdefault("input_mode", self.input_mode)
        action = str(facts.get("action", "")).strip()
        self.last_generation_issue = None
        self.last_raw_generation = ""

        prompt = self._build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            facts=facts,
            strict_retry=False,
        )
        generated = self._generate_from_prompt(prompt, sample=False)
        self.last_raw_generation = generated
        if not self.output_guard_enabled:
            return self._normalize_output(generated)
        try:
            return self._finalize_generated(generated, action=action, facts=facts)
        except RuntimeError as exc:
            first_issue = self._issue_code_from_exception(exc)

        retry_facts = dict(facts)
        retry_facts["generation_retry_issue"] = first_issue
        retry_facts["previous_candidate"] = generated
        retry_prompt = self._build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            facts=retry_facts,
            strict_retry=True,
        )
        last_retry_issue = first_issue
        retry_generated = ""
        for retry_index in range(3):
            variant_prompt = retry_prompt.replace(
                "\nreply:",
                f"\nretry_variant: {retry_index + 1}\nreply:",
            )
            retry_generated = self._generate_from_prompt(variant_prompt, sample=True)
            self.last_raw_generation = retry_generated
            try:
                reply = self._finalize_generated(retry_generated, action=action, facts=facts)
                self.last_generation_issue = f"{first_issue}:recovered"
                return reply
            except RuntimeError as exc:
                last_retry_issue = self._issue_code_from_exception(exc)
                retry_facts["previous_candidate"] = retry_generated or retry_facts["previous_candidate"]

        self.last_generation_issue = f"{first_issue};retry_{last_retry_issue}"
        return self._normalize_output(retry_generated or generated)

    def _generate_from_prompt(self, prompt: str, *, sample: bool) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(self.device_name) for key, value in inputs.items()}

        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "num_beams": 1 if sample else self.num_beams,
                "do_sample": sample,
                "early_stopping": not sample,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
            }
            if sample:
                generation_kwargs["top_p"] = 0.92
                generation_kwargs["temperature"] = 0.85
            output_ids = self._model.generate(**generation_kwargs)

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    @staticmethod
    def _issue_code_from_exception(exc: RuntimeError) -> str:
        lowered = str(exc).lower()
        if "empty response" in lowered:
            return "llm_empty_reply"
        if "echoed the prompt" in lowered:
            return "llm_prompt_echo"
        if "unusable reply" in lowered:
            return "llm_unusable_reply"
        if "drifted away from the user's topic" in lowered:
            return "llm_topic_drift"
        if "draft negation" in lowered:
            return "llm_draft_negation_missing"
        if "draft anchor" in lowered:
            return "llm_draft_anchor_missing"
        if "draft meaning" in lowered:
            return "llm_draft_semantic_drift"
        if "copied draft" in lowered:
            return "llm_draft_copy"
        return "llm_generation_issue"

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return normalized

    @staticmethod
    def _load_generation_config(model_name_or_path: str):
        model_dir = Path(model_name_or_path)
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                model_type = str(config_payload.pop("model_type", "")).strip()
                config_payload.pop("id2label", None)
                config_payload.pop("label2id", None)
                if model_type:
                    return AutoConfig.for_model(model_type, **config_payload)
            except Exception:
                pass
        return AutoConfig.from_pretrained(model_name_or_path)

    @staticmethod
    def _load_tokenizer(model_name_or_path: str):
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as exc:
            if isinstance(exc, ValueError) and "Tokenizer class TokenizersBackend" not in str(exc):
                raise

            model_dir = Path(model_name_or_path)
            tokenizer_file = model_dir / "tokenizer.json"
            if not tokenizer_file.exists():
                return AutoTokenizer.from_pretrained(KoBartGenerationClient._BASE_TOKENIZER_NAME)

            tokenizer_config_path = model_dir / "tokenizer_config.json"
            tokenizer_config: dict[str, object] = {}
            if tokenizer_config_path.exists():
                tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

            special_tokens_map_path = model_dir / "special_tokens_map.json"
            special_tokens_map: dict[str, object] = {}
            if special_tokens_map_path.exists():
                special_tokens_map = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))

            init_kwargs: dict[str, object] = {"tokenizer_file": str(tokenizer_file)}
            for key in (
                "bos_token",
                "eos_token",
                "pad_token",
                "unk_token",
                "mask_token",
                "cls_token",
                "sep_token",
            ):
                value = special_tokens_map.get(key, tokenizer_config.get(key))
                if value:
                    init_kwargs[key] = value

            model_max_length = tokenizer_config.get("model_max_length")
            if isinstance(model_max_length, int):
                init_kwargs["model_max_length"] = model_max_length

            try:
                tokenizer = PreTrainedTokenizerFast(**init_kwargs)
                padding_side = str(tokenizer_config.get("padding_side") or "").strip()
                truncation_side = str(tokenizer_config.get("truncation_side") or "").strip()
                if padding_side:
                    tokenizer.padding_side = padding_side
                if truncation_side:
                    tokenizer.truncation_side = truncation_side
                return tokenizer
            except Exception:
                return AutoTokenizer.from_pretrained(KoBartGenerationClient._BASE_TOKENIZER_NAME)

    @staticmethod
    def _build_prompt(
        *,
        system_prompt: str,
        user_prompt: str,
        facts: dict | None = None,
        strict_retry: bool = False,
    ) -> str:
        facts = facts or KoBartGenerationClient._extract_facts(user_prompt)
        draft_rewrite_prompt = KoBartGenerationClient._build_draft_rewrite_prompt(
            facts=facts,
            strict_retry=strict_retry,
        )
        if draft_rewrite_prompt:
            return draft_rewrite_prompt
        user_text = str(facts.get("user_text", "")).strip()
        action = str(facts.get("action", "")).strip()
        input_mode = str(facts.get("input_mode") or "full").strip().lower()
        action_rule = KoBartGenerationClient._ACTION_RULES.get(
            action,
            "reply naturally and follow the action exactly",
        )
        reason_code = str(facts.get("reason_code", "")).strip()[:120]
        raw_reason_flags = facts.get("reason_flags") or []
        reason_flags = []
        if isinstance(raw_reason_flags, list):
            reason_flags = [str(item).strip() for item in raw_reason_flags if str(item).strip()]
        reason_summary = str(facts.get("reason_summary", "")).strip()[:120]
        persona = KoBartGenerationClient._style_hint(system_prompt)

        intent = ""
        context = ""
        constraints: list[str] = []
        decomposition: dict | None = None
        grounding_bundle: dict | None = None
        action_payload: dict | None = None
        response_plan: dict | None = None
        draft_utterance: dict | None = None
        phrasing_plan = facts.get("phrasing_plan")
        world_state = facts.get("world_state")
        raw_decomposition = facts.get("current_turn_decomposition")
        if isinstance(raw_decomposition, dict):
            decomposition = raw_decomposition
        raw_grounding_bundle = facts.get("grounding_bundle")
        if isinstance(raw_grounding_bundle, dict):
            grounding_bundle = raw_grounding_bundle
        raw_action_payload = facts.get("action_payload")
        if isinstance(raw_action_payload, dict):
            action_payload = raw_action_payload
        raw_response_plan = facts.get("response_plan")
        if isinstance(raw_response_plan, dict):
            response_plan = raw_response_plan
        raw_draft_utterance = facts.get("draft_utterance")
        if isinstance(raw_draft_utterance, dict):
            draft_utterance = raw_draft_utterance
        if isinstance(world_state, dict):
            intent = str(world_state.get("dominant_intent") or "").strip()
            context = str(world_state.get("memory_summary") or "").strip()[:160]
            raw_constraints = world_state.get("constraints") or []
            if isinstance(raw_constraints, list):
                constraints = [str(item).strip() for item in raw_constraints if str(item).strip()]
        if input_mode == "slim":
            intent = str(facts.get("intent") or intent).strip()
            context = ""
            raw_constraints = facts.get("constraints") or []
            if isinstance(raw_constraints, list):
                constraints = [str(item).strip() for item in raw_constraints if str(item).strip()]

        weather_line = ""
        weather = facts.get("weather")
        if isinstance(weather, dict):
            weather_line = (
                "\nweather:"
                f" location={weather.get('location')},"
                f" description={weather.get('description')},"
                f" temperature_c={weather.get('temperature_c')},"
                f" wind_kph={weather.get('wind_kph')}"
            )

        constraint_lines: list[str] = []
        if "avoid_overfamiliarity" in constraints:
            constraint_lines.append("- avoid slang, overfriendly teasing, and casual fillers like ㅋㅋ / ㅎㅇ / 반가워~")
        if "respect_boundary_history" in constraints:
            constraint_lines.append("- do not push the user to continue or ask for one more line")
        if "do_not_guess_facts" in constraints:
            constraint_lines.append("- do not guess missing facts or pretend you already know them")
        if "collect_location_before_answer" in constraints:
            constraint_lines.append("- collect the location before giving any weather judgment")
        if "no_question_mark" in constraints:
            constraint_lines.append("- do not end the reply with a question mark")
        if "avoid_self_insertion" in constraints:
            constraint_lines.append("- do not turn the reply into your own mood or state with phrases like '나는 ...'")
        if "direct_opinion_only" in constraints:
            constraint_lines.append("- answer the opinion directly without hedging or bouncing it back")
        if "self_style_anchor" in constraints:
            constraint_lines.append("- if asked about your own style, answer with one concrete opener you would actually say")
        if "keep_topic_anchor" in constraints:
            constraint_lines.append("- keep the user's main topic noun visible in the reply instead of replacing it with vague words")
        if "concrete_preference_disclosure" in constraints:
            constraint_lines.append("- answer like revealing your own preference directly, not like comforting or coaching")
        if "habit_anchor" in constraints:
            constraint_lines.append("- answer in terms of your usual tendency or habit, not as advice to the user")
        if "short_conditional_judgment" in constraints:
            constraint_lines.append("- give one short judgment with at most one simple condition")
        if "start_with_first_step" in constraints:
            constraint_lines.append("- start with the first thing to check or do, not with abstract commentary")
        if "conditional_advice" in constraints:
            constraint_lines.append("- give conditional go-or-no-go style advice instead of fact lookup or emotional comfort")
        if "activity_recommendation" in constraints:
            constraint_lines.append("- answer as an activity recommendation: name concrete things to do, not a feeling reaction")
        if "concrete_activity_options" in constraints:
            constraint_lines.append("- include two or three concrete activity options from the response plan")
        if "avoid_emotional_comfort" in constraints:
            constraint_lines.append("- do not comfort emotionally with phrases like 괜찮아 / 그럴 수 있어 / 힘들겠다")
        if "avoid_repetition" in constraints:
            constraint_lines.append("- avoid repeating the same subject or time word across both sentences")
        if "avoid_weather_restatement" in constraints:
            constraint_lines.append("- do not waste the reply by merely restating that it is a weather day")
        if "location_only" in constraints:
            constraint_lines.append("- ask only for the city or region needed to ground the weather answer")
        if "no_weather_claim" in constraints:
            constraint_lines.append("- do not claim what the weather is like unless weather facts are explicitly provided")
        if "weather_facts_only" in constraints:
            constraint_lines.append("- use only the provided weather facts and keep the reply grounded")
        if "no_location_reask" in constraints:
            constraint_lines.append("- do not ask for the location again")
        if "retry_with_location_only" in constraints:
            constraint_lines.append("- briefly ask for one retry with a location, without giving a weather judgment")
        if strict_retry:
            constraint_lines.extend(
                [
                    "- do not echo the user's wording or prompt fields",
                    "- if unsure, reply shorter and plainer instead of becoming awkward",
                    "- avoid broken phrasing, repeated fragments, and decorative intimacy",
                ]
            )

        phrasing_lines: list[str] = []
        phrasing_line = ""
        decomposition_line = ""
        grounding_line = ""
        action_payload_line = ""
        response_plan_line = ""
        draft_utterance_line = ""
        prompt_user_text = user_text
        user_text_policy_line = ""
        topic_line = ""
        reply_style_line = ""
        reply_focus_line = ""
        plan_reason_line = ""
        answer_blueprint_line = ""
        if input_mode != "slim" and isinstance(phrasing_plan, dict):
            opener = str(phrasing_plan.get("opener") or "").strip()
            question_mode = str(phrasing_plan.get("question_mode") or "").strip()
            closer = str(phrasing_plan.get("closer") or "").strip()
            distance = str(phrasing_plan.get("distance") or "").strip()
            asks_followup = bool(phrasing_plan.get("asks_followup"))
            notes = phrasing_plan.get("notes") or []
            if isinstance(notes, list):
                note_text = ", ".join(str(item).strip() for item in notes if str(item).strip())
            else:
                note_text = ""
            note_text = note_text[:120]

            phrasing_line = (
                "\nphrasing_plan:"
                f" opener={opener},"
                f" question_mode={question_mode},"
                f" closer={closer},"
                f" distance={distance},"
                f" asks_followup={str(asks_followup).lower()},"
                f" notes={note_text or 'none'}"
            )

            if opener == "clarifying":
                phrasing_lines.append("- open by briefly marking that you are clarifying, not comforting")
            elif opener == "reactive":
                phrasing_lines.append("- open with a short reaction, not an explanation")
            elif opener == "informative":
                phrasing_lines.append("- open directly with the answer, not with empathy talk")
            elif opener == "brief":
                phrasing_lines.append("- open briefly and do not add a long second sentence")
            elif opener == "grounded":
                phrasing_lines.append("- sound grounded and concrete, not poetic or analytical")
            elif opener in {"bridging", "light", "warm"}:
                phrasing_lines.append("- keep the opening conversational and easy to continue")

            if question_mode == "none" or not asks_followup:
                phrasing_lines.append("- do not add a follow-up question")
            elif question_mode == "soft":
                phrasing_lines.append("- if you ask back, use only one soft follow-up question")
            elif question_mode == "direct":
                phrasing_lines.append("- if you ask back, use only one direct follow-up question")

            if closer == "soft_close":
                phrasing_lines.append("- end in a way that lets the user stop comfortably")
            elif closer == "keep_open":
                phrasing_lines.append("- leave the reply lightly open for the next turn")

            if distance == "playful":
                phrasing_lines.append("- keep it playful but still natural Korean")
            elif distance == "soft":
                phrasing_lines.append("- keep some distance and avoid sounding overly close")
            elif distance == "steady":
                phrasing_lines.append("- keep the tone steady and plain")
        elif input_mode == "slim":
            if "no_followup" in constraints:
                phrasing_lines.append("- do not add a follow-up question")

        if "schema_preference_disclosure" in reason_flags:
            phrasing_lines.append("- treat this as a preference disclosure question and answer with your own simple like/dislike stance")
        if "schema_habit_preference" in reason_flags:
            phrasing_lines.append("- treat this as a habit question and answer with your usual tendency")
        if "schema_self_style" in reason_flags:
            phrasing_lines.append("- treat this as a self-style question and answer with one concrete opener you would actually use")
        if "schema_reflective_judgment" in reason_flags:
            phrasing_lines.append("- treat this as a reflective judgment question and pick a side briefly")
        if "schema_process_advice" in reason_flags:
            phrasing_lines.append("- treat this as a process question and name the first thing to check or do")
        if "schema_soft_decision" in reason_flags:
            phrasing_lines.append("- treat this as a soft decision question and give a conditional go-or-no-go opinion")
        if "schema_activity_recommendation" in reason_flags:
            phrasing_lines.append("- treat this as a place-based activity recommendation and list concrete play/activity ideas")

        if input_mode != "slim" and isinstance(decomposition, dict):
            clauses = decomposition.get("clauses") or []
            propositions = decomposition.get("propositions") or []
            cue_names = decomposition.get("context_cues") or []
            context_dependency_level = str(decomposition.get("context_dependency_level") or "").strip()

            if isinstance(clauses, list):
                clause_text = " | ".join(str(item).strip() for item in clauses[:3] if str(item).strip())
            else:
                clause_text = ""
            proposition_parts: list[str] = []
            if isinstance(propositions, list):
                for item in propositions[:4]:
                    if not isinstance(item, dict):
                        continue
                    kind = str(item.get("kind") or "").strip()
                    obj = str(item.get("object") or "").strip()
                    value = str(item.get("value") or "").strip()
                    if kind:
                        proposition_parts.append(f"{kind}:{obj or value}")
            decomposition_line = (
                "\ndecomposition:"
                f" clauses={clause_text or 'none'},"
                f" propositions={', '.join(part for part in proposition_parts if part) or 'none'},"
                f" context_dependency={context_dependency_level or 'low'},"
                f" context_cues={', '.join(str(item).strip() for item in cue_names[:4] if str(item).strip()) or 'none'}"
            )
            if context_dependency_level in {"medium", "high"}:
                phrasing_lines.append("- respect the recent handoff and prior flow; do not answer as if this were isolated")
            if isinstance(cue_names, list) and "quiet_mode" in cue_names:
                phrasing_lines.append("- keep the reply low-pressure and do not force a new energetic turn")
            if isinstance(cue_names, list) and "aftereffect_hold" in cue_names:
                phrasing_lines.append("- stay with the lingering feeling instead of switching to advice or explanation")

        if input_mode != "slim" and isinstance(grounding_bundle, dict):
            must_include_topics = grounding_bundle.get("must_include_topics") or []
            allowed_evidence = grounding_bundle.get("allowed_evidence") or []
            forbidden_patterns = grounding_bundle.get("forbidden_patterns") or []
            tone_contract = str(grounding_bundle.get("tone_contract") or "").strip()
            followup_policy = str(grounding_bundle.get("followup_policy") or "").strip()
            grounding_line = (
                "\ngrounding:"
                f" topics={', '.join(str(item).strip() for item in must_include_topics[:4] if str(item).strip()) or 'none'},"
                f" evidence={'; '.join(str(item).strip() for item in allowed_evidence[:4] if str(item).strip()) or 'none'},"
                f" tone_contract={tone_contract or 'none'},"
                f" followup_policy={followup_policy or 'none'}"
            )
            if isinstance(must_include_topics, list) and must_include_topics:
                phrasing_lines.append("- keep at least one grounding topic from the grounding bundle in the reply")
            if isinstance(forbidden_patterns, list):
                if "pushy_followup" in forbidden_patterns:
                    phrasing_lines.append("- do not push for another turn or add a steering follow-up")
                if "meta_explanation" in forbidden_patterns:
                    phrasing_lines.append("- do not explain the action or describe the response strategy itself")
                if "prompt_echo" in forbidden_patterns:
                    phrasing_lines.append("- never echo prompt field names or structured labels")
            if followup_policy == "no_extra_followup":
                phrasing_lines.append("- do not add any extra follow-up beyond the grounding bundle")

        if input_mode != "slim" and isinstance(action_payload, dict):
            payload_parts = [
                f"{key}={str(value).strip() or 'none'}"
                for key, value in action_payload.items()
                if value is not None and str(value).strip()
            ]
            action_payload_line = (
                "\naction_payload:"
                f" {', '.join(payload_parts) if payload_parts else 'none'}"
            )
            if action == "ask_location":
                phrasing_lines.append("- ask only for the location needed for the weather check")
                phrasing_lines.append("- do not judge whether the activity sounds good yet")
            elif action == "weather_lookup":
                phrasing_lines.append("- answer with the checked weather facts first")
            elif action == "weather_unavailable":
                phrasing_lines.append("- keep it to a short failure notice plus one retry request")

        if isinstance(response_plan, dict):
            stance = str(response_plan.get("stance") or "").strip()
            anchor = str(response_plan.get("anchor") or "").strip()[:100]
            followup_policy = str(response_plan.get("followup_policy") or "").strip()
            sentence_budget = str(response_plan.get("sentence_budget") or "").strip()
            tone = str(response_plan.get("tone") or "").strip()
            raw_must_include = response_plan.get("must_include") or []
            raw_avoid = response_plan.get("avoid") or []
            raw_notes = response_plan.get("notes") or []
            must_include = []
            avoid = []
            notes = []
            if isinstance(raw_must_include, list):
                must_include = [str(item).strip() for item in raw_must_include if str(item).strip()]
            if isinstance(raw_avoid, list):
                avoid = [str(item).strip() for item in raw_avoid if str(item).strip()]
            if isinstance(raw_notes, list):
                notes = [str(item).strip() for item in raw_notes if str(item).strip()]

            response_plan_line = (
                "\nresponse_plan:"
                f" stance={stance or 'neutral'},"
                f" anchor={anchor or 'none'},"
                f" must_include={', '.join(must_include[:5]) if must_include else 'none'},"
                f" avoid={', '.join(avoid[:6]) if avoid else 'none'},"
                f" followup_policy={followup_policy or 'auto'},"
                f" sentence_budget={sentence_budget or 'one_or_two_short'},"
                f" tone={tone or 'steady'},"
                f" notes={', '.join(notes[:4]) if notes else 'none'}"
            )
            if stance:
                phrasing_lines.append(f"- response plan stance: {stance}")
            if anchor:
                topic_line = f"topic: {anchor}\n"
                phrasing_lines.append(f"- answer the response_plan anchor directly: {anchor}")
            if action == "share_opinion":
                reply_style_line = "reply_style: direct_judgment\n"
                phrasing_lines.append("- give a judgment about the user's choice, not a promise that you will do the action")
            if stance == "practical_activity_recommendation" and anchor:
                if not context:
                    context = "activity_recommendation"
                reply_style_line = "reply_style: practical_recommendation\n"
                reply_focus_line = "reply_focus: concrete_activity_options\n"
                option_hint = ", ".join(must_include[1:5] if len(must_include) > 1 else must_include[:4])
                plan_reason_line = (
                    f"reason: {anchor}에서 할 만한 활동을 두세 가지 구체적으로 추천한다.\n"
                )
                answer_blueprint_line = (
                    f"answer_blueprint: {anchor} 추천 후보 = {option_hint or ', '.join(must_include[:4])}; "
                    "후보 중 두세 개를 자연스럽게 말한다; 위로나 일반 반응으로 답하지 않는다.\n"
                )
                phrasing_lines.append(
                    f"- recommend activities for '{anchor}' using concrete options"
                    + (f": {option_hint}" if option_hint else "")
                )
                phrasing_lines.append("- do not answer with stopping, passing, ending the day, or vague comfort")
            if stance == "conditional_go_or_no_go" and anchor:
                if not context:
                    context = "soft_decision"
                reply_focus_line = "reply_focus: conditional_decision\n"
                plan_reason_line = f"reason: {anchor}에 대해 조건부로 해볼 만한지 짧게 판단한다.\n"
                phrasing_lines.append(
                    f"- decide whether '{anchor}' is worth trying under a simple condition; do not mention arriving, going, or being late"
                )
            if action in {"continue_conversation", "small_talk"} and "social_return_acknowledgement" in notes:
                context = "social_return"
                reply_focus_line = "reply_focus: social_return_acknowledgement\n"
                answer_blueprint_line = (
                    "answer_blueprint: 상대가 다시 왔다는 점을 먼저 받는다; "
                    "부담 주지 않고 짧게 이어준다; 그런 건/그런 결 같은 대체어로 뭉개지 않는다.\n"
                )
                phrasing_lines.append("- acknowledge that the user came back after a while")
                phrasing_lines.append("- include a concrete cue like 다시 왔 or 오랜만 instead of vague comfort")
            if (
                action in {"share_opinion", "share_feeling", "continue_conversation"}
                and anchor
                and anchor != user_text
                and followup_policy == "no_followup"
            ):
                prompt_user_text = anchor
                user_text_policy_line = "user_text_policy: original_user_text_withheld_to_reduce_echo\n"
            if must_include:
                phrasing_lines.append(
                    "- include at least one response_plan must_include item: "
                    + ", ".join(must_include[:4])
                )
            if avoid:
                phrasing_lines.append(
                    "- avoid these exact generic phrases: "
                    + ", ".join(avoid[:8])
                )
            if followup_policy == "no_followup":
                phrasing_lines.append("- do not add a follow-up question")
            elif followup_policy == "one_required_question":
                phrasing_lines.append("- ask exactly the one required missing-slot question")
            elif followup_policy == "one_soft_followup":
                phrasing_lines.append("- use at most one soft follow-up question")
            elif followup_policy == "one_direct_followup":
                phrasing_lines.append("- use at most one direct follow-up question")
            if sentence_budget == "one_short":
                phrasing_lines.append("- keep the reply to one short sentence")
            elif sentence_budget == "one_or_two_short_no_question":
                phrasing_lines.append("- use one or two short sentences and no question")
            elif sentence_budget == "short_reaction_fragment_ok":
                phrasing_lines.append("- a short reaction fragment is acceptable")
            if "answer_anchor_before_generic_reaction" in notes:
                phrasing_lines.append("- do not replace the anchor with vague comfort")
            if "use_action_payload_as_source" in notes:
                phrasing_lines.append("- prefer action_payload facts over guesses")

        if isinstance(draft_utterance, dict):
            draft_reply = str(draft_utterance.get("draft_reply") or "").strip()[:180]
            draft_anchor = str(draft_utterance.get("anchor") or "").strip()[:80]
            raw_draft_must = draft_utterance.get("must_include") or []
            raw_draft_avoid = draft_utterance.get("avoid") or []
            draft_must = []
            draft_avoid = []
            if isinstance(raw_draft_must, list):
                draft_must = [str(item).strip() for item in raw_draft_must if str(item).strip()]
            if isinstance(raw_draft_avoid, list):
                draft_avoid = [str(item).strip() for item in raw_draft_avoid if str(item).strip()]
            draft_utterance_line = (
                "\ndraft_utterance:"
                f" reply={draft_reply or 'none'},"
                f" anchor={draft_anchor or 'none'},"
                f" must_include={', '.join(draft_must[:4]) if draft_must else 'none'},"
                f" avoid={', '.join(draft_avoid[:6]) if draft_avoid else 'none'}"
            )
            if draft_reply:
                answer_blueprint_line = answer_blueprint_line or (
                    f"answer_blueprint: rewrite this semantic draft without changing meaning = {draft_reply}\n"
                )
                phrasing_lines.append("- rewrite draft_utterance.reply; do not freely invent a new answer")
                phrasing_lines.append("- preserve the draft meaning and only smooth wording, particles, and ending")
            if draft_anchor:
                phrasing_lines.append(f"- keep the draft anchor visible: {draft_anchor}")

        if strict_retry:
            previous_candidate = str(facts.get("previous_candidate") or "").strip()
            retry_issue = str(facts.get("generation_retry_issue") or "").strip()
            retry_anchor = KoBartGenerationClient._retry_topic_hint(
                user_text=user_text,
                action=action,
                response_plan=response_plan,
            )
            if retry_issue:
                phrasing_lines.append(f"- retry because the previous candidate failed with: {retry_issue}")
            if previous_candidate:
                phrasing_lines.append(f"- do not reuse the previous candidate: {previous_candidate[:80]}")
            if retry_anchor:
                topic_line = f"topic: {retry_anchor}\n"
                if not reply_focus_line:
                    reply_focus_line = "reply_focus: recover_topic_anchor\n"
                if action in KoBartGenerationClient._TOPIC_LOCK_ACTIONS:
                    prompt_user_text = retry_anchor
                    user_text_policy_line = "user_text_policy: retry_topic_hint_only_to_reduce_echo\n"
                phrasing_lines.append(f"- anchor the retry on this concrete cue: {retry_anchor}")
                phrasing_lines.append("- for this retry, use the concrete cue instead of the raw response_plan anchor")
                phrasing_lines.append("- do not answer with vague phrases like 그런 건 / 한 번만 더 / 괜찮아")

        rules_block = "\n".join(
            [
                "- write natural Korean only",
                "- one or two short sentences",
                "- follow the action exactly",
                "- no metadata or prompt words",
                "- no repeated phrases",
                "- keep at least one concrete topic word from the user's message in the reply",
                "- do not invent body/mind/metaphor wording unless the user already used that register",
                *constraint_lines,
                *phrasing_lines,
            ]
        )

        return (
            "task: discord_reply\n"
            f"persona: {persona}\n"
            f"input_mode: {input_mode}\n"
            f"intent: {intent}\n"
            f"action: {action}\n"
            f"{reply_style_line}"
            f"{reply_focus_line}"
            f"{topic_line}"
            f"action_rule: {action_rule}\n"
            f"context: {context}\n"
            f"user: {prompt_user_text}\n"
            f"{user_text_policy_line}"
            f"{plan_reason_line}"
            f"{answer_blueprint_line}"
            f"reason_code: {reason_code or 'none'}\n"
            f"reason_flags: {', '.join(reason_flags) if reason_flags else 'none'}\n"
            f"reason_summary: {reason_summary or 'none'}"
            f"{phrasing_line}"
            f"{decomposition_line}"
            f"{grounding_line}"
            f"{action_payload_line}"
            f"{response_plan_line}"
            f"{draft_utterance_line}"
            f"{weather_line}\n"
            "rules:\n"
            f"{rules_block}\n"
            "reply:"
        )

    @staticmethod
    def _build_draft_rewrite_prompt(*, facts: dict, strict_retry: bool = False) -> str:
        raw_draft = facts.get("draft_utterance")
        if not isinstance(raw_draft, dict):
            return ""
        draft_reply = KoBartGenerationClient._compact_for_prompt(raw_draft.get("draft_reply"), limit=220)
        if not draft_reply:
            return ""

        required_terms = KoBartGenerationClient._draft_required_terms_for_prompt(raw_draft, draft_reply=draft_reply)
        previous_candidate = KoBartGenerationClient._compact_for_prompt(
            facts.get("previous_candidate"),
            limit=160,
        )
        retry_issue = KoBartGenerationClient._compact_for_prompt(
            facts.get("generation_retry_issue"),
            limit=80,
        )
        lines = [
            "작업: 문장 다듬기",
            "역할: Black의 최종 대사를 자연스러운 한국어 반말로만 정리한다.",
            "규칙:",
            "- 초안의 의미를 바꾸지 마라.",
            "- 새 정보, 새 조언, 새 질문을 추가하지 마라.",
            "- 메타 설명이나 항목 이름을 출력하지 마라.",
            "- 한두 문장으로만 답해라.",
        ]
        if required_terms:
            lines.append("- 반드시 남길 표현: " + ", ".join(required_terms[:4]))
        if strict_retry:
            lines.extend(
                [
                    "- 이전 출력이 실패했으니 초안을 거의 그대로 두고 어미와 조사만 다듬어라.",
                    f"실패 이유: {retry_issue or 'unknown'}",
                ]
            )
            if previous_candidate:
                lines.append(f"이전 실패 출력: {previous_candidate}")
        lines.extend(
            [
                f"초안: {draft_reply}",
                "출력:",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _compact_for_prompt(value: object, *, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:limit].strip()

    @staticmethod
    def _draft_required_terms_for_prompt(draft_utterance: dict, *, draft_reply: str) -> list[str]:
        raw_must_include = draft_utterance.get("must_include") or []
        must_include = raw_must_include if isinstance(raw_must_include, list) else []
        raw_terms = [
            str(draft_utterance.get("anchor") or "").strip(),
            *[
                str(item).strip()
                for item in must_include
                if str(item).strip()
            ],
        ]
        draft_norm = KoBartGenerationClient._normalize_for_prompt_term(draft_reply)
        terms: list[str] = []
        for term in raw_terms:
            compact = KoBartGenerationClient._compact_for_prompt(term, limit=60)
            normalized = KoBartGenerationClient._normalize_for_prompt_term(compact)
            if len(normalized) < 2:
                continue
            if normalized not in draft_norm:
                continue
            if compact not in terms:
                terms.append(compact)
        return terms

    @staticmethod
    def _normalize_for_prompt_term(value: object) -> str:
        return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "")).lower()

    @staticmethod
    def _extract_facts(user_prompt: str) -> dict:
        start = user_prompt.find("{")
        if start == -1:
            return {"user_text": user_prompt}
        try:
            return json.loads(user_prompt[start:])
        except json.JSONDecodeError:
            return {"user_text": user_prompt}

    @classmethod
    def _extract_action(cls, user_prompt: str) -> str:
        facts = cls._extract_facts(user_prompt)
        return str(facts.get("action", "")).strip()

    @staticmethod
    def _style_hint(system_prompt: str) -> str:
        if "'Black'" in system_prompt or "Black" in system_prompt:
            return "black_casual"
        if "'White'" in system_prompt or "White" in system_prompt:
            return "white_calm"
        return "default_casual"

    @staticmethod
    def _retry_topic_hint(*, user_text: str, action: str, response_plan: dict | None) -> str:
        text = re.sub(r"\s+", " ", str(user_text or "")).strip()
        if not text:
            return ""
        if any(marker in text for marker in ("한동안", "오랜만", "다시 와", "다시 왔", "말 안 하다가")):
            return "한동안 말이 없다가 다시 온 흐름"
        if any(marker in text for marker in ("말수가", "말수", "말 안", "조용히")):
            return "말을 줄이고 조용히 있으려는 흐름"
        if any(marker in text for marker in ("같이 있어", "붙어 있는", "설명보다")):
            return "설명보다 곁에 있어주길 바라는 흐름"
        if isinstance(response_plan, dict):
            anchor = str(response_plan.get("anchor") or "").strip()
            if anchor and len(anchor) <= 40:
                return anchor
            must_include = response_plan.get("must_include") or []
            if isinstance(must_include, list):
                for item in must_include:
                    candidate = str(item or "").strip()
                    if candidate and len(candidate) <= 40:
                        return candidate
        if action in KoBartGenerationClient._TOPIC_LOCK_ACTIONS:
            anchors = KoBartGenerationClient._extract_topic_anchors(text)
            if anchors:
                return " ".join(anchors[:3])
        return ""

    @staticmethod
    def _looks_like_prompt_echo(text: str, *, user_text: str = "") -> bool:
        markers = (
            "task:",
            "persona:",
            "intent:",
            "action:",
            "context:",
            "user:",
            "reason:",
            "rules:",
            "reply:",
        )
        if sum(marker in text for marker in markers) >= 2:
            return True

        if user_text:
            normalized_text = re.sub(r"[^\w가-힣]+", "", text).lower()
            normalized_user = re.sub(r"[^\w가-힣]+", "", user_text).lower()
            if normalized_user and normalized_text:
                if normalized_text == normalized_user:
                    return True
                if len(normalized_user) >= 8 and normalized_user in normalized_text:
                    return True

        return False

    @staticmethod
    def _normalize_output(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        compact = re.sub(r"\s+([,.!?])", r"\1", compact)
        compact = re.sub(r"^(assistant|reply|답변)\s*:\s*", "", compact, flags=re.IGNORECASE)
        compact = compact.strip("\"'`[]() ")

        parts = re.findall(r"[^.!?]+[.!?]?", compact)
        cleaned_parts: list[str] = []
        for part in parts:
            piece = part.strip()
            if not piece:
                continue
            piece = re.sub(r"^(assistant|reply|답변)\s*:\s*", "", piece, flags=re.IGNORECASE).strip("\"'`[]() ")
            if cleaned_parts and piece == cleaned_parts[-1]:
                continue
            cleaned_parts.append(piece)
            if len(cleaned_parts) >= 2:
                break

        if cleaned_parts:
            compact = " ".join(cleaned_parts).strip()

        if len(compact) > 80:
            clipped = compact[:80].rsplit(" ", 1)[0].strip()
            if clipped:
                compact = clipped

        return compact.strip()

    @staticmethod
    def _looks_unusable(text: str, *, action: str, constraints: tuple[str, ...] = ()) -> bool:
        if KoBartGenerationClient._has_malformed_surface_text(text):
            return True

        hard_blocked_snippets = (
            "문장 다듬기 작업",
            "Noir",
            "그럭",
            "그 톤면",
            "그런 건은",
            "그런 결은 있지",
            "좋아졌다고졌다고",
            "괜찮져진",
            "지금 결이 너무",
            "여운이 남는 쪽이 결국 오래",
            "고른 쪽이 맞아",
            "한 번 받아두자",
            "반응은 바로 가능",
            "감정 표현",
            "감정적으로",
            "나의 반응은",
            "단어가 적당",
            "라는 점을 확인",
            "가지고 있는 자리에",
            "그 말은 여기서 낮게 받아들일게",
            "이제 낮게 받아들일게",
            "한 번만 더 짚어봐",
            "한 번만 더 붙이면",
            "그 감정은 그 순간부터 반응하는 쪽이 더 맞아 보여",
            "쪽으로 딱 맞는 헤드라인",
            "오늘날 `",
            "선택지 옳은 쪽",
            "않을 게 아닐",
            "게임 한 판 게임",
            "게임 한판 게임",
            "게시판에서는",
            "빡센 게시판",
            "점심이면면",
            "옛수풀",
            "잡진 얘기자",
            "가벼운 게임 산책.",
            "물놀이 모래사장 산책.",
            "자전거은",
        )
        if any(snippet in text for snippet in hard_blocked_snippets):
            return True
        if KoBartGenerationClient._has_forbidden_polite_ending(text):
            return True
        if action == "news_answer" and ("`" in text or text.rstrip().endswith(("전체", "전체로", "전체적"))):
            return True

        blocked_snippets = (
            "한 줄만 더",
            "한 줄 더",
            "또 오면",
            "바로 이어",
            "그다음은",
            "나중에 또",
            "하나만 더",
        )
        if sum(text.count(snippet) for snippet in blocked_snippets) >= 2:
            return True

        if KoBartGenerationClient._has_immediate_repeated_token(text):
            return True
        if KoBartGenerationClient._has_repeated_hangul_fragment(text):
            return True

        if len(text) >= 70 and text.count(".") == 0 and text.count("!") == 0 and text.count("?") == 0:
            return True

        action_blocked = KoBartGenerationClient._ACTION_BLOCKED_SNIPPETS.get(action, ())
        if any(snippet in text for snippet in action_blocked):
            return True
        if action == "share_opinion" and KoBartGenerationClient._looks_like_bare_comfort_reply(text):
            return True

        if "avoid_overfamiliarity" in constraints:
            if any(token in text for token in ("ㅎㅇ", "반가워~", "왔구나", "야 ", "ㅋㅋ")):
                return True

        if "respect_boundary_history" in constraints:
            if any(token in text for token in ("더 말해봐", "계속해", "이어봐", "한 줄만 더")):
                return True

        if "no_question_mark" in constraints and "?" in text:
            return True

        if "avoid_self_insertion" in constraints:
            stripped = text.strip()
            if stripped.startswith("나는 ") or stripped.startswith("난 ") or "나는 막 " in stripped:
                return True

        if "avoid_emotional_comfort" in constraints:
            if any(snippet in text for snippet in ("그럴 수 있어", "힘들겠다", "위로", "공감")):
                return True
            if KoBartGenerationClient._looks_like_bare_comfort_reply(text):
                return True

        if "direct_opinion_only" in constraints:
            if any(snippet in text for snippet in ("그 감정", "그 여운", "여운이 오래", "반응하는 쪽")):
                return True
            if "템포" in text:
                return True

        if action == "share_feeling" and any(
            snippet in text for snippet in ("그 감정의 여운", "그 감정은 그 순간부터", "반응하는 쪽이 더 맞아 보여")
        ):
            return True

        return False

    @staticmethod
    def _has_malformed_surface_text(text: str) -> bool:
        raw = str(text or "")
        if "\ufffd" in raw:
            return True
        if re.search(r"[\u4e00-\u9fff]", raw):
            return True
        for token in re.findall(r"[A-Za-z가-힣0-9]+", raw):
            if re.search(r"[a-z]", token) and re.search(r"[가-힣]", token):
                return True
            if KoBartGenerationClient._looks_like_garbled_latin_token(token):
                return True
        return False

    @staticmethod
    def _looks_like_garbled_latin_token(token: str) -> bool:
        if not re.search(r"[A-Za-z]", token) or re.search(r"[가-힣]", token):
            return False
        normalized = re.sub(r"[^A-Za-z0-9]+", "", token)
        if len(normalized) < 4:
            return False
        allowed = {
            "AKMU",
            "BFX",
            "FPS",
            "HTML",
            "LCK",
            "LOL",
            "MBTI",
            "OBS",
            "PC",
            "RPG",
            "T1",
            "TTS",
            "USB",
            "VRM",
        }
        if normalized.upper() in allowed:
            return False
        if normalized.isupper() and len(normalized) <= 8:
            return False
        return bool(re.search(r"[a-z]{4,}", normalized))

    @staticmethod
    def _looks_like_bare_comfort_reply(text: str) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()
        if compact in {"괜찮아", "괜찮지", "그럴수있어", "그럴수있지"}:
            return True
        if compact.startswith(("괜찮아", "괜찮지")) and len(compact) <= 18:
            return True
        return False

    @staticmethod
    def _has_forbidden_polite_ending(text: str) -> bool:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if not compact:
            return False
        for sentence in re.split(r"[.!?~。]+", compact):
            if re.search(r"(요|죠|니다|세요|습니까|나요|까요)\s*$", sentence.strip()):
                return True
        return False

    @classmethod
    def _has_immediate_repeated_token(cls, text: str) -> bool:
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
        prev = ""
        for token in tokens:
            if token == prev and len(token) >= 2 and token not in cls._ALLOWED_REPEAT_TOKENS:
                return True
            prev = token
        return False

    @staticmethod
    def _has_repeated_hangul_fragment(text: str) -> bool:
        for token in re.findall(r"[가-힣]{6,}", text):
            for size in range(3, min(6, len(token) // 2 + 1)):
                seen: dict[str, int] = {}
                for index in range(0, len(token) - size + 1):
                    fragment = token[index : index + size]
                    first_index = seen.get(fragment)
                    if first_index is not None and index - first_index >= size:
                        return True
                    seen.setdefault(fragment, index)
        return False

    @classmethod
    def _topic_lock_missing(cls, *, user_text: str, reply_text: str, action: str) -> bool:
        if action not in cls._TOPIC_LOCK_ACTIONS:
            return False
        user_anchors = cls._extract_topic_anchors(user_text)
        if len(user_anchors) < 3:
            return False
        reply_norm = re.sub(r"[^가-힣A-Za-z0-9]+", "", reply_text)
        return not any(anchor in reply_norm for anchor in user_anchors)

    @classmethod
    def _extract_topic_anchors(cls, text: str) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[가-힣A-Za-z0-9]+", text):
            for form in cls._topic_forms(token):
                if len(form) < 2 or form in cls._TOPIC_STOPWORDS or form in seen:
                    continue
                seen.add(form)
                anchors.append(form)
        return anchors[:8]

    @classmethod
    def _topic_forms(cls, token: str) -> tuple[str, ...]:
        candidates = [token]
        for suffix in (
            "이었다",
            "였다",
            "이었다",
            "이었",
            "했다",
            "했어",
            "했네",
            "했는데",
            "하면",
            "하는",
            "한다",
            "하다",
            "하게",
            "하고",
            "해",
            "일까봐",
            "일까",
            "인가봐",
            "인가",
            "같진",
            "같지",
            "같은",
            "같아",
            "같네",
            "으로",
            "에서",
            "에게",
            "한테",
            "처럼",
            "까지",
            "부터",
            "보다",
            "이라",
            "라서",
            "아서",
            "어서",
            "는데",
            "으로는",
            "로는",
            "이랑",
            "랑",
            "하고",
            "으로도",
            "로도",
            "으로만",
            "로만",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "도",
            "만",
            "에",
            "의",
        ):
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                candidates.append(token[: -len(suffix)])
        return tuple(dict.fromkeys(candidates))

    @classmethod
    def _finalize_generated(cls, generated: str, *, action: str, facts: dict) -> str:
        normalized = cls._normalize_output(generated)
        if not normalized:
            raise RuntimeError("KoBART returned an empty response.")

        user_text = str(facts.get("user_text", "")).strip()
        world_state = facts.get("world_state")
        constraints: tuple[str, ...] = ()
        if isinstance(world_state, dict):
            raw_constraints = world_state.get("constraints") or []
            if isinstance(raw_constraints, list):
                constraints = tuple(str(item).strip() for item in raw_constraints if str(item).strip())
        elif isinstance(facts.get("constraints"), list):
            constraints = tuple(str(item).strip() for item in facts.get("constraints", []) if str(item).strip())

        if cls._looks_like_prompt_echo(normalized, user_text=user_text):
            raise RuntimeError("KoBART echoed the prompt instead of generating a reply.")
        if cls._looks_unusable(normalized, action=action, constraints=constraints):
            raise RuntimeError("KoBART generated an unusable reply.")
        draft_reply = ""
        draft_utterance = facts.get("draft_utterance")
        if isinstance(draft_utterance, dict):
            draft_reply = str(draft_utterance.get("draft_reply") or "")
        if cls._topic_lock_missing(user_text=user_text, reply_text=normalized, action=action) and not cls._draft_lock_satisfied(
            draft_reply=draft_reply,
            reply_text=normalized,
        ):
            raise RuntimeError("KoBART drifted away from the user's topic.")
        return normalized

    @classmethod
    def _draft_lock_satisfied(cls, *, draft_reply: str, reply_text: str) -> bool:
        if not draft_reply:
            return False
        draft_anchors = cls._extract_topic_anchors(draft_reply)
        if not draft_anchors:
            return False
        reply_norm = re.sub(r"[^가-힣A-Za-z0-9]+", "", reply_text)
        matched = [anchor for anchor in draft_anchors if anchor in reply_norm]
        if len(draft_anchors) <= 2:
            return bool(matched)
        return len(matched) >= 2
