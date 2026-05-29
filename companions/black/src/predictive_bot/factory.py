from __future__ import annotations

import logging
from pathlib import Path
import sys

from predictive_bot.config import AppConfig
from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.policy import HierarchicalPolicy, PolicyActionScorer
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore, SQLiteStateStore
from predictive_bot.core.tools import (
    BasicKnowledgeService,
    CuratedRecommendationService,
    GoogleNewsRssService,
    OpenMeteoWeatherService,
    SystemTimeService,
    WikidataKnowledgeService,
)
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder
logger = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from bot_shared.speech import VoiceProfile, build_speech_runtime


def build_engine(config: AppConfig) -> PredictiveEngine:
    llm_client = None
    if config.black_draft_only:
        logger.info("BLACK_DRAFT_ONLY=true 이므로 생성기를 로드하지 않고 draft-only 렌더링을 사용합니다.")
    elif config.generation_backend == "kobart":
        from predictive_bot.llm.kobart_client import KoBartGenerationClient

        logger.info("KoBART 생성기를 로드합니다: %s", config.kobart_model_name_or_path)
        llm_client = KoBartGenerationClient(
            model_name_or_path=config.kobart_model_name_or_path,
            device=config.kobart_device,
            max_new_tokens=config.kobart_max_new_tokens,
            num_beams=config.kobart_num_beams,
            input_mode=config.kobart_input_mode,
            output_guard_enabled=config.llm_output_guard_enabled,
        )
        logger.info("KoBART 생성기 로드 완료")
    elif config.generation_backend in {"causal_lm", "transformers_causal_lm", "gemma"}:
        from predictive_bot.llm.causal_client import CausalLMGenerationClient

        logger.info("Causal LM 생성기를 로드합니다: %s", config.causal_lm_model_name_or_path)
        llm_client = CausalLMGenerationClient(
            model_name_or_path=config.causal_lm_model_name_or_path,
            device=config.causal_lm_device,
            max_new_tokens=config.causal_lm_max_new_tokens,
            temperature=config.causal_lm_temperature,
            top_p=config.causal_lm_top_p,
            quantization=config.causal_lm_quantization,
            output_guard_enabled=config.llm_output_guard_enabled,
        )
        logger.info("Causal LM 생성기 로드 완료")
    elif config.generation_backend == "openai" and config.llm_enabled:
        from predictive_bot.llm.client import OpenAICompatibleClient

        llm_client = OpenAICompatibleClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url,
            timeout_seconds=config.openai_timeout_seconds,
        )
    elif config.generation_backend == "openai":
        logger.warning(
            "GENERATION_BACKEND=openai 이지만 OPENAI_API_KEY / OPENAI_MODEL이 없어 템플릿 모드로 동작합니다."
        )

    action_selector = ActionSelector(default_location=config.default_location)
    renderer = ResponseRenderer(
        llm_client=llm_client,
        persona=config.bot_persona,
        kobart_input_mode=config.kobart_input_mode,
        strict_llm_only=config.strict_llm_only,
        draft_only=config.black_draft_only,
        output_guard_enabled=config.llm_output_guard_enabled,
    )

    bert_model = None
    learned_model = None
    meaning_trusted_axes = None
    action_scorer = None

    if config.intent_model_type == "kcbert":
        kcbert_path = Path(config.kcbert_model_path)
        if kcbert_path.exists():
            try:
                from predictive_bot.core.bert_classifier import KcBertIntentClassifier

                logger.info("KcBERT 분류기 로드 중: %s", kcbert_path)
                bert_model = KcBertIntentClassifier(
                    model_dir=kcbert_path,
                    device=config.kcbert_device,
                )
                logger.info("KcBERT 분류기 로드 완료")
            except Exception as exc:
                logger.warning(
                    "KcBERT 분류기를 로드할 수 없어 heuristic 모드로 동작합니다: %s",
                    exc,
                )
        else:
            logger.warning("KcBERT 모델 경로가 존재하지 않습니다: %s (heuristic 모드로 동작)", kcbert_path)
    elif config.intent_model_type in {"meaning", "meaning_model", "modernbert", "modernbert_meaning", "multihead"}:
        meaning_model_path = Path(config.kcbert_model_path)
        meaning_trusted_axes = config.meaning_trusted_axes
        if meaning_model_path.exists():
            try:
                from predictive_bot.core.meaning_classifier import MultiHeadMeaningClassifier

                logger.info("Multi-head meaning 분류기 로드 중: %s", meaning_model_path)
                bert_model = MultiHeadMeaningClassifier(
                    model_dir=meaning_model_path,
                    device=config.kcbert_device,
                )
                logger.info("Multi-head meaning 분류기 로드 완료")
            except Exception as exc:
                logger.warning(
                    "Multi-head meaning 분류기를 로드할 수 없어 heuristic 모드로 동작합니다: %s",
                    exc,
                )
        else:
            logger.warning("Meaning 모델 경로가 존재하지 않습니다: %s (heuristic 모드로 동작)", meaning_model_path)
    elif config.intent_model_type == "charngram":
        if config.intent_model_path:
            from predictive_bot.core.intent_model import CharNgramCentroidModel

            learned_model = CharNgramCentroidModel.load(Path(config.intent_model_path))
            logger.info("CharNgram 분류기 로드 완료")
    else:
        logger.warning("알 수 없는 INTENT_MODEL_TYPE: %s (heuristic 모드로 동작)", config.intent_model_type)

    if config.policy_action_model_path:
        action_model_path = Path(config.policy_action_model_path)
        if action_model_path.exists():
            from predictive_bot.core.intent_model import CharNgramCentroidModel

            action_scorer = PolicyActionScorer(CharNgramCentroidModel.load(action_model_path))
            logger.info("Policy action scorer 로드 완료: %s", action_model_path)
        else:
            logger.warning("Policy action scorer 경로가 존재하지 않습니다: %s", action_model_path)

    if config.state_backend == "memory":
        state_store = MemoryStateStore(max_recent_turns=config.state_max_recent_turns)
    else:
        state_store = SQLiteStateStore(
            db_path=config.state_db_path,
            max_recent_turns=config.state_max_recent_turns,
        )

    knowledge_service = BasicKnowledgeService()
    if config.knowledge_backend == "wikidata":
        knowledge_service = BasicKnowledgeService(
            fallback_service=WikidataKnowledgeService(
                user_agent=config.wikidata_user_agent,
                timeout_seconds=config.wikidata_timeout_seconds,
            )
        )
    elif config.knowledge_backend != "builtin":
        logger.warning(
            "지원하지 않는 KNOWLEDGE_BACKEND=%s 입니다. builtin 모드로 동작합니다.",
            config.knowledge_backend,
        )

    return PredictiveEngine(
        classifier=HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=bert_model,
            learned_model=learned_model,
            min_confidence=config.intent_model_min_confidence,
            meaning_trusted_axes=meaning_trusted_axes if bert_model is not None else None,
        ),
        goal_manager=GoalManager(default_location=config.default_location),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector, action_scorer=action_scorer),
        renderer=renderer,
        verifier=ResponseVerifier(),
        weather_service=OpenMeteoWeatherService(),
        knowledge_service=knowledge_service,
        time_service=SystemTimeService(),
        news_service=GoogleNewsRssService(),
        recommendation_service=CuratedRecommendationService(),
        state_store=state_store,
    )


def build_speech_runtime_for_bot(config: AppConfig):
    return build_speech_runtime(
        enabled=config.tts_enabled,
        mode=config.tts_mode,
        provider_name=config.tts_provider,
        output_dir=config.tts_output_dir,
        command_template=config.tts_command_template,
        play_command_template=config.tts_play_command_template,
        audio_format=config.tts_audio_format,
        max_chars=config.tts_max_chars,
        local_player_name=config.tts_local_player,
        obs_output_dir=config.tts_obs_output_dir,
        xtts_server_url=config.tts_xtts_server_url,
        xtts_server_token=config.tts_xtts_server_token,
        xtts_language=config.tts_xtts_language,
        gptsovits_server_url=config.tts_gptsovits_server_url,
        gptsovits_server_token=config.tts_gptsovits_server_token,
        gptsovits_language=config.tts_gptsovits_language,
        profiles={
            "white": VoiceProfile(
                name="white",
                voice_id=config.tts_white_voice_id,
                speed=config.tts_white_speed,
                style=config.tts_white_style,
            ),
            "black": VoiceProfile(
                name="black",
                voice_id=config.tts_black_voice_id,
                speed=config.tts_black_speed,
                style=config.tts_black_style,
            ),
        },
        elevenlabs_api_key=config.tts_elevenlabs_api_key,
        elevenlabs_model_id=config.tts_elevenlabs_model_id,
        elevenlabs_base_url=config.tts_elevenlabs_base_url,
        elevenlabs_request_timeout_seconds=config.tts_elevenlabs_request_timeout_seconds,
    )
