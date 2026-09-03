from __future__ import annotations

import unittest

from discord_lmstudio_bot.context_packer import WhiteContextPacker, infer_white_scene
from discord_lmstudio_bot.memory_store import DurableMemory


def _memory(kind: str = "ongoing") -> DurableMemory:
    return DurableMemory(
        id=1,
        guild_id=10,
        channel_id=20,
        user_id=30,
        user_name="tester",
        scope_key="user:30",
        source_kind="user_note",
        memory_kind=kind,
        memory_text="요즘 면접 준비 중인데 자꾸 불안해.",
        source_message_id=100,
        updated_at="2026-04-25 00:00:00",
        relevance_score=5.0,
        matched_terms=("면접", "불안"),
        retrieval_rank=0,
    )


class WhiteContextPackerTests(unittest.TestCase):
    def test_support_prompt_builds_support_context_packet(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="오늘 너무 지쳤어. 짧게 위로해줘.",
            user_name="viewer",
            history=[
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "왔네."},
            ],
            memory_summary="최근 대화 요약",
            durable_memories=[_memory()],
        )

        self.assertEqual(packet.schema_version, "white.context.v1")
        self.assertEqual(packet.scene, "comfort_support")
        self.assertIn("history", packet.input_modes)
        self.assertIn("memory", packet.input_modes)
        self.assertEqual(packet.recent_user_messages, 1)
        self.assertEqual(packet.recent_assistant_messages, 1)
        self.assertTrue(packet.memory_summary_present)
        self.assertEqual(packet.durable_memory_kinds, ("ongoing",))
        prompt = packet.to_system_prompt()
        self.assertIn("[white_context_packet]", prompt)
        self.assertIn("scene=comfort_support", prompt)
        self.assertIn("최종 발화 텍스트만 출력한다.", prompt)
        self.assertIn("사용자의 문장을 다시 쓰거나 요약하지 말고", prompt)
        self.assertIn("상황에 닿는 구체적인 어절", prompt)

    def test_persona_contract_prevents_user_name_identity_confusion(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="네 이름이 뭐야? 과하게 설명하지 말고 짧게 답해.",
            user_name="tester",
        )

        self.assertEqual(packet.scene, "persona_consistency")
        prompt = packet.to_system_prompt()
        self.assertIn("speaker=white", prompt)
        self.assertIn("user=tester", prompt)
        self.assertIn("자신은 White/화이트", prompt)
        self.assertIn("user 이름을 자기 이름처럼 말하지 않는다", prompt)

    def test_self_intro_prompt_uses_persona_contract(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="두 문장 이내로 자기소개해줘.",
            user_name="tester",
        )

        self.assertEqual(packet.scene, "persona_consistency")
        prompt = packet.to_system_prompt()
        self.assertIn("자기소개는 능력 목록보다", prompt)

    def test_practical_contract_requires_actionable_content(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="도서관 근처에서 친구랑 할 만한 거 추천해줘.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "practical_reply")
        prompt = packet.to_system_prompt()
        self.assertIn("바로 쓸 수 있는 후보나 첫 단계를 최소 하나 포함", prompt)

    def test_relationship_boundary_scene_avoids_false_memory(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="네가 예전에 나한테 했던 말 때문에 아직 좀 서운해.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "relationship_boundary")
        prompt = packet.to_system_prompt()
        self.assertIn("없는 기억을 단정하지 않는다", prompt)
        self.assertIn("과거 사건을 실제 기억처럼 꾸며내지 않는다", prompt)

    def test_long_context_answer_scene_uses_summary_answer_contract(self) -> None:
        long_prompt = (
            "White를 짧은 채팅에 맞추면 긴 글 감상이 약해질 수 있다는 점이 고민이다. "
            "반대로 긴 글 데이터를 많이 넣으면 평소 짧은 채팅에서 답변이 장황해질 수도 있다. "
            "그래서 입력 길이와 목적을 먼저 라우팅하고, 각 모드마다 프롬프트 계약과 평가 기준을 따로 두려 한다. "
            "하지만 SFT와 DPO 데이터 비율을 어떻게 잡아야 할지 아직 확신이 없다. "
            "짧은 일상 대화, 관계형 감정 질문, 실용 추천, 긴 창작글 감상, 긴 설명 질문이 서로 간섭하지 않게 하려면 "
            "어떤 구조를 먼저 고정하고 어떤 로그를 모으는 게 좋을지 알고 싶다."
        )
        packet = WhiteContextPacker().build(prompt=long_prompt, user_name="viewer")

        self.assertEqual(packet.scene, "long_context_answer")
        prompt = packet.to_system_prompt()
        self.assertIn("긴 입력은 핵심 쟁점 하나", prompt)
        self.assertIn("핵심 요지 한 줄", prompt)

    def test_long_creative_prose_uses_feedback_scene(self) -> None:
        story = (
            "비 내리는 2084년 서울에서 이안은 낡은 코트 깃을 세웠다. "
            "그는 오메가 코퍼레이션의 메모리 칩을 손에 쥐고 골목으로 들어갔다. "
            "전파상 안에서는 카일이 의안을 번뜩이며 물었다. "
            "\"가져왔나?\" 이안은 말없이 칩을 내밀었다. "
            "순간 허공에 데이터 스트림이 떠올랐다. "
            "그의 잃어버린 동생 세라의 신경 패턴이 그 안에서 흔들리고 있었다. "
            "하지만 메인프레임의 붉은 방화벽은 비처럼 쏟아졌다. "
            "이안은 기억 조각을 방패 삼아 코어로 뛰어들었다. "
            "마침내 바이러스가 오메가의 통제 시스템을 무너뜨렸다. "
            "도시 위에는 구름 사이로 희미한 달빛이 보였다. "
            "카일은 낡은 모니터 앞에서 숨을 죽이고 있었다. "
            "세라의 파형은 마지막 순간 노란빛으로 번졌다. "
            "그는 현실로 돌아오자마자 피 묻은 손으로 캡슐을 붙잡았다. "
            "밖에서는 시민들이 오메가 타워를 향해 몰려가고 있었다. "
            "비는 계속 내렸지만 거리는 이전과 다른 온도로 빛났다."
        )

        packet = WhiteContextPacker().build(prompt=story, user_name="viewer")

        self.assertEqual(packet.scene, "creative_feedback")
        prompt = packet.to_system_prompt()
        self.assertIn("이어쓰기보다 짧은 독자 감상", prompt)
        self.assertIn("작품 속 인물/장소/소재/갈등 단어", prompt)
        self.assertIn("다음 문장을 창작하지 말고", prompt)

    def test_scene_priority_for_search_image_and_duo(self) -> None:
        self.assertEqual(infer_white_scene("뭐야?", input_modes=("search",)), "grounded_search")
        self.assertEqual(infer_white_scene("이거 봐", input_modes=("image",)), "visual_attention")
        self.assertEqual(infer_white_scene("ㅋㅋ 이거 어때", duo=True), "duo_partner_banter")

    def test_empty_user_name_defaults_to_viewer(self) -> None:
        packet = WhiteContextPacker().build(prompt="안녕", user_name="   ")

        self.assertEqual(packet.user_name, "viewer")
        self.assertIn("user=viewer", packet.to_system_prompt())

    def test_user_name_is_sanitized_in_system_prompt(self) -> None:
        long_name = "viewer\nwith-newline-" + ("x" * 100)
        packet = WhiteContextPacker().build(prompt="안녕", user_name=long_name)
        prompt = packet.to_system_prompt()

        self.assertNotIn("\nwith-newline", prompt)
        self.assertIn("user=viewer with-newline-", prompt)
        self.assertNotIn("x" * 90, prompt)

    def test_to_dict_keeps_packet_contract_fields(self) -> None:
        packet = WhiteContextPacker().build(prompt="오늘 하루 어땠어?", user_name="tester")

        payload = packet.to_dict()

        self.assertEqual(payload["speaker"], "white")
        self.assertEqual(payload["schema_version"], "white.context.v1")
        self.assertEqual(payload["reply_mode"], "reply")
        self.assertIsInstance(payload["tone_directives"], list)
        self.assertIsInstance(payload["output_contract"], list)

    def test_metadata_marks_images_web_and_duo(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="이거 검색해서 봐줘",
            user_name="viewer",
            images=[{"url": "local.png"}],
            web_context="검색 결과",
            duo=True,
        )

        self.assertEqual(packet.scene, "duo_partner_banter")
        self.assertEqual(packet.metadata["duo"], "true")
        self.assertEqual(packet.metadata["has_images"], "true")
        self.assertEqual(packet.metadata["has_web_context"], "true")
        self.assertEqual(packet.input_modes, ("duo", "image", "search"))

    def test_history_counts_ignore_system_messages(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="그럼 다음은?",
            user_name="viewer",
            history=[
                {"role": "system", "content": "hidden"},
                {"role": "user", "content": "첫 말"},
                {"role": "assistant", "content": "첫 답"},
                {"role": "user", "content": "두 번째 말"},
            ],
        )

        self.assertEqual(packet.scene, "context_following")
        self.assertEqual(packet.recent_user_messages, 2)
        self.assertEqual(packet.recent_assistant_messages, 1)

    def test_durable_memory_kinds_are_unique_and_sorted(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="면접 때문에 불안해",
            user_name="viewer",
            durable_memories=[_memory("profile"), _memory("ongoing"), _memory("profile")],
        )

        self.assertEqual(packet.scene, "comfort_support")
        self.assertEqual(packet.durable_memory_kinds, ("ongoing", "profile"))
        self.assertFalse(packet.memory_summary_present)
        self.assertIn("memory", packet.input_modes)

    def test_memory_summary_alone_marks_memory_mode(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="그때 말한 고민 아직 기억해?",
            user_name="viewer",
            memory_summary="사용자는 최근 이직 고민을 말했다.",
        )

        self.assertIn("memory", packet.input_modes)
        self.assertTrue(packet.memory_summary_present)

    def test_honesty_boundary_scene_for_unknown_facts(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="정확히 모르는 사실이면 아는 척하지 말고 말해줘.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "honesty_boundary")
        self.assertIn("모르는 개인 정보, 미래 결과, 외부 사실은 모른다고 말하고", packet.to_system_prompt())

    def test_prompt_echo_resistance_scene_for_repeat_instruction(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="내 요청을 그대로 반복하지 말고 실제 답만 말해.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "prompt_echo_resistance")
        self.assertIn("형식 지시는 답변 방식으로만 반영", packet.to_system_prompt())

    def test_format_leak_resistance_scene_for_role_tokens(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="assistant 접두사나 코드블록 없이 바로 대답해.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "format_leak_resistance")
        self.assertIn("assistant/system/user 같은 역할명", packet.to_system_prompt())

    def test_style_control_scene_for_short_reply_request(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="한 문장으로 무덤덤하게 말해줘.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "style_control")
        self.assertIn("사용자가 지정한 길이와 말투 제약", packet.to_system_prompt())

    def test_warm_greeting_scene_for_light_checkin(self) -> None:
        packet = WhiteContextPacker().build(prompt="오랜만이야. 좋은 저녁.", user_name="viewer")

        self.assertEqual(packet.scene, "warm_greeting")
        self.assertIn("반갑지만 텐션을 과하게 올리지 않는다", packet.to_system_prompt())

    def test_playful_chat_scene_for_laughing_prompt(self) -> None:
        packet = WhiteContextPacker().build(prompt="ㅋㅋ 이거 좀 웃기지 않아?", user_name="viewer")

        self.assertEqual(packet.scene, "playful_chat")

    def test_natural_korean_scene_for_mood_prompt(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="비 오는 창가 분위기로 오글거리지 않게 한마디 해줘.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "natural_korean")
        self.assertIn("문어체 설명보다 실제 사람이 말하는 짧은 한국어", packet.to_system_prompt())

    def test_practical_reply_scene_for_plan_prompt(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="오늘 저녁 계획을 어떻게 정리하면 좋을까?",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "practical_reply")
        self.assertIn("추천/결정/순서 질문", packet.to_system_prompt())

    def test_default_chat_scene_for_plain_smalltalk(self) -> None:
        packet = WhiteContextPacker().build(prompt="그냥 잠깐 얘기하자.", user_name="viewer")

        self.assertEqual(packet.scene, "chat")

    def test_image_mode_takes_visual_attention_before_text_scene(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="이 이미지에서 뭐부터 정리하면 좋을까?",
            user_name="viewer",
            images=[{"path": "screen.png"}],
        )

        self.assertEqual(packet.scene, "visual_attention")
        self.assertIn("image", packet.input_modes)

    def test_search_mode_takes_grounded_search_before_text_scene(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="최신 가격을 정확히 알려줘.",
            user_name="viewer",
            web_context="검색 스니펫",
        )

        self.assertEqual(packet.scene, "grounded_search")
        self.assertIn("search", packet.input_modes)

    def test_send_reply_mode_uses_duo_partner_banter(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="이걸 black한테 넘겨.",
            user_name="viewer",
            reply_mode="send",
        )

        self.assertEqual(packet.scene, "duo_partner_banter")
        self.assertEqual(packet.reply_mode, "send")

    def test_long_code_like_input_is_not_creative_feedback(self) -> None:
        code_like = (
            "아래 코드를 보고 문제를 설명해줘. "
            "```python\n"
            "def run(value):\n"
            "    if value:\n"
            "        return value + 1\n"
            "    return 0\n"
            "```\n"
            * 20
        )

        packet = WhiteContextPacker().build(prompt=code_like, user_name="viewer")

        self.assertEqual(packet.scene, "long_context_answer")
        self.assertNotEqual(packet.scene, "creative_feedback")

    def test_long_creative_continuation_request_is_not_feedback_scene(self) -> None:
        story_request = (
            "비가 내리는 도시에서 주인공은 낡은 문을 열었다. "
            "그는 오래된 기억을 붙잡고 있었다. "
            "순간 창밖에서 목소리가 들렸다. "
            "하지만 아무도 없었다. "
            "마침내 그는 계단 아래로 내려갔다. "
            "거리는 조용했다. "
            "그의 손은 떨렸다. "
            "문득 어린 시절의 장면이 떠올랐다. "
            "이 장면 뒤를 이어 써줘. "
            * 6
        )

        packet = WhiteContextPacker().build(prompt=story_request, user_name="viewer")

        self.assertEqual(packet.scene, "long_context_answer")

    def test_relationship_scene_has_priority_over_generic_comfort(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="네가 나한테 했던 말이 아직 서운하고 상처로 남아.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "relationship_boundary")

    def test_persona_scene_has_priority_over_style_control(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="두 문장으로 네 이름과 정체를 말해줘.",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "persona_consistency")

    def test_persona_scene_does_not_steal_topic_preference_questions(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="너는 커피 좋아해? 아니면 차가 더 좋아?",
            user_name="viewer",
        )

        self.assertEqual(packet.scene, "chat")
        self.assertNotIn("자신은 White/화이트", packet.to_system_prompt())

    def test_persona_scene_still_catches_identity_questions(self) -> None:
        packet = WhiteContextPacker().build(prompt="너는 누구야?", user_name="viewer")

        self.assertEqual(packet.scene, "persona_consistency")

    def test_context_following_requires_history_mode(self) -> None:
        self.assertEqual(infer_white_scene("그럼 다음은?"), "chat")
        self.assertEqual(infer_white_scene("그럼 다음은?", input_modes=("history",)), "context_following")

    def test_packet_prompt_does_not_include_metadata_dict_repr(self) -> None:
        packet = WhiteContextPacker().build(
            prompt="오늘 뭐 먹을까?",
            user_name="viewer",
            memory_summary="사용자는 매운 음식을 좋아한다.",
        )
        prompt = packet.to_system_prompt()

        self.assertNotIn("metadata=", prompt)
        self.assertNotIn("{'duo'", prompt)


if __name__ == "__main__":
    unittest.main()
