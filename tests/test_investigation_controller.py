import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.controller import investigation_controller as controller
from src.database.evidence_loader import load_scenario_dataset
from src.rag.schemas.chunk_schema import Chunk, ChunkMetadata
from src.rag.schemas.retrieval_result import RetrievalResult


def build_result(source_id: str, level: int, score: float) -> RetrievalResult:
    """RAG 검색 결과처럼 보이는 테스트용 RetrievalResult를 만든다."""
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=f"case001_{source_id}",
            document_id="case001",
            content=f"content for {source_id}",
            metadata=ChunkMetadata(
                scenario_id="case001",
                level=level,
                category="test",
                source_id=source_id,
                tags=[],
            ),
            embedding=[score],
        ),
        score=score,
    )


def build_evidence(
    source_id: str,
    level: int,
    visibility: str,
    spoiler_level: str,
    *,
    title: str,
    keywords: list[str] | None = None,
    entities: list[SimpleNamespace] | None = None,
    image_url: str | None = None,
    image_alt: str | None = None,
):
    """Pydantic 모델 대신 빠르게 쓰는 테스트용 단서 객체."""
    return SimpleNamespace(
        id=source_id,
        title=title,
        level=level,
        visibility=visibility,
        spoiler_level=spoiler_level,
        summary="",
        location="",
        keywords=keywords or [],
        tags=[],
        entities=entities or [],
        image_url=image_url,
        image_alt=image_alt,
    )


def build_dataset():
    """잠금 해제 규칙을 검증하기 위한 최소 시나리오."""
    return SimpleNamespace(
        scenario_id="case001",
        title="Test Case",
        difficulty=2,
        overview=SimpleNamespace(
            player_intro="A detective is missing.",
            internal_summary="Hidden solution summary.",
        ),
        characters=[
            SimpleNamespace(id="C_DETECTIVE", name="detective", aliases=["friend"]),
        ],
        evidence=[
            build_evidence(
                "E01",
                1,
                "public",
                "hint",
                title="memo clue",
                keywords=["final location", "trojan horse"],
                image_url="/assets/evidence/memo-clue.png",
                image_alt="memo image",
            ),
            build_evidence(
                "E02",
                1,
                "public",
                "critical",
                title="beer clue",
                keywords=["beer", "world record"],
                image_url="/assets/evidence/beer-clue.png",
                image_alt="beer image",
            ),
            build_evidence(
                "E03",
                3,
                "locked",
                "solution",
                title="solution clue",
                keywords=["Yeoju", "trojan horse"],
                entities=[SimpleNamespace(name="Yeoju", aliases=[])],
            ),
        ],
        unlock_rules=[
            SimpleNamespace(
                target_evidence_id="E03",
                condition=SimpleNamespace(
                    type="evidence_found",
                    evidence_ids=["E01", "E02"],
                    min_level=None,
                    question_ids=[],
                    fact_ids=[],
                ),
            )
        ],
        questions=[
            SimpleNamespace(
                id="Q01",
                question="What does the beer mean?",
                expected_evidence_ids=["E02"],
                min_level=1,
            ),
            SimpleNamespace(
                id="Q02",
                question="Where is the final location?",
                expected_evidence_ids=["E03"],
                min_level=3,
            ),
        ],
        accusation_checks=SimpleNamespace(
            culprit_character_id="C_DETECTIVE",
            required_evidence_ids=["E01", "E02", "E03"],
            accepted_motives=["quiz"],
            accepted_methods=["memo"],
        ),
        solution=SimpleNamespace(
            culprit_character_id="C_DETECTIVE",
            motive="quiz",
            method="memo and beer",
            decisive_evidence_ids=["E01", "E02", "E03"],
            explanation="Yeoju is the final location.",
        ),
    )


class FakeRAGService:
    """컨트롤러 테스트에서 실제 벡터 검색 대신 쓰는 가짜 RAG 서비스."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.final_results = [
            build_result("E03", 3, 0.9),
            build_result("E01", 1, 0.4),
            build_result("E02", 1, 0.3),
        ]
        self.beer_results = [
            build_result("E02", 1, 0.8),
            build_result("E01", 1, 0.3),
            build_result("E03", 3, 0.2),
        ]
        self.irrelevant_results = [
            build_result("E01", 1, 0.1),
            build_result("E02", 1, 0.09),
            build_result("E03", 3, 0.08),
        ]

    def search(self, **kwargs):
        self.calls.append(kwargs)
        allowed_level = kwargs.get("level_lte")
        query = kwargs.get("query", "")
        if "beer" in query:
            results = self.beer_results
        elif "weather" in query:
            results = self.irrelevant_results
        else:
            results = self.final_results
        return [
            result
            for result in results
            if allowed_level is None or result.chunk.metadata.level <= allowed_level
        ]


class InvestigationControllerTests(unittest.TestCase):
    """게임 진행, 잠금 규칙, LLM 의도 라우팅, 세션 저장을 함께 검증한다."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_intent_classifier_flag = os.environ.get("LLM_INTENT_CLASSIFIER")
        os.environ["LLM_INTENT_CLASSIFIER"] = "0"
        os.environ["SESSION_SQLITE_PATH"] = os.path.join(
            self.temp_dir.name,
            "sessions.sqlite3",
        )
        controller._sessions = {}
        controller._session_store = None
        controller._session_store_path = None

    def tearDown(self) -> None:
        os.environ.pop("SESSION_SQLITE_PATH", None)
        if self.old_intent_classifier_flag is None:
            os.environ.pop("LLM_INTENT_CLASSIFIER", None)
        else:
            os.environ["LLM_INTENT_CLASSIFIER"] = self.old_intent_classifier_flag
        controller._sessions = {}
        controller._session_store = None
        controller._session_store_path = None
        self.temp_dir.cleanup()

    def test_locked_solution_is_not_returned_before_unlock_rule_is_met(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            response = controller._process_question_sync(
                "final location?",
                session_id="session-a",
            )

        self.assertEqual(response["allowed_level"], 1)
        self.assertEqual(response["next_allowed_level"], 1)
        self.assertEqual(service.calls[0]["level_lte"], 1)
        self.assertEqual(response["answer"], "answer from E01")
        self.assertEqual([item["id"] for item in response["evidence"]], ["E01"])
        self.assertEqual(response["evidence"][0]["image"]["url"], "/assets/evidence/memo-clue.png")

    def test_case_start_returns_player_safe_opening(self) -> None:
        with patch(
            "src.controller.investigation_controller._load_current_dataset",
            return_value=build_dataset(),
        ):
            response = controller._case_start_sync(session_id="session-start")

        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["title"], "Test Case")
        self.assertEqual(response["intro"], "A detective is missing.")
        self.assertEqual(response["state"]["allowed_level"], 1)
        self.assertEqual(response["suggested_questions"], ["What does the beer mean?"])
        self.assertNotIn("Hidden solution summary.", str(response))
        self.assertNotIn("Yeoju", str(response))

    def test_progress_reports_initial_state_without_solution_leak(self) -> None:
        with patch(
            "src.controller.investigation_controller._load_current_dataset",
            return_value=build_dataset(),
        ):
            response = controller._progress_sync(session_id="session-progress-a")

        self.assertEqual(response["status"], "progress")
        self.assertEqual(response["allowed_level"], 1)
        self.assertEqual(response["discovered_evidence_count"], 0)
        self.assertEqual(response["found_required_evidence_count"], 0)
        self.assertEqual(response["required_evidence_count"], 3)
        self.assertFalse(response["can_submit"])
        self.assertEqual(response["discovered_evidence"], [])
        self.assertNotIn("Hidden solution summary.", str(response))
        self.assertNotIn("Yeoju", str(response))

    def test_progress_updates_after_discovered_evidence(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._load_current_dataset",
            return_value=build_dataset(),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-progress-b")
            controller._process_question_sync("beer clue?", session_id="session-progress-b")
            controller._process_question_sync("final location?", session_id="session-progress-b")
            response = controller._progress_sync(session_id="session-progress-b")

        self.assertEqual(response["allowed_level"], 3)
        self.assertEqual(response["question_count"], 3)
        self.assertEqual(response["discovered_evidence_count"], 3)
        self.assertEqual(response["found_required_evidence_count"], 3)
        self.assertEqual(response["missing_required_evidence_count"], 0)
        self.assertTrue(response["can_submit"])
        self.assertEqual(
            [item["id"] for item in response["discovered_evidence"]],
            ["E01", "E02", "E03"],
        )

    def test_repeating_same_question_does_not_unlock_solution(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-b")
            response = controller._process_question_sync(
                "final location?",
                session_id="session-b",
            )

        self.assertEqual(response["allowed_level"], 1)
        self.assertEqual(service.calls[1]["level_lte"], 1)
        self.assertEqual(response["evidence"][0]["id"], "E01")

    def test_unlock_rule_allows_solution_after_distinct_required_clue(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-b")
            beer_response = controller._process_question_sync(
                "beer clue?",
                session_id="session-b",
            )
            response = controller._process_question_sync(
                "final location?",
                session_id="session-b",
            )

        self.assertEqual(beer_response["evidence"][0]["id"], "E02")
        self.assertEqual(beer_response["next_allowed_level"], 3)
        self.assertEqual(response["allowed_level"], 3)
        self.assertEqual(response["evidence"][0]["id"], "E03")

    def test_unlock_state_does_not_cross_sessions(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-c")
            response = controller._process_question_sync(
                "final location?",
                session_id="session-d",
            )

        self.assertEqual(response["allowed_level"], 1)
        self.assertNotIn("E03", [item["id"] for item in response["evidence"]])

    def test_locked_specific_term_does_not_discover_public_clue(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            response = controller._process_question_sync(
                "is Yeoju the answer?",
                session_id="session-e",
            )

        self.assertEqual(response["status"], "locked_clue")
        self.assertEqual(response["allowed_level"], 1)
        self.assertEqual(response["next_allowed_level"], 1)
        self.assertEqual(response["evidence"], [])
        self.assertIn("메모 조사", response["answer"])
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-e"].discovered_evidence_ids, set())

    def test_irrelevant_low_confidence_result_does_not_discover_evidence(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            response = controller._process_question_sync(
                "weather?",
                session_id="session-f",
            )

        self.assertEqual(response["status"], "no_clue")
        self.assertEqual(response["evidence"], [])
        self.assertEqual(controller._sessions["session-f"].discovered_evidence_ids, set())

    def test_broad_action_gives_scene_options_without_discovering_evidence(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            response = controller._process_question_sync(
                "서재를 탐색한다",
                session_id="session-action-a",
            )

        self.assertEqual(response["status"], "action_hint")
        self.assertEqual(response["input_type"], "action")
        self.assertEqual(response["evidence"], [])
        self.assertIn("테이블", response["answer"])
        self.assertIn("쓰레기통", response["answer"])
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-action-a"].discovered_evidence_ids, set())

    def test_flavor_actions_do_not_discover_evidence(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            beer_response = controller._process_question_sync(
                "맥주를 마신다",
                session_id="session-flavor-a",
            )
            chair_response = controller._process_question_sync(
                "안락의자에 앉는다",
                session_id="session-flavor-a",
            )
            phone_response = controller._process_question_sync(
                "탐정에게 전화한다",
                session_id="session-flavor-a",
            )
            damage_response = controller._process_question_sync(
                "서재를 부순다",
                session_id="session-flavor-a",
            )

        for response in [beer_response, chair_response, phone_response, damage_response]:
            self.assertEqual(response["status"], "flavor_action")
            self.assertEqual(response["input_type"], "action")
            self.assertEqual(response["evidence"], [])

        self.assertIn("위장", beer_response["answer"])
        self.assertIn("잔소리", chair_response["answer"])
        self.assertIn("받지 않습니다", phone_response["answer"])
        self.assertIn("현장 보존", damage_response["answer"])
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-flavor-a"].discovered_evidence_ids, set())

    def test_memo_action_discovers_public_evidence_without_rag_search(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            response = controller._process_question_sync(
                "테이블 위 메모를 확인한다",
                session_id="session-action-b",
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["input_type"], "action")
        self.assertIn("가장 큰 책", response["answer"])
        self.assertNotIn("입력해보세요", response["answer"])
        self.assertIn("눌린 자국", response["answer"])
        self.assertEqual([item["id"] for item in response["evidence"]], ["E01"])
        self.assertIn("공통 수식어", response["evidence"][0]["content"])
        self.assertNotIn("눌러쓴 자국", response["evidence"][0]["content"])
        self.assertEqual(
            response["evidence"][0]["image"],
            {"url": "/assets/evidence/memo-clue.png", "alt": "memo image"},
        )
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-action-b"].discovered_evidence_ids, {"E01"})

    def test_correct_guided_flow_does_not_force_next_input_on_success(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ):
            responses = [
                controller._process_question_sync(
                    "메모 조사",
                    session_id="session-action-natural-flow",
                ),
                controller._process_question_sync(
                    "가장 큰 책 검색",
                    session_id="session-action-natural-flow",
                ),
                controller._process_question_sync(
                    "가장 큰 우체통 검색",
                    session_id="session-action-natural-flow",
                ),
                controller._process_question_sync(
                    "흑맥주 조사",
                    session_id="session-action-natural-flow",
                ),
                controller._process_question_sync(
                    "기네스 세계기록이랑 관련 있나?",
                    session_id="session-action-natural-flow",
                ),
            ]

        for response in responses:
            self.assertEqual(response["status"], "success")
            self.assertNotIn("입력해보세요", response["answer"])

        self.assertEqual(
            [response["evidence"][0]["id"] for response in responses],
            ["E01", "E04", "E05", "E02", "E03"],
        )
        self.assertEqual(service.calls, [])

    def test_wrong_mixed_input_guides_next_step_after_beer_is_found(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ):
            for question in [
                "메모 조사",
                "가장 큰 책 검색",
                "가장 큰 우체통 검색",
                "흑맥주 조사",
            ]:
                controller._process_question_sync(
                    question,
                    session_id="session-action-wrong-mixed",
                )

            response = controller._process_question_sync(
                "가장 큰 흑맥주",
                session_id="session-action-wrong-mixed",
            )

        self.assertIn("기네스 세계기록이랑 관련 있나?", response["answer"])
        self.assertIn("입력해보세요", response["answer"])
        self.assertEqual(service.calls, [])

    def test_llm_relation_intent_discovers_unseen_unlocked_relation_clue(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            side_effect=[
                controller.IntentClassification(intent="inspect", target="memo", confidence=0.95),
                controller.IntentClassification(
                    intent="inspect",
                    target="book_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="inspect",
                    target="postbox_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(intent="inspect", target="beer", confidence=0.95),
                controller.IntentClassification(
                    intent="ask_relation",
                    target="record_system",
                    confidence=0.95,
                ),
            ],
        ):
            for question in [
                "메모 조사",
                "가장 큰 책 검색",
                "가장 큰 우체통 검색",
                "흑맥주 조사",
            ]:
                controller._process_question_sync(
                    question,
                    session_id="session-action-llm-relation",
                )

            response = controller._process_question_sync(
                "기네스 세계기록이랑 관련 있나?",
                session_id="session-action-llm-relation",
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual([item["id"] for item in response["evidence"]], ["E03"])
        self.assertNotIn("입력해보세요", response["answer"])
        self.assertEqual(service.calls, [])

    def test_llm_final_location_intent_for_trojan_query_discovers_pattern_after_sundial(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            side_effect=[
                controller.IntentClassification(intent="inspect", target="memo", confidence=0.95),
                controller.IntentClassification(
                    intent="inspect",
                    target="book_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="inspect",
                    target="postbox_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(intent="inspect", target="beer", confidence=0.95),
                controller.IntentClassification(
                    intent="ask_relation",
                    target="record_system",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="inspect",
                    target="sundial_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="inspect",
                    target="final_location",
                    confidence=0.95,
                ),
            ],
        ):
            for question in [
                "메모 조사",
                "가장 큰 책 검색",
                "가장 큰 우체통 검색",
                "흑맥주 조사",
                "기네스 세계기록이랑 관련 있나?",
                "가장 큰 해시계를 조사한다",
            ]:
                controller._process_question_sync(
                    question,
                    session_id="session-action-trojan-llm",
                )

            response = controller._process_question_sync(
                "가장 큰 트로이 목마를 조사한다",
                session_id="session-action-trojan-llm",
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual([item["id"] for item in response["evidence"]], ["E07"])
        self.assertNotIn("최종 목적지는 아직", response["answer"])
        self.assertNotIn("입력해보세요", response["answer"])
        self.assertEqual(service.calls, [])

    def test_explicit_sundial_keyword_overrides_wrong_llm_beer_intent(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            side_effect=[
                controller.IntentClassification(intent="inspect", target="memo", confidence=0.95),
                controller.IntentClassification(
                    intent="inspect",
                    target="book_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="inspect",
                    target="postbox_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="inspect",
                    target="sundial_record",
                    confidence=0.95,
                ),
                controller.IntentClassification(intent="inspect", target="beer", confidence=0.95),
                controller.IntentClassification(intent="inspect", target="beer", confidence=0.95),
            ],
        ):
            for question in [
                "메모 조사",
                "가장 큰 책 검색",
                "가장 큰 우체통 검색",
            ]:
                controller._process_question_sync(
                    question,
                    session_id="session-action-sundial-override",
                )

            early_sundial_response = controller._process_question_sync(
                "가장 큰 해시계 검색",
                session_id="session-action-sundial-override",
            )
            controller._process_question_sync(
                "흑맥주 조사",
                session_id="session-action-sundial-override",
            )
            response = controller._process_question_sync(
                "가장 큰 해시계 검색",
                session_id="session-action-sundial-override",
            )

        self.assertEqual(early_sundial_response["status"], "locked_clue")
        self.assertIn("흑맥주 조사", early_sundial_response["answer"])
        self.assertEqual(response["status"], "locked_clue")
        self.assertEqual(response["evidence"], [])
        self.assertNotIn("이미 확인한 단서", response["answer"])
        self.assertIn("기네스 세계기록이랑 관련 있나?", response["answer"])
        self.assertNotIn("E06", controller._sessions["session-action-sundial-override"].discovered_evidence_ids)
        self.assertEqual(service.calls, [])

    def test_llm_intent_routes_natural_inspection_without_keyword_match(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            return_value=controller.IntentClassification(
                intent="inspect",
                target="memo",
                confidence=0.95,
                rationale="natural memo inspection",
            ),
        ):
            response = controller._process_question_sync(
                "눌린 글씨를 비스듬히 비춰본다",
                session_id="session-intent-a",
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["input_type"], "action")
        self.assertEqual([item["id"] for item in response["evidence"]], ["E01"])
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-intent-a"].discovered_evidence_ids, {"E01"})

    def test_llm_flavor_intent_keeps_odd_action_from_touching_rag(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            return_value=controller.IntentClassification(
                intent="flavor",
                target=None,
                confidence=0.94,
                rationale="odd but harmless action",
            ),
        ):
            response = controller._process_question_sync(
                "탐정 흉내를 내며 방을 한 바퀴 돈다",
                session_id="session-intent-b",
            )

        self.assertEqual(response["status"], "flavor_action")
        self.assertEqual(response["input_type"], "action")
        self.assertEqual(response["evidence"], [])
        self.assertIn("탐정 흉내", response["answer"])
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-intent-b"].discovered_evidence_ids, set())

    def test_low_confidence_llm_intent_falls_back_to_keyword_action(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            return_value=controller.IntentClassification(
                intent="unknown",
                target=None,
                confidence=0.2,
                rationale="uncertain",
            ),
        ):
            response = controller._process_question_sync(
                "테이블 위 메모를 확인한다",
                session_id="session-intent-c",
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual([item["id"] for item in response["evidence"]], ["E01"])
        self.assertEqual(service.calls, [])

    def test_llm_intent_still_cannot_unlock_final_target_early(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            return_value=controller.IntentClassification(
                intent="inspect",
                target="final_location",
                confidence=0.98,
                rationale="user jumped to final",
            ),
        ):
            response = controller._process_question_sync(
                "저 말이 있는 곳으로 가자",
                session_id="session-intent-d",
            )

        self.assertEqual(response["status"], "locked_clue")
        self.assertEqual(response["evidence"], [])
        self.assertIn("메모", response["answer"])
        self.assertEqual(service.calls, [])
        self.assertEqual(controller._sessions["session-intent-d"].discovered_evidence_ids, set())

    def test_llm_relation_intent_can_use_recent_context_without_magic_words(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._classify_input_intent",
            side_effect=[
                controller.IntentClassification(
                    intent="inspect",
                    target="memo",
                    confidence=0.95,
                ),
                controller.IntentClassification(
                    intent="ask_relation",
                    target="memo",
                    confidence=0.93,
                ),
            ],
        ):
            controller._process_question_sync(
                "눌린 글씨를 비스듬히 비춰본다",
                session_id="session-intent-e",
            )
            response = controller._process_question_sync(
                "그래서 이게 방금 본 것들이랑 어떻게 붙어?",
                session_id="session-intent-e",
            )

        self.assertEqual(response["status"], "context_hint")
        self.assertEqual(response["evidence"], [])
        self.assertIn("메모", response["answer"])
        self.assertEqual(service.calls, [])

    def test_high_level_action_at_start_guides_previous_actions_without_unlocking(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ):
            response = controller._process_question_sync(
                "트로이의 목마가 있는 최종 여행지는 어디야?",
                session_id="session-action-jump-start",
            )

        self.assertEqual(response["status"], "locked_clue")
        self.assertEqual(response["evidence"], [])
        self.assertIn("메모 조사", response["answer"])
        self.assertEqual(
            controller._sessions["session-action-jump-start"].discovered_evidence_ids,
            set(),
        )
        self.assertEqual(service.calls, [])

    def test_action_flow_preserves_unlock_rules_for_solution_clue(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            controller._process_question_sync(
                "테이블 위 메모를 확인한다",
                session_id="session-action-c",
            )
            locked_response = controller._process_question_sync(
                "트로이의 목마를 조사한다",
                session_id="session-action-c",
            )
            beer_response = controller._process_question_sync(
                "쓰레기통을 뒤져본다",
                session_id="session-action-c",
            )
            final_response = controller._process_question_sync(
                "트로이의 목마를 조사한다",
                session_id="session-action-c",
            )

        self.assertEqual(locked_response["status"], "locked_clue")
        self.assertEqual(locked_response["evidence"], [])
        self.assertIn("가장 큰 책 검색", locked_response["answer"])
        self.assertEqual(beer_response["next_allowed_level"], 3)
        self.assertEqual(final_response["status"], "success")
        self.assertEqual([item["id"] for item in final_response["evidence"]], ["E03"])
        self.assertEqual(service.calls, [])

    def test_sample_scenario_action_flow_requires_intermediate_record_clues(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ):
            memo_response = controller._process_question_sync(
                "테이블 위 메모를 확인한다",
                session_id="session-action-detailed",
            )
            early_final_response = controller._process_question_sync(
                "트로이의 목마를 조사한다",
                session_id="session-action-detailed",
            )
            book_response = controller._process_question_sync(
                "가장 큰 책 기록을 조사한다",
                session_id="session-action-detailed",
            )
            postbox_response = controller._process_question_sync(
                "가장 큰 우체통 기록을 조사한다",
                session_id="session-action-detailed",
            )
            beer_response = controller._process_question_sync(
                "쓰레기통을 뒤져본다",
                session_id="session-action-detailed",
            )
            record_response = controller._process_question_sync(
                "기네스 기록을 확인한다",
                session_id="session-action-detailed",
            )
            sundial_response = controller._process_question_sync(
                "가장 큰 해시계 기록을 조사한다",
                session_id="session-action-detailed",
            )
            pattern_response = controller._process_question_sync(
                "세계에서 가장 큰 트로이의 목마 검색",
                session_id="session-action-detailed",
            )
            final_response = controller._process_question_sync(
                "국내 트로이의 목마 장소 검색",
                session_id="session-action-detailed",
            )

        self.assertEqual([item["id"] for item in memo_response["evidence"]], ["E01"])
        self.assertEqual(early_final_response["status"], "locked_clue")
        self.assertIn("가장 큰 책 검색", early_final_response["answer"])
        self.assertEqual([item["id"] for item in book_response["evidence"]], ["E04"])
        self.assertEqual([item["id"] for item in postbox_response["evidence"]], ["E05"])
        self.assertEqual([item["id"] for item in beer_response["evidence"]], ["E02"])
        self.assertEqual([item["id"] for item in record_response["evidence"]], ["E03"])
        self.assertEqual([item["id"] for item in sundial_response["evidence"]], ["E06"])
        self.assertEqual([item["id"] for item in pattern_response["evidence"]], ["E07"])
        self.assertEqual([item["id"] for item in final_response["evidence"]], ["E08"])
        self.assertEqual(final_response["allowed_level"], 3)
        self.assertEqual(service.calls, [])

    def test_skipped_pattern_question_guides_missing_record_actions(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ):
            controller._process_question_sync(
                "테이블 위 메모를 확인한다",
                session_id="session-action-skipped",
            )
            controller._process_question_sync(
                "가장 큰 책 기록을 조사한다",
                session_id="session-action-skipped",
            )
            response = controller._process_question_sync(
                "네 단어의 공통점을 정리한다",
                session_id="session-action-skipped",
            )

        self.assertEqual(response["status"], "locked_clue")
        self.assertEqual(response["evidence"], [])
        self.assertIn("가장 큰 우체통", response["answer"])
        self.assertEqual(service.calls, [])

    def test_related_followup_uses_latest_evidence_context(self) -> None:
        service = FakeRAGService()
        dataset = load_scenario_dataset(Path("src/database/scenario_v2.example.json"))

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, dataset),
        ):
            controller._process_question_sync(
                "테이블 위 메모를 확인한다",
                session_id="session-followup-a",
            )
            controller._process_question_sync(
                "쓰레기통을 뒤져본다",
                session_id="session-followup-a",
            )
            controller._process_question_sync(
                "기네스 기록을 확인한다",
                session_id="session-followup-a",
            )
            controller._process_question_sync(
                "가장 큰 책 기록을 조사한다",
                session_id="session-followup-a",
            )
            response = controller._process_question_sync(
                "그건 뭐랑 연결 돼?",
                session_id="session-followup-a",
            )

        self.assertEqual(response["status"], "context_hint")
        self.assertEqual(response["evidence"], [])
        self.assertIn("메모", response["answer"])
        self.assertIn("기네스", response["answer"])
        self.assertIn("우체통", response["answer"])
        self.assertIn("해시계", response["answer"])
        self.assertEqual(service.calls, [])

    def test_submit_requires_required_evidence_before_grading(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ):
            response = controller._submit_solution_sync(
                "Yeoju",
                session_id="session-g",
            )

        self.assertEqual(response["status"], "needs_more_evidence")
        self.assertFalse(response["solved"])
        self.assertGreater(response["missing_evidence_count"], 0)
        self.assertFalse(controller._sessions["session-g"].solved)

    def test_submit_solves_after_required_evidence_is_found(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-h")
            controller._process_question_sync("beer clue?", session_id="session-h")
            controller._process_question_sync("final location?", session_id="session-h")
            response = controller._submit_solution_sync(
                "The final location is Yeoju.",
                session_id="session-h",
            )

        self.assertEqual(response["status"], "solved")
        self.assertTrue(response["solved"])
        self.assertTrue(controller._sessions["session-h"].solved)
        self.assertEqual(response["clear_rank"]["rank"], "S")
        self.assertEqual(response["clear_rank"]["question_count"], 3)
        self.assertIn("클리어 랭크: S 랭크", response["answer"])
        self.assertEqual(response["solution"]["decisive_evidence_ids"], ["E01", "E02", "E03"])

    def test_wrong_submit_after_required_evidence_does_not_reveal_solution(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-i")
            controller._process_question_sync("beer clue?", session_id="session-i")
            controller._process_question_sync("final location?", session_id="session-i")
            response = controller._submit_solution_sync(
                "The final location is Seoul.",
                session_id="session-i",
            )

        self.assertEqual(response["status"], "incorrect")
        self.assertFalse(response["solved"])
        self.assertNotIn("solution", response)
        self.assertFalse(controller._sessions["session-i"].solved)

    def test_session_state_persists_in_sqlite_store(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("final location?", session_id="session-j")
            controller._process_question_sync("beer clue?", session_id="session-j")
            controller._process_question_sync("final location?", session_id="session-j")
            controller._sessions = {}
            controller._session_store = None
            controller._session_store_path = None
            response = controller._submit_solution_sync(
                "The final location is Yeoju.",
                session_id="session-j",
            )

        self.assertEqual(response["status"], "solved")
        self.assertTrue(response["solved"])
        self.assertEqual(controller._sessions["session-j"].question_count, 3)
        self.assertEqual(response["clear_rank"]["rank"], "S")

    def test_clear_rank_drops_as_question_count_increases(self) -> None:
        service = FakeRAGService()

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=lambda **kwargs: f"answer from {kwargs['evidence'][0]['id']}",
        ):
            controller._process_question_sync("weather?", session_id="session-rank-a")
            controller._process_question_sync("final location?", session_id="session-rank-a")
            controller._process_question_sync("beer clue?", session_id="session-rank-a")
            controller._process_question_sync("weather?", session_id="session-rank-a")
            controller._process_question_sync("final location?", session_id="session-rank-a")
            response = controller._submit_solution_sync(
                "The final location is Yeoju.",
                session_id="session-rank-a",
            )

        self.assertEqual(response["status"], "solved")
        self.assertEqual(response["clear_rank"]["rank"], "A")
        self.assertEqual(response["clear_rank"]["baseline_question_count"], 3)
        self.assertEqual(response["clear_rank"]["question_count"], 5)

    def test_rag_answer_receives_recent_chat_history(self) -> None:
        service = FakeRAGService()
        calls: list[dict] = []

        def answer_with_history(**kwargs):
            calls.append(kwargs)
            return f"answer from {kwargs['evidence'][0]['id']}"

        with patch(
            "src.controller.investigation_controller._get_rag_runtime",
            return_value=(service, build_dataset()),
        ), patch(
            "src.controller.investigation_controller._generate_chat_answer",
            side_effect=answer_with_history,
        ):
            controller._process_question_sync(
                "final location?",
                session_id="session-memory-a",
            )
            controller._process_question_sync(
                "beer clue?",
                session_id="session-memory-a",
            )

        self.assertEqual(calls[0]["chat_history"], [])
        self.assertEqual([item["role"] for item in calls[1]["chat_history"]], ["user", "assistant"])
        self.assertEqual(calls[1]["chat_history"][0]["content"], "final location?")
        self.assertEqual(calls[1]["chat_history"][1]["content"], "answer from E01")
        self.assertEqual(calls[1]["chat_history"][1]["status"], "success")
        self.assertEqual(calls[1]["chat_history"][1]["evidence_ids"], ["E01"])

    def test_chat_answer_falls_back_without_openai_client(self) -> None:
        with patch(
            "src.controller.investigation_controller._get_openai_client",
            return_value=None,
        ):
            answer = controller._generate_chat_answer(
                question="question",
                evidence=[{"content": "allowed clue"}],
                allowed_level=1,
                next_allowed_level=1,
            )

        self.assertIn("단서는 손에 들어왔습니다", answer)
        self.assertNotIn("allowed clue", answer)
