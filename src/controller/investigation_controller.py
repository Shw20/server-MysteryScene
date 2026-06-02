"""MysteryScene 조사 진행 컨트롤러.

이 파일은 채팅 입력 하나를 받아서 다음 순서로 처리한다.
1. 현재 세션에서 열린 단서 목록을 계산한다.
2. 명확한 키워드 입력은 규칙 기반으로 먼저 라우팅한다.
3. 키워드로 부족한 자연어 입력만 LLM 의도 분류로 보조한다.
4. unlock_rules로 잠긴 단서는 공개하지 않고, 다음 행동만 안내한다.
5. 열린 단서만 discovered_evidence_ids에 저장하고 진행도를 갱신한다.

핵심 원칙: LLM은 "보조 판단"만 하고, 단서 공개 여부는 서버 규칙이 최종 결정한다.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.controller.intent_keywords import (
    ACTION_VERB_TERMS,
    BEER_ACTION_TERMS,
    BOOK_ACTION_TERMS,
    CALL_ACTION_TERMS,
    CHAIR_ACTION_TERMS,
    DAMAGE_ACTION_TERMS,
    DRINK_ACTION_TERMS,
    FINAL_ACTION_TERMS,
    FOLLOWUP_REFERENCE_TERMS,
    FOLLOWUP_RELATION_TERMS,
    GENERIC_LOCKED_TERMS,
    LEAVE_ACTION_TERMS,
    MEMO_ACTION_TERMS,
    ODD_OBJECT_ACTION_TERMS,
    PATTERN_ACTION_TERMS,
    POSTBOX_ACTION_TERMS,
    RECORD_ACTION_TERMS,
    SIT_ACTION_TERMS,
    STUDY_ACTION_TERMS,
    SUNDIAL_ACTION_TERMS,
    TROJAN_PATTERN_TERMS,
)
from src.database.evidence_loader import load_evidence_items, load_scenario_dataset
from src.database.scenario_schema import ScenarioDataset
from src.database.session_store import SessionSnapshot, SQLiteSessionStore
from src.rag.bootstrap import create_service
from src.rag.config import RAGSettings
from src.rag.service import RAGService


DEFAULT_SCENARIO_PATH = Path("src/database/scenario_v2.example.json")
SEARCH_POOL_SIZE = 10
RESPONSE_TOP_K = 1
HIGH_CONFIDENCE_SCORE = 0.22
CHAT_HISTORY_LIMIT = 8
CHAT_HISTORY_MAX_CHARS = 500

# LLM 의도 분류는 “사용자 말이 어떤 행동인지”만 판단한다.
# 실제 단서 해금 여부는 아래 서버 규칙이 다시 검사하므로 정답 누출을 막을 수 있다.
INTENT_CONFIDENCE_THRESHOLD = 0.62
INTENT_MAX_RATIONALE_CHARS = 200
ALLOWED_INTENTS = {
    "inspect",
    "ask_relation",
    "flavor",
    "broad_scene",
    "submit_guess",
    "irrelevant",
    "unknown",
}

# LLM이 반환할 수 있는 추상 대상명이다.
# 구체 단서 ID(E01 등)는 INTENT_TARGET_EVIDENCE에서만 매핑한다.
ALLOWED_INTENT_TARGETS = {
    "memo",
    "beer",
    "record_system",
    "book_record",
    "postbox_record",
    "sundial_record",
    "pattern",
    "final_location",
    "scene",
    "chair",
    "none",
}

# 의도 분류 결과를 실제 시나리오 단서 ID로 바꾸는 표.
# final_location은 샘플 테스트 데이터처럼 E08이 없을 때 E03으로 대체될 수 있다.
INTENT_TARGET_EVIDENCE = {
    "memo": ["E01"],
    "beer": ["E02"],
    "record_system": ["E03"],
    "book_record": ["E04"],
    "postbox_record": ["E05"],
    "sundial_record": ["E06"],
    "pattern": ["E07"],
    "final_location": ["E08", "E03"],
}

# 규칙 기반 라우팅이 찾은 단서 ID를 다시 의도명으로 되돌릴 때 사용한다.
EVIDENCE_INTENT_TARGETS = {
    "E01": "memo",
    "E02": "beer",
    "E03": "record_system",
    "E04": "book_record",
    "E05": "postbox_record",
    "E06": "sundial_record",
    "E07": "pattern",
    "E08": "final_location",
}

# 사용자가 올바른 행동을 했을 때 채팅창에 바로 보여주는 조사 기록 문구다.
# 성공 응답에는 "입력해보세요" 같은 직접 가이드를 넣지 않는다.
# 직접 가이드는 locked_clue/no_clue/반복 입력처럼 막혔을 때만 출력한다.
CLUE_ANSWER_TEMPLATES = {
    "E01": (
        "종이에는 눌린 자국이 희미하게 남아 있다.\n"
        "- 책\n- 우체통\n- 해시계\n- 트로이의 목마\n\n"
        "처음에는 서로 관련 없어 보인다. 하지만 이상하게도, 모든 단어 앞에 '가장 큰'을 붙이면 "
        "실제 존재하는 장소처럼 느껴진다. 특히 '가장 큰 책'은 실제로 검색해볼 만한 단서다."
    ),
    "E02": (
        "캔에는 'Guinness'라는 이름이 적혀 있다. 문득 떠오른다.\n"
        "'기네스'는 세계 기록으로도 유명한 이름 아니었던가?\n\n"
        "탐정은 아마 '세계에서 가장 큰 것들'을 의도적으로 연결하고 있는 것 같다."
    ),
    "E03": (
        "좋은 접근이다. 지금까지 조사한 단서들을 보면, 메모 속 단어들은 모두 "
        "'세계에서 가장 큰 것들'과 연결된다.\n\n"
        "아직 남은 단어는:\n- 해시계\n- 트로이의 목마"
    ),
    "E04": (
        "세계에서 가장 큰 책으로 알려진 장소가 검색된다. "
        "탐정은 단순 단어가 아니라, 실제 존재하는 장소들을 단서로 남기고 있는 것 같다.\n\n"
        "다른 단어들도 같은 방식으로 조사할 수 있을지 모른다."
    ),
    "E05": (
        "세계에서 가장 큰 우체통 역시 실제 존재한다. 메모 속 단어들은 모두 "
        "'세계에서 가장 큰 것들'과 관련되어 있는 듯하다.\n\n"
        "그런데... 왜 하필 흑맥주였을까?"
    ),
    "E06": (
        "세계에서 가장 큰 해시계 역시 실제 관광지로 존재한다. 이제 거의 확실하다.\n\n"
        "탐정은 '세계에서 가장 큰 것들'을 따라 여행하고 있었던 것 같다. "
        "마지막 단서는 '트로이의 목마'다."
    ),
    "E07": (
        "세계에서 가장 큰 트로이의 목마와 관련된 장소가 검색된다. "
        "사진 속 목마의 모습이 어딘가 익숙하다.\n\n"
        "탐정은 마지막 단서를 해외가 아니라 국내 장소와 연결한 것 같다. "
        "관련 지역을 조금 더 조사해볼 필요가 있다."
    ),
    "E08": (
        "검색 결과, 거대한 트로이의 목마 조형물이 있는 장소가 나온다.\n\n"
        "경기도 여주.\n\n"
        "탐정의 마지막 목적지는 아마 이곳이었던 것 같다."
    ),
}

# evidence 배열에 함께 내려가는 단서 카드용 요약 문구다.
# 채팅 답변보다 짧고, UI의 진행도/확보 단서 목록에서 읽히는 것을 목표로 한다.
CLUE_DISPLAY_TEMPLATES = {
    "E01": (
        "메모는 네 물건을 그대로 던져두고, 공통 수식어는 일부러 숨겼다. "
        "같은 말을 앞에 붙였을 때 자연스러운지 확인해 보자."
    ),
    "E02": (
        "흑맥주 캔은 술보다 이름값이 더 큰 물건이다. "
        "브랜드가 끌고 오는 다른 분야를 떠올려 보자."
    ),
    "E03": (
        "이 단서는 메모의 물건들을 하나의 기록 체계 안으로 밀어 넣는다. "
        "'가장 큰'이라는 말이 괜히 허세를 부린 게 아니다."
    ),
    "E04": (
        "책은 읽을거리라기보다 기록 대상처럼 보인다. "
        "내용보다 크기를 먼저 재는 쪽이 탐정 취향에 가깝다."
    ),
    "E05": (
        "우체통도 같은 규칙 아래 놓인다. "
        "기능이 아니라 규모가 단서의 중심이다."
    ),
    "E06": (
        "해시계까지 같은 방향을 가리키면 우연이라고 우기기 어렵다. "
        "세 단어가 같은 기록 형식을 반복한다."
    ),
    "E07": (
        "앞의 세 단어가 만든 규칙을 마지막 단어에 넘겨야 한다. "
        "남은 것은 말의 정체가 아니라 말의 크기다."
    ),
    "E08": (
        "마지막 단어도 같은 기록 방식으로 좁혀진다. "
        "목적지를 쓰기 전에, 앞선 조형물 기록들과 같은 구조인지 확인해 보자."
    ),
}


@dataclass
class GameSessionState:
    """한 플레이 세션의 진행 상태.

    discovered_evidence_ids가 늘어나면 unlock_rules를 다시 평가해서 다음 단서가 열린다.
    question_count는 클리어 랭크 산정에도 사용한다.
    """

    discovered_evidence_ids: set[str] = field(default_factory=set)
    answered_question_ids: set[str] = field(default_factory=set)
    confirmed_fact_ids: set[str] = field(default_factory=set)
    question_count: int = 0
    submission_count: int = 0
    solved: bool = False


@dataclass
class IntentClassification:
    """LLM 또는 규칙 라우터가 해석한 사용자 입력 의도.

    intent/target은 라우팅용 힌트일 뿐이며, 실제 단서 공개는 서버 잠금 규칙이 결정한다.
    """

    intent: str = "unknown"
    target: str | None = None
    confidence: float = 0.0
    rationale: str = ""


_rag_service: RAGService | None = None
_scenario_dataset: ScenarioDataset | None = None
_scenario_id: str | None = None
_index_ready = False
_sessions: dict[str, GameSessionState] = {}
_openai_client: object | None = None
_session_store: SQLiteSessionStore | None = None
_session_store_path: str | None = None

# 위 전역 변수들은 런타임 캐시다.
# RAG 인덱스, 시나리오 데이터, 세션 상태를 매 요청마다 다시 만들지 않기 위해 보관한다.


def _scenario_path() -> Path:
    return Path(os.getenv("RAG_SCENARIO_PATH", str(DEFAULT_SCENARIO_PATH)))


def _normalize_session_id(session_id: str | None) -> str:
    value = str(session_id or "default").strip()
    return value[:80] if value else "default"


def _session_db_path() -> str:
    return os.getenv("SESSION_SQLITE_PATH", "data/mysteryscene_sessions.sqlite3")


def _get_session_store() -> SQLiteSessionStore:
    global _session_store, _session_store_path

    path = _session_db_path()
    if _session_store is None or _session_store_path != path:
        _session_store = SQLiteSessionStore(path)
        _session_store_path = path
    return _session_store


def _state_from_snapshot(snapshot: SessionSnapshot) -> GameSessionState:
    return GameSessionState(
        discovered_evidence_ids=set(snapshot.discovered_evidence_ids),
        answered_question_ids=set(snapshot.answered_question_ids),
        confirmed_fact_ids=set(snapshot.confirmed_fact_ids),
        question_count=snapshot.question_count,
        submission_count=snapshot.submission_count,
        solved=snapshot.solved,
    )


def _snapshot_from_state(state: GameSessionState) -> SessionSnapshot:
    return SessionSnapshot(
        discovered_evidence_ids=set(state.discovered_evidence_ids),
        answered_question_ids=set(state.answered_question_ids),
        confirmed_fact_ids=set(state.confirmed_fact_ids),
        question_count=state.question_count,
        submission_count=state.submission_count,
        solved=state.solved,
    )


def _get_session_state(session_id: str, scenario_id: str) -> GameSessionState:
    if session_id not in _sessions:
        snapshot = _get_session_store().load_session(session_id, scenario_id)
        _sessions[session_id] = _state_from_snapshot(snapshot)
    return _sessions[session_id]


def _save_session_state(
    session_id: str,
    scenario_id: str,
    state: GameSessionState,
) -> None:
    _get_session_store().save_session(
        session_id,
        scenario_id,
        _snapshot_from_state(state),
    )


def _record_question_exchange(
    session_id: str,
    scenario_id: str,
    question: str,
    response: dict,
) -> None:
    store = _get_session_store()
    evidence_ids = [item["id"] for item in response.get("evidence", [])]
    store.record_chat_message(
        session_id,
        scenario_id,
        "user",
        question,
    )
    store.record_chat_message(
        session_id,
        scenario_id,
        "assistant",
        str(response.get("answer", "")),
        status=str(response.get("status", "")),
        evidence_ids=evidence_ids,
    )


def _record_submission(
    session_id: str,
    scenario_id: str,
    answer: str,
    response: dict,
) -> None:
    _get_session_store().record_submission(
        session_id,
        scenario_id,
        answer,
        str(response.get("status", "")),
        bool(response.get("solved", False)),
    )


def _recent_chat_history(session_id: str, scenario_id: str) -> list[dict]:
    return _get_session_store().load_recent_chat_messages(
        session_id,
        scenario_id,
        limit=CHAT_HISTORY_LIMIT,
    )


def _get_openai_client(settings: RAGSettings) -> object | None:
    global _openai_client

    if not settings.openai_api_key:
        return None

    if _openai_client is not None:
        return _openai_client

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client_kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url

    _openai_client = OpenAI(**client_kwargs)
    return _openai_client


def _get_rag_runtime() -> tuple[RAGService, ScenarioDataset]:
    global _rag_service, _scenario_dataset, _scenario_id, _index_ready, _sessions

    scenario_path = _scenario_path()
    dataset = load_scenario_dataset(scenario_path)

    if _rag_service is None or _scenario_id != dataset.scenario_id:
        _rag_service = create_service(RAGSettings.from_env())
        _scenario_dataset = dataset
        _scenario_id = dataset.scenario_id
        _index_ready = False
        _sessions = {}

    if not _index_ready:
        items = load_evidence_items(scenario_path)
        _rag_service.index_items(
            items,
            scenario_id=dataset.scenario_id,
            replace_existing=True,
        )
        _index_ready = True

    _scenario_dataset = dataset
    return _rag_service, dataset


def _load_current_dataset() -> ScenarioDataset:
    return load_scenario_dataset(_scenario_path())


def _evidence_by_id(dataset: ScenarioDataset) -> dict[str, object]:
    return {item.id: item for item in dataset.evidence}


def _normalize_match_text(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _terms_from_evidence(item) -> set[str]:
    terms: set[str] = set()
    for value in (
        getattr(item, "title", ""),
        getattr(item, "summary", ""),
        getattr(item, "location", ""),
    ):
        if value:
            terms.add(str(value))

    for value in getattr(item, "keywords", []) or []:
        terms.add(str(value))
    for value in getattr(item, "tags", []) or []:
        terms.add(str(value))

    for entity in getattr(item, "entities", []) or []:
        name = getattr(entity, "name", "")
        if name:
            terms.add(str(name))
        for alias in getattr(entity, "aliases", []) or []:
            terms.add(str(alias))

    return {
        normalized
        for normalized in (_normalize_match_text(term) for term in terms)
        if len(normalized) >= 2
    }


def _specific_terms_from_evidence(item) -> set[str]:
    terms: set[str] = set()
    for value in getattr(item, "keywords", []) or []:
        terms.add(str(value))

    for entity in getattr(item, "entities", []) or []:
        name = getattr(entity, "name", "")
        if name:
            terms.add(str(name))
        for alias in getattr(entity, "aliases", []) or []:
            terms.add(str(alias))

    location = getattr(item, "location", "")
    if location:
        terms.add(str(location))

    return {
        normalized
        for normalized in (_normalize_match_text(term) for term in terms)
        if len(normalized) >= 2 and normalized not in GENERIC_LOCKED_TERMS
    }


def _terms_from_character(character) -> set[str]:
    terms = {str(getattr(character, "name", ""))}
    for alias in getattr(character, "aliases", []) or []:
        terms.add(str(alias))

    return {
        normalized
        for normalized in (_normalize_match_text(term) for term in terms)
        if len(normalized) >= 2
    }


def _has_question_term_match(question: str, item) -> bool:
    question_text = _normalize_match_text(question)
    if not question_text:
        return False

    for term in _terms_from_evidence(item):
        if term in question_text or question_text in term:
            return True
    return False


def _result_matches_question(dataset: ScenarioDataset, result, question: str) -> bool:
    if result.score >= HIGH_CONFIDENCE_SCORE:
        return True

    evidence = _evidence_by_id(dataset)
    item = evidence.get(result.chunk.metadata.source_id)
    return bool(item and _has_question_term_match(question, item))


def _mentions_locked_specific_term(
    dataset: ScenarioDataset,
    question: str,
    unlocked_ids: set[str],
) -> str | None:
    question_text = _normalize_match_text(question)
    if not question_text:
        return None

    evidence = _evidence_by_id(dataset)
    unlocked_terms: set[str] = set()
    for evidence_id in unlocked_ids:
        item = evidence.get(evidence_id)
        if item:
            unlocked_terms.update(_terms_from_evidence(item))

    for item in dataset.evidence:
        if item.id in unlocked_ids:
            continue
        if getattr(item, "visibility", "") == "hidden":
            continue
        if getattr(item, "spoiler_level", "") != "solution":
            continue

        for term in _specific_terms_from_evidence(item):
            if term in unlocked_terms:
                continue
            if term in question_text:
                return item.id

    return None


def _required_evidence_ids(dataset: ScenarioDataset) -> set[str]:
    checks = getattr(dataset, "accusation_checks", None)
    if checks and getattr(checks, "required_evidence_ids", None):
        return set(checks.required_evidence_ids)

    solution = getattr(dataset, "solution", None)
    if solution and getattr(solution, "decisive_evidence_ids", None):
        return set(solution.decisive_evidence_ids)

    return {
        item.id
        for item in dataset.evidence
        if getattr(item, "spoiler_level", "") == "solution"
    }


def _solution_answer_terms(dataset: ScenarioDataset) -> set[str]:
    evidence = _evidence_by_id(dataset)
    terms: set[str] = set()
    for evidence_id in _required_evidence_ids(dataset):
        item = evidence.get(evidence_id)
        if item and getattr(item, "spoiler_level", "") == "solution":
            terms.update(_specific_terms_from_evidence(item))

    if terms:
        return terms

    checks = getattr(dataset, "accusation_checks", None)
    if checks:
        for value in getattr(checks, "accepted_motives", []) or []:
            terms.add(_normalize_match_text(str(value)))
        for value in getattr(checks, "accepted_methods", []) or []:
            terms.add(_normalize_match_text(str(value)))

        culprit_id = getattr(checks, "culprit_character_id", "")
        for character in getattr(dataset, "characters", []) or []:
            if getattr(character, "id", "") == culprit_id:
                terms.update(_terms_from_character(character))

    return {term for term in terms if len(term) >= 2}


def _submission_text(
    answer: str,
    culprit: str | None = None,
    motive: str | None = None,
    method: str | None = None,
) -> str:
    return " ".join(
        part.strip()
        for part in [answer, culprit or "", motive or "", method or ""]
        if part and part.strip()
    )


def _submission_matches_solution(dataset: ScenarioDataset, submission: str) -> bool:
    normalized_submission = _normalize_match_text(submission)
    if not normalized_submission:
        return False

    for term in _solution_answer_terms(dataset):
        if term and term in normalized_submission:
            return True

    return False


def _clear_rank(dataset: ScenarioDataset, state: GameSessionState) -> dict:
    baseline = max(1, len(_required_evidence_ids(dataset)))
    question_count = state.question_count
    thresholds = {
        "S": baseline,
        "A": baseline + 2,
        "B": baseline + 5,
        "C": baseline + 8,
    }

    if question_count <= thresholds["S"]:
        rank = "S"
        description = "필수 단서를 거의 낭비 없이 따라간 완벽한 추리입니다."
    elif question_count <= thresholds["A"]:
        rank = "A"
        description = "조금 헤맸지만 탐정의 장난에 오래 끌려다니지는 않았습니다."
    elif question_count <= thresholds["B"]:
        rank = "B"
        description = "단서의 길은 잡았지만, 탐정이 던진 미끼도 꽤 밟았습니다."
    elif question_count <= thresholds["C"]:
        rank = "C"
        description = "사건은 해결했지만 서재 바닥과 꽤 친해진 수사였습니다."
    else:
        rank = "D"
        description = "해결은 해결입니다. 탐정이 웃고 있을 가능성은 조금 높습니다."

    return {
        "rank": rank,
        "label": f"{rank} 랭크",
        "question_count": question_count,
        "baseline_question_count": baseline,
        "thresholds": thresholds,
        "description": description,
    }


def _opening_objective(dataset: ScenarioDataset) -> str:
    title = getattr(dataset, "title", "사건")
    return f"질문을 통해 단서를 확인하고, '{title}'의 최종 결론을 제출하세요."


def _start_suggested_questions(dataset: ScenarioDataset) -> list[str]:
    questions = [
        item.question
        for item in getattr(dataset, "questions", []) or []
        if getattr(item, "min_level", 1) <= 1
    ]
    if questions:
        return questions[:3]

    return [
        "메모에는 뭐라고 적혀 있었어?",
        "흑맥주 단서는 무엇을 의미해?",
    ]


def _case_start_sync(session_id: str | None = None) -> dict:
    active_session_id = _normalize_session_id(session_id)
    dataset = _load_current_dataset()
    state = _get_session_state(active_session_id, dataset.scenario_id)
    allowed_level = _highest_unlocked_level(dataset, _unlocked_evidence_ids(dataset, state))

    return {
        "status": "ready",
        "scenario_id": dataset.scenario_id,
        "session_id": active_session_id,
        "title": dataset.title,
        "difficulty": dataset.difficulty,
        "intro": dataset.overview.player_intro,
        "objective": _opening_objective(dataset),
        "flow": [
            "서재에서 메모와 흑맥주 흔적을 확인한다.",
            "흑맥주가 가리키는 기록 체계를 확인한다.",
            "책, 우체통, 해시계 기록을 각각 대조한다.",
            "공통 패턴으로 트로이의 목마가 가리키는 최종 목적지를 추리한다.",
        ],
        "suggested_questions": _start_suggested_questions(dataset),
        "state": {
            "allowed_level": allowed_level,
            "discovered_evidence_count": len(state.discovered_evidence_ids),
            "question_count": state.question_count,
            "submission_count": state.submission_count,
            "solved": state.solved,
        },
    }


def _discovered_progress_items(
    dataset: ScenarioDataset,
    state: GameSessionState,
    required_ids: set[str],
) -> list[dict]:
    evidence = _evidence_by_id(dataset)
    items: list[dict] = []
    for evidence_id in sorted(state.discovered_evidence_ids):
        item = evidence.get(evidence_id)
        if not item or getattr(item, "visibility", "") == "hidden":
            continue
        items.append(
            {
                "id": item.id,
                "title": getattr(item, "title", item.id),
                "level": getattr(item, "level", 1),
                "category": getattr(item, "category", "evidence"),
                "is_required": item.id in required_ids,
            }
        )
    return items


def _progress_sync(session_id: str | None = None) -> dict:
    active_session_id = _normalize_session_id(session_id)
    dataset = _load_current_dataset()
    state = _get_session_state(active_session_id, dataset.scenario_id)
    unlocked_ids = _unlocked_evidence_ids(dataset, state)
    allowed_level = _highest_unlocked_level(dataset, unlocked_ids)
    required_ids = _required_evidence_ids(dataset)
    found_required_ids = required_ids & state.discovered_evidence_ids
    discovered_items = _discovered_progress_items(dataset, state, required_ids)
    visible_evidence_count = len(
        [item for item in dataset.evidence if getattr(item, "visibility", "") != "hidden"]
    )

    response = {
        "status": "progress",
        "scenario_id": dataset.scenario_id,
        "session_id": active_session_id,
        "title": dataset.title,
        "allowed_level": allowed_level,
        "question_count": state.question_count,
        "submission_count": state.submission_count,
        "solved": state.solved,
        "discovered_evidence_count": len(discovered_items),
        "total_evidence_count": visible_evidence_count,
        "required_evidence_count": len(required_ids),
        "found_required_evidence_count": len(found_required_ids),
        "missing_required_evidence_count": len(required_ids - state.discovered_evidence_ids),
        "can_submit": required_ids.issubset(state.discovered_evidence_ids),
        "discovered_evidence": discovered_items,
    }
    if state.solved:
        response["clear_rank"] = _clear_rank(dataset, state)
    return response


def _initial_unlocked_evidence_ids(dataset: ScenarioDataset) -> set[str]:
    return {
        item.id
        for item in dataset.evidence
        if item.visibility == "public" and item.spoiler_level != "solution"
    }


def _highest_unlocked_level(dataset: ScenarioDataset, evidence_ids: set[str]) -> int:
    evidence = _evidence_by_id(dataset)
    levels = [
        evidence[evidence_id].level
        for evidence_id in evidence_ids
        if evidence_id in evidence
    ]
    return max(levels, default=1)


def _unlock_condition_met(
    dataset: ScenarioDataset,
    state: GameSessionState,
    condition,
    unlocked_ids: set[str],
) -> bool:
    if condition.type == "evidence_found":
        return set(condition.evidence_ids).issubset(state.discovered_evidence_ids)

    if condition.type == "level_reached":
        required_level = condition.min_level or 1
        return _highest_unlocked_level(dataset, unlocked_ids) >= required_level

    if condition.type == "question_answered":
        return set(condition.question_ids).issubset(state.answered_question_ids)

    if condition.type == "fact_confirmed":
        return set(condition.fact_ids).issubset(state.confirmed_fact_ids)

    return False


def _unlocked_evidence_ids(
    dataset: ScenarioDataset,
    state: GameSessionState,
) -> set[str]:
    evidence = _evidence_by_id(dataset)
    unlocked_ids = _initial_unlocked_evidence_ids(dataset)
    unlocked_ids.update(
        evidence_id
        for evidence_id in state.discovered_evidence_ids
        if evidence_id in evidence and evidence[evidence_id].visibility != "hidden"
    )

    changed = True
    while changed:
        changed = False
        for rule in dataset.unlock_rules:
            if rule.target_evidence_id in unlocked_ids:
                continue
            if rule.target_evidence_id not in evidence:
                continue
            if _unlock_condition_met(dataset, state, rule.condition, unlocked_ids):
                unlocked_ids.add(rule.target_evidence_id)
                changed = True

    return unlocked_ids


def _format_evidence(result) -> dict:
    chunk = result.chunk
    data = {
        "id": chunk.metadata.source_id,
        "category": chunk.metadata.category,
        "level": chunk.metadata.level,
        "score": round(result.score, 4),
        "content": chunk.content,
        "tags": chunk.metadata.tags,
    }
    return data


def _evidence_image(item) -> dict | None:
    image_url = getattr(item, "image_url", None)
    if not image_url:
        return None
    return {
        "url": str(image_url),
        "alt": str(getattr(item, "image_alt", "") or getattr(item, "title", "단서 이미지")),
    }


def _attach_evidence_image(data: dict, item) -> dict:
    image = _evidence_image(item)
    if image:
        data["image"] = image
    return data


def _format_evidence_result(result, dataset: ScenarioDataset) -> dict:
    data = _format_evidence(result)
    item = _evidence_by_id(dataset).get(result.chunk.metadata.source_id)
    if item:
        data["title"] = getattr(item, "title", item.id)
        data["content"] = _clue_display_text(item)
        return _attach_evidence_image(data, item)
    return data


def _format_evidence_item(item, score: float = 1.0) -> dict:
    data = {
        "id": item.id,
        "title": getattr(item, "title", item.id),
        "category": getattr(item, "category", "evidence"),
        "level": getattr(item, "level", 1),
        "score": round(score, 4),
        "content": _clue_display_text(item),
        "tags": getattr(item, "tags", []) or [],
    }
    return _attach_evidence_image(data, item)


def _discover_results(state: GameSessionState, results: list) -> None:
    for result in results:
        state.discovered_evidence_ids.add(result.chunk.metadata.source_id)


def _has_any_term(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _looks_like_action(text: str) -> bool:
    return _has_any_term(text, ACTION_VERB_TERMS)


def _flavor_action_answer(text: str) -> str | None:
    if _has_any_term(text, DAMAGE_ACTION_TERMS):
        return (
            "탐정의 서재를 현장 보존이 아니라 철거 현장으로 만들 생각이라면, 꽤 과감한 추리법입니다. "
            "하지만 단서를 망가뜨리면 탐정만 신나게 비웃을 겁니다. "
            "부수기 전에 테이블의 메모나 쓰레기통 쪽을 먼저 확인해 보세요."
        )
    if _has_any_term(text, DRINK_ACTION_TERMS) and _has_any_term(text, BEER_ACTION_TERMS):
        return (
            "훌륭합니다. 단서를 위장 속에 보관하려는 시도군요. "
            "다만 탐정도 그 정도의 희생까지 바라진 않았을 겁니다. "
            "캔의 맛보다, 그 이름이 어디에 붙어 있는지를 의심해 보세요."
        )
    if _has_any_term(text, SIT_ACTION_TERMS) and _has_any_term(text, CHAIR_ACTION_TERMS):
        return (
            "안락의자에 앉아보니 탐정의 잔소리 냄새가 아직 남아 있습니다. "
            "불쾌하게도 결정적 단서는 아닙니다. "
            "시선은 다시 사이드 테이블과 쓰레기통 쪽으로 돌아갑니다."
        )
    if _has_any_term(text, CALL_ACTION_TERMS):
        return (
            "전화를 걸어도 탐정은 받지 않습니다. "
            "늘 귀찮게 굴던 사람이 조용하니, 이 침묵이 더 얄밉군요. "
            "결국 남은 건 서재의 메모와 쓰레기통뿐입니다."
        )
    if _has_any_term(text, LEAVE_ACTION_TERMS) and (
        _has_any_term(text, STUDY_ACTION_TERMS) or "밖" in text
    ):
        return (
            "서재 밖으로 나가면 잠깐 자유로워질 수는 있겠죠. "
            "하지만 탐정은 단서를 방 안에 두고 사람을 밖으로 밀어내는 취미가 없습니다. "
            "나가기 전에 테이블과 쓰레기통을 한 번 더 괴롭혀 보세요."
        )
    if _has_any_term(text, ODD_OBJECT_ACTION_TERMS) and _looks_like_action(text):
        return (
            "그쪽도 살펴볼 수는 있지만, 탐정의 장난은 그리 넓은 예산을 쓰지 않은 모양입니다. "
            "지금 반응하는 물건은 안락의자, 사이드 테이블, 쓰레기통 쪽에 더 가깝습니다. "
            "괜히 방 전체와 싸우기 전에 눈에 띄는 단서부터 좁혀 보세요."
        )
    return None


def _looks_like_related_followup(text: str) -> bool:
    return _has_any_term(text, FOLLOWUP_REFERENCE_TERMS) and _has_any_term(
        text,
        FOLLOWUP_RELATION_TERMS,
    )


def _latest_context_evidence_id(dataset: ScenarioDataset, chat_history: list[dict]) -> str | None:
    evidence = _evidence_by_id(dataset)
    for message in reversed(chat_history):
        if message.get("role") != "assistant":
            continue
        for evidence_id in reversed(message.get("evidence_ids") or []):
            item = evidence.get(str(evidence_id))
            if item and getattr(item, "visibility", "") != "hidden":
                return item.id
    return None


def _first_existing_evidence_id(dataset: ScenarioDataset, candidates: list[str]) -> str | None:
    evidence = _evidence_by_id(dataset)
    for candidate in candidates:
        if candidate in evidence:
            return candidate
    return None


def _target_or_intro(
    target_id: str | None,
    state: GameSessionState,
    unlocked_ids: set[str],
    intro_id: str | None = "E01",
) -> str | None:
    """향후 인트로 단서로 우회할 수 있도록 남겨둔 라우팅 확장 지점."""

    return target_id


def _should_route_trojan_query_to_pattern(text: str, state: GameSessionState, dataset: ScenarioDataset) -> bool:
    """해시계 이후의 트로이 목마 질문은 최종지(E08)가 아니라 패턴 단서(E07)로 먼저 보낸다."""

    pattern_id = _first_existing_evidence_id(dataset, ["E07"])
    return bool(
        pattern_id
        and "E06" in state.discovered_evidence_ids
        and "E07" not in state.discovered_evidence_ids
        and _has_any_term(text, TROJAN_PATTERN_TERMS)
    )


def _action_target_evidence_id(
    text: str,
    state: GameSessionState,
    unlocked_ids: set[str],
    dataset: ScenarioDataset,
) -> str | None:
    """명확한 키워드 입력을 시나리오 단서 ID로 변환한다.

    이 함수의 결과는 LLM 의도 분류보다 우선한다.
    예를 들어 "가장 큰 해시계 검색"은 LLM이 흑맥주로 오분류해도 E06으로 고정된다.
    """

    record_id = _first_existing_evidence_id(dataset, ["E03"])
    book_id = _first_existing_evidence_id(dataset, ["E04"])
    postbox_id = _first_existing_evidence_id(dataset, ["E05"])
    sundial_id = _first_existing_evidence_id(dataset, ["E06"])
    pattern_id = _first_existing_evidence_id(dataset, ["E07"])
    final_id = _first_existing_evidence_id(dataset, ["E08", "E03"])

    # 기록 대상 단서들은 단어가 서로 겹칠 수 있으므로 구체적인 물건부터 검사한다.
    # 책/우체통/해시계는 각각 E04/E05/E06으로 직접 연결된다.
    if _has_any_term(text, BOOK_ACTION_TERMS):
        return _target_or_intro(book_id, state, unlocked_ids)
    if _has_any_term(text, POSTBOX_ACTION_TERMS):
        return _target_or_intro(postbox_id, state, unlocked_ids)
    if _has_any_term(text, SUNDIAL_ACTION_TERMS):
        return _target_or_intro(sundial_id, state, unlocked_ids)

    # 공통점/패턴 질문은 세 물건을 묶는 E07 쪽으로 보낸다.
    if _has_any_term(text, PATTERN_ACTION_TERMS):
        return _target_or_intro(pattern_id, state, unlocked_ids)

    # 현장 물건 조사는 초반 단서다. 맥주/메모는 레벨 1 단서라 바로 라우팅한다.
    if _has_any_term(text, BEER_ACTION_TERMS):
        return "E02"
    if _has_any_term(text, MEMO_ACTION_TERMS):
        return "E01"

    # 트로이 목마 표현은 최종 목적지(E08)와 패턴 단서(E07)에 모두 걸릴 수 있다.
    # 해시계까지 확인한 상태라면 먼저 E07을 열어야 게임 흐름이 건너뛰지 않는다.
    if _has_any_term(text, FINAL_ACTION_TERMS):
        if _should_route_trojan_query_to_pattern(text, state, dataset):
            return _target_or_intro(pattern_id, state, unlocked_ids)
        return _target_or_intro(final_id, state, unlocked_ids)

    # "기네스", "세계기록" 계열은 맥주를 기록 체계로 연결하는 E03이다.
    if _has_any_term(text, RECORD_ACTION_TERMS):
        if record_id and record_id in unlocked_ids:
            return record_id
        return record_id
    return None


def _intent_classifier_enabled() -> bool:
    """운영 중 의도 분류 LLM을 끄고 싶을 때 쓰는 안전 스위치."""
    flag = os.getenv("LLM_INTENT_CLASSIFIER", "1").strip().casefold()
    return flag not in {"0", "false", "no", "off"}


def _intent_is_confident(intent: IntentClassification | None) -> bool:
    """LLM 판단을 라우팅에 써도 될 만큼 확신도가 높은지 확인한다."""
    return bool(intent and intent.confidence >= INTENT_CONFIDENCE_THRESHOLD)


def _normalize_intent_target(target: object) -> str | None:
    """LLM이 반환한 target 값을 허용된 내부 target 이름으로 정리한다."""
    if target is None:
        return None

    normalized = _normalize_match_text(str(target))
    if not normalized or normalized == "none":
        return None

    for candidate in ALLOWED_INTENT_TARGETS:
        if _normalize_match_text(candidate) == normalized:
            return None if candidate == "none" else candidate

    for evidence_id, candidate in EVIDENCE_INTENT_TARGETS.items():
        if _normalize_match_text(evidence_id) == normalized:
            return candidate

    return None


def _intent_target_evidence_id(target: str | None, dataset: ScenarioDataset) -> str | None:
    """의도 target을 현재 시나리오에 존재하는 실제 단서 ID로 변환한다."""
    if not target:
        return None
    if target in EVIDENCE_INTENT_TARGETS:
        return _first_existing_evidence_id(dataset, [target])
    return _first_existing_evidence_id(dataset, INTENT_TARGET_EVIDENCE.get(target, []))


def _fallback_intent_from_rules(
    question: str,
    dataset: ScenarioDataset,
    state: GameSessionState,
    unlocked_ids: set[str],
) -> IntentClassification:
    """LLM이 없거나 불확실할 때 기존 키워드 규칙으로 최소한의 의도를 잡는다."""
    text = _normalize_match_text(question)
    if _flavor_action_answer(text):
        return IntentClassification(intent="flavor", confidence=0.9, rationale="rule_flavor")
    if _looks_like_related_followup(text):
        return IntentClassification(intent="ask_relation", confidence=0.9, rationale="rule_followup")

    target_id = _action_target_evidence_id(text, state, unlocked_ids, dataset)
    if target_id:
        return IntentClassification(
            intent="inspect",
            target=EVIDENCE_INTENT_TARGETS.get(target_id),
            confidence=0.9,
            rationale="rule_evidence_target",
        )

    if _looks_like_action(text) and (
        _has_any_term(text, STUDY_ACTION_TERMS) or _has_any_term(text, CHAIR_ACTION_TERMS)
    ):
        target = "chair" if _has_any_term(text, CHAIR_ACTION_TERMS) else "scene"
        return IntentClassification(
            intent="broad_scene",
            target=target,
            confidence=0.8,
            rationale="rule_broad_scene",
        )

    return IntentClassification()


def _parse_intent_json(content: str) -> IntentClassification:
    """LLM 응답 JSON을 안전하게 읽고, 허용된 intent/target만 남긴다."""
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.casefold().startswith("json"):
            text = text[4:].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return IntentClassification()

    if not isinstance(payload, dict):
        return IntentClassification()

    intent = str(payload.get("intent", "unknown")).strip()
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"

    target = _normalize_intent_target(payload.get("target"))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = " ".join(str(payload.get("rationale", "") or "").split())
    if len(rationale) > INTENT_MAX_RATIONALE_CHARS:
        rationale = f"{rationale[:INTENT_MAX_RATIONALE_CHARS].rstrip()}..."

    return IntentClassification(
        intent=intent,
        target=target,
        confidence=confidence,
        rationale=rationale,
    )


def _intent_state_context(
    dataset: ScenarioDataset,
    state: GameSessionState,
    unlocked_ids: set[str],
) -> str:
    """LLM에게 현재 열린/잠긴 대상만 알려주는 요약문을 만든다."""
    descriptions = {
        "memo": "테이블 위 메모, 눌러쓴 흔적, 가장 큰 책/우체통/해시계/트로이의 목마 단어",
        "beer": "쓰레기통의 흑맥주 캔, 캔의 브랜드나 이름",
        "record_system": "흑맥주 브랜드가 세계 기록 체계로 이어지는지",
        "book_record": "가장 큰 책 기록",
        "postbox_record": "가장 큰 우체통 기록",
        "sundial_record": "가장 큰 해시계 기록",
        "pattern": "책, 우체통, 해시계가 공유하는 규칙",
        "final_location": "트로이의 목마와 최종 여행지",
        "scene": "서재 전체, 테이블, 쓰레기통 같은 주변 탐색",
        "chair": "안락의자에 앉기, 기대기, 냄새 맡기 같은 분위기 행동",
    }
    lines = []
    for target, description in descriptions.items():
        evidence_id = _intent_target_evidence_id(target, dataset)
        state_label = "not_evidence"
        if evidence_id:
            if evidence_id in state.discovered_evidence_ids:
                state_label = "found"
            elif evidence_id in unlocked_ids:
                state_label = "unlocked"
            else:
                state_label = "locked"
        lines.append(f"- {target}: {state_label}; {description}")
    return "\n".join(lines)


def _classify_input_intent(
    question: str,
    dataset: ScenarioDataset,
    state: GameSessionState,
    unlocked_ids: set[str],
    active_session_id: str,
) -> IntentClassification:
    """사용자 입력을 행동 의도로 분류한다.

    이 함수는 정답을 생성하지 않는다. “무엇을 하려는가”만 JSON으로 받아오고,
    낮은 확신도나 API 실패 시에는 규칙 기반 fallback으로 돌아간다.
    """

    fallback = _fallback_intent_from_rules(question, dataset, state, unlocked_ids)
    if not _intent_classifier_enabled():
        return fallback

    settings = RAGSettings.from_env()
    client = _get_openai_client(settings)
    if client is None:
        return fallback

    chat_history = _recent_chat_history(active_session_id, dataset.scenario_id)
    system_prompt = (
        "너는 MysteryScene 추리 게임의 입력 의도 분류기다. "
        "정답을 추론하거나 사용자에게 답하지 말고, 입력의 의도만 JSON으로 분류한다. "
        "허용 intent는 inspect, ask_relation, flavor, broad_scene, submit_guess, irrelevant, unknown이다. "
        "허용 target은 memo, beer, record_system, book_record, postbox_record, sundial_record, "
        "pattern, final_location, scene, chair, none이다. "
        "잠긴 단서를 열지 말고, 분류만 한다. 확신이 낮으면 confidence를 0.6 미만으로 둔다. "
        "반드시 JSON 객체 하나만 출력한다."
    )
    user_prompt = (
        f"사용자 입력: {question}\n\n"
        f"현재 상태와 대상:\n{_intent_state_context(dataset, state, unlocked_ids)}\n\n"
        f"최근 대화:\n{_build_history_context(chat_history)}\n\n"
        "출력 형식: "
        '{"intent":"inspect","target":"memo","confidence":0.0,"rationale":"short reason"}'
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=120,
        )
        parsed = _parse_intent_json(str(response.choices[0].message.content or ""))
    except Exception:
        return fallback

    if parsed.confidence < INTENT_CONFIDENCE_THRESHOLD and fallback.confidence > parsed.confidence:
        return fallback
    return parsed


def _generic_flavor_action_answer(question: str) -> str:
    """단서와 무관한 자유 행동에 분위기만 살리는 응답을 준다."""
    action = " ".join(str(question or "").split()) or "그 행동"
    return (
        f"{action}. 기록해 두면 꽤 그럴듯한 수사처럼 보이긴 합니다. "
        "하지만 탐정의 장난은 아직 그 행동에 반응하지 않습니다. "
        "테이블의 메모, 쓰레기통의 캔, 그리고 방금 확보한 단서와의 연결을 다시 건드려 보세요."
    )


def _missing_record_names(state: GameSessionState) -> list[str]:
    names = []
    if "E04" not in state.discovered_evidence_ids:
        names.append("가장 큰 책")
    if "E05" not in state.discovered_evidence_ids:
        names.append("가장 큰 우체통")
    if "E06" not in state.discovered_evidence_ids:
        names.append("가장 큰 해시계")
    return names


def _format_previous_steps(steps: list[str]) -> str:
    if len(steps) <= 1:
        return steps[0] if steps else "앞선 단서"
    return ", ".join(steps[:-1]) + f", 그리고 {steps[-1]}"


def _next_guided_input(state: GameSessionState) -> str:
    """난이도 하 데모용으로 다음 입력 예시를 직접 제시한다."""

    # 이 순서가 실제 플레이 가이드의 기준선이다.
    # 사용자가 막히거나 순서를 건너뛰면 아래 문구 중 현재 필요한 것만 안내한다.
    if "E01" not in state.discovered_evidence_ids:
        return "'메모 조사'"
    if "E04" not in state.discovered_evidence_ids:
        return "'가장 큰 책 검색'"
    if "E05" not in state.discovered_evidence_ids:
        return "'가장 큰 우체통 검색'"
    if "E02" not in state.discovered_evidence_ids:
        return "'흑맥주 조사'"
    if "E03" not in state.discovered_evidence_ids:
        return "'기네스 세계기록이랑 관련 있나?'"
    if "E06" not in state.discovered_evidence_ids:
        return "'가장 큰 해시계 검색'"
    if "E07" not in state.discovered_evidence_ids:
        return "'세계에서 가장 큰 트로이의 목마 검색'"
    if "E08" not in state.discovered_evidence_ids:
        return "'국내 트로이의 목마 장소 검색'"
    return "'경기도 여주'"


def _guided_step_answer(state: GameSessionState, prefix: str | None = None) -> str:
    lead = prefix or "아직 그 단서를 해석하기엔 순서가 조금 어긋났습니다."
    return f"{lead}\n\n지금은 {_next_guided_input(state)}라고 입력해보세요."


def _basic_clue_gap_answer(state: GameSessionState) -> str:
    return _guided_step_answer(
        state,
        "탐정이 일부러 순서를 흩뜨려 놓았지만, 지금은 다음 단서부터 잡는 편이 빠릅니다.",
    )


def _locked_action_answer(target_id: str | None, state: GameSessionState) -> str:
    """아직 열리지 않은 단서를 요구했을 때 다음 행동을 안내한다."""

    if target_id == "E03":
        return _guided_step_answer(
            state,
            "기네스 쪽 감은 좋지만, 그 연결을 확정하려면 앞선 단서가 조금 더 필요합니다.",
        )
    if target_id in {"E04", "E05", "E06"} and "E03" not in state.discovered_evidence_ids:
        return _guided_step_answer(
            state,
            "그 검색어도 맞는 방향이지만, 지금 단계에서 먼저 확인해야 할 단서가 있습니다.",
        )
    if target_id == "E07":
        return _guided_step_answer(
            state,
            "트로이의 목마로 가는 길은 거의 보이지만, 아직 마지막 직전의 단서가 남아 있습니다.",
        )
    if target_id == "E08":
        return _guided_step_answer(
            state,
            "최종 목적지는 아직 바로 말하기엔 이릅니다.",
        )
    return _basic_clue_gap_answer(state)


def _action_hint_answer(text: str) -> str:
    if _has_any_term(text, CHAIR_ACTION_TERMS):
        return (
            "안락의자에는 탐정의 흔적이 남아 있지만, 지금 바로 단서로 확정할 만한 것은 보이지 않습니다. "
            "시선은 사이드 테이블과 쓰레기통 쪽으로 더 자연스럽게 갑니다."
        )
    return (
        "서재를 둘러보니 안락의자와 사이드 테이블, 쓰레기통이 눈에 들어옵니다. "
        "테이블 위의 메모를 확인하거나 쓰레기통을 뒤져볼 수 있습니다."
    )


def _clue_display_text(item) -> str:
    template = CLUE_DISPLAY_TEMPLATES.get(getattr(item, "id", ""))
    if template:
        return template

    summary = str(getattr(item, "summary", "") or "").strip()
    if summary:
        return summary

    title = str(getattr(item, "title", "단서") or "단서").strip()
    return f"{title} 단서를 확보했습니다. 이제 이 단서가 앞뒤 맥락에서 어떤 역할을 하는지 따져보세요."


def _action_evidence_answer(item, was_already_found: bool, state: GameSessionState) -> str:
    """행동 입력으로 단서를 발견했을 때 채팅창에 보여줄 문구를 만든다."""

    if was_already_found:
        # 최종 단서는 반복 입력 시에도 사건 종결 느낌이 나도록 별도 문구를 준다.
        if getattr(item, "id", "") == "E08":
            return (
                "모든 단서가 하나로 이어진다.\n\n"
                "탐정은 '세계에서 가장 큰 것들'을 따라 이동했고, 마지막 목적지는 "
                "트로이의 목마와 연결된 장소였다.\n\n"
                "당신은 탐정이 남긴 마지막 행선지를 찾아냈다."
            )
        return _guided_step_answer(
            state,
            "이미 확인한 단서입니다. 같은 곳을 다시 보기보다 다음 단서로 넘어가면 됩니다.",
        )

    # 처음 발견한 단서는 직접 가이드 없이 조사 기록만 보여준다.
    template = CLUE_ANSWER_TEMPLATES.get(getattr(item, "id", ""))
    if template:
        return template

    return _clue_display_text(item)


def _action_response(
    *,
    answer: str,
    status: str,
    dataset: ScenarioDataset,
    session_id: str,
    state: GameSessionState,
    allowed_level: int,
    next_allowed_level: int,
    evidence: list[dict] | None = None,
    score: float | None = None,
) -> dict:
    response = {
        "answer": answer,
        "status": status,
        "input_type": "action",
        "level": evidence[0]["level"] if evidence else allowed_level,
        "allowed_level": allowed_level,
        "next_allowed_level": next_allowed_level,
        "scenario_id": dataset.scenario_id,
        "session_id": session_id,
        "question_count": state.question_count,
        "evidence": evidence or [],
    }
    if score is not None:
        response["score"] = round(score, 4)
    return response


def _related_followup_answer(item, state: GameSessionState) -> str:
    evidence_id = getattr(item, "id", "")
    if evidence_id == "E01":
        return (
            "그 메모는 혼자 잘난 척하는 종이가 아닙니다. "
            "쓰레기통의 흑맥주 캔과 만나야 '가장 큰'이라는 말이 기록 쪽으로 고개를 돌립니다. "
            "종이와 캔을 한 줄에 세워 보세요."
        )
    if evidence_id == "E02":
        return (
            "그 캔은 술자리보다 기록 보관소에 더 어울리는 얼굴입니다. "
            "메모의 '가장 큰'이라는 말과 연결하면, 브랜드 이름이 기록 체계를 끌고 옵니다. "
            "이제 맥주가 아니라 그 이름이 붙은 기록을 의심해 보세요."
        )
    if evidence_id == "E03":
        return (
            "그 기록 체계는 메모의 단어들을 차례로 끌고 가는 끈입니다. "
            "책, 우체통, 해시계를 각각 같은 기준으로 재보라는 뜻에 가깝습니다. "
            "셋이 같은 모양으로 서는지 확인해 보세요."
        )
    if evidence_id == "E04":
        missing = _format_previous_steps(_missing_record_names(state))
        return (
            "그 책은 메모와 기네스 기록 사이에 놓인 첫 번째 표본입니다. "
            f"아직 {missing} 쪽도 같은 장난을 치는지 확인해야 합니다. "
            "책 하나만 크다고 우기면 탐정이 너무 쉽게 웃습니다."
        )
    if evidence_id == "E05":
        missing = _format_previous_steps(_missing_record_names(state))
        return (
            "그 우체통도 메모와 기록 체계가 같은 편이라는 증거입니다. "
            f"아직 {missing} 기록이 남아 있다면 그쪽도 맞춰 보세요. "
            "편지는 안 보내도, 규칙은 꽤 또렷하게 배달됩니다."
        )
    if evidence_id == "E06":
        missing = _format_previous_steps(_missing_record_names(state))
        return (
            "그 해시계는 앞선 기록들과 같은 방향을 가리키는 물건입니다. "
            f"아직 {missing} 기록이 비어 있다면 먼저 채워야 합니다. "
            "시간보다 중요한 건 세 물건이 같은 기준으로 묶인다는 점입니다."
        )
    if evidence_id == "E07":
        return (
            "그 공통 패턴은 마지막 단어로 가는 다리입니다. "
            "책, 우체통, 해시계가 같은 규칙을 증명했으니, 이제 트로이의 목마도 같은 방식으로 의심하면 됩니다. "
            "말의 이야기가 아니라 말의 덩치를 보세요."
        )
    if evidence_id == "E08":
        return (
            "그 마지막 단서는 앞선 기록들이 만든 줄의 끝입니다. "
            "목적지가 왜 나왔는지 다시 보려면 책, 우체통, 해시계가 만든 같은 규칙을 거꾸로 따라가면 됩니다. "
            "탐정은 끝까지 같은 농담을 반복한 셈입니다."
        )

    return (
        "그 단서는 앞뒤 단서와 따로 놀지 않습니다. "
        "방금 확인한 물건이 메모, 흑맥주, 기록 체계 중 어디에 붙는지 다시 놓아 보세요."
    )


def _process_followup_input(
    question: str,
    dataset: ScenarioDataset,
    state: GameSessionState,
    active_session_id: str,
    allowed_level: int,
    intent: IntentClassification | None = None,
) -> dict | None:
    """“그건 뭐랑 연결돼?” 같은 후속 질문을 최근 단서 맥락에 붙인다."""
    text = _normalize_match_text(question)
    uses_intent = _intent_is_confident(intent) and intent.intent == "ask_relation"
    if not uses_intent and not _looks_like_related_followup(text):
        return None

    chat_history = _recent_chat_history(active_session_id, dataset.scenario_id)
    evidence_id = None
    if uses_intent:
        evidence_id = _intent_target_evidence_id(intent.target, dataset)
        if evidence_id not in state.discovered_evidence_ids:
            evidence_id = None
    evidence_id = evidence_id or _latest_context_evidence_id(dataset, chat_history)
    if not evidence_id:
        return None

    item = _evidence_by_id(dataset).get(evidence_id)
    if not item:
        return None

    return _action_response(
        answer=_related_followup_answer(item, state),
        status="context_hint",
        dataset=dataset,
        session_id=active_session_id,
        state=state,
        allowed_level=allowed_level,
        next_allowed_level=allowed_level,
    )


def _process_action_input(
    question: str,
    dataset: ScenarioDataset,
    state: GameSessionState,
    active_session_id: str,
    unlocked_before: set[str],
    allowed_level: int,
    intent: IntentClassification | None = None,
) -> dict | None:
    """행동형 입력을 처리한다.

    처리 우선순위:
    1) 엉뚱한 자유 행동은 flavor_action으로 받아준다.
    2) 명확한 키워드 매핑은 LLM보다 먼저 적용한다.
    3) 키워드로 못 잡은 자연어만 LLM 의도 분류 결과를 사용한다.
    4) 잠긴 단서는 locked_clue로 막고 다음 행동을 안내한다.
    5) 열린 단서만 discovered_evidence_ids에 추가한다.
    """

    text = _normalize_match_text(question)
    confident_intent = intent if _intent_is_confident(intent) else None

    flavor_answer = _flavor_action_answer(text)
    if not flavor_answer and confident_intent and confident_intent.intent == "flavor":
        flavor_answer = _generic_flavor_action_answer(question)
    if flavor_answer:
        return _action_response(
            answer=flavor_answer,
            status="flavor_action",
            dataset=dataset,
            session_id=active_session_id,
            state=state,
            allowed_level=allowed_level,
            next_allowed_level=allowed_level,
        )

    # 팀원분이 제공한 키워드 사전 기반 라우팅이다.
    # "가장 큰 해시계 검색"처럼 명확한 입력은 여기서 먼저 단서 ID가 결정된다.
    rule_target_id = _action_target_evidence_id(text, state, unlocked_before, dataset)

    target_id = rule_target_id
    if target_id is None and confident_intent and confident_intent.intent == "ask_relation":
        # "이거랑 관련 있어?" 같은 관계 질문이 새 단서를 가리키면 단서 발견으로 처리한다.
        # 이미 발견한 단서라면 후속 질문 처리(_process_followup_input)로 넘긴다.
        relation_target_id = _intent_target_evidence_id(confident_intent.target, dataset)
        if relation_target_id and relation_target_id not in state.discovered_evidence_ids:
            target_id = relation_target_id
        else:
            return None

    if target_id is None and confident_intent and confident_intent.intent in {"inspect", "submit_guess"}:
        # 키워드로 못 잡은 자연어 행동만 LLM target을 사용한다.
        # 그래도 아래 unlock 검사에서 잠긴 단서 공개는 막힌다.
        target = confident_intent.target
        if confident_intent.intent == "submit_guess" and target is None:
            target = "final_location"
        target_id = _intent_target_evidence_id(target, dataset)

    final_id = _first_existing_evidence_id(dataset, ["E08", "E03"])
    pattern_id = _first_existing_evidence_id(dataset, ["E07"])
    if target_id == final_id and _should_route_trojan_query_to_pattern(text, state, dataset):
        # LLM이 트로이 목마 검색을 최종 목적지로 오분류하는 경우를 보정한다.
        target_id = pattern_id

    if target_id is None:
        if confident_intent and confident_intent.intent == "broad_scene":
            return _action_response(
                answer=_action_hint_answer(text),
                status="action_hint",
                dataset=dataset,
                session_id=active_session_id,
                state=state,
                allowed_level=allowed_level,
                next_allowed_level=allowed_level,
            )
        if not _looks_like_action(text) or not (
            _has_any_term(text, STUDY_ACTION_TERMS) or _has_any_term(text, CHAIR_ACTION_TERMS)
        ):
            return None
        return _action_response(
            answer=_action_hint_answer(text),
            status="action_hint",
            dataset=dataset,
            session_id=active_session_id,
            state=state,
            allowed_level=allowed_level,
            next_allowed_level=allowed_level,
        )

    evidence = _evidence_by_id(dataset)
    item = evidence.get(target_id)
    if item is None:
        return None
    if target_id not in unlocked_before or getattr(item, "visibility", "") == "hidden":
        # 여기서 최종 안전장치를 건다.
        # 라우팅이 맞아도 unlock_rules상 아직 열리지 않았으면 단서는 공개하지 않는다.
        return _action_response(
            answer=_locked_action_answer(target_id, state),
            status="locked_clue",
            dataset=dataset,
            session_id=active_session_id,
            state=state,
            allowed_level=allowed_level,
            next_allowed_level=allowed_level,
        )

    was_already_found = target_id in state.discovered_evidence_ids
    state.discovered_evidence_ids.add(target_id)
    # 단서를 하나 추가한 뒤 unlock_rules를 다시 계산해 다음 단계 레벨을 갱신한다.
    unlocked_after = _unlocked_evidence_ids(dataset, state)
    next_allowed_level = _highest_unlocked_level(dataset, unlocked_after)
    formatted_evidence = [_format_evidence_item(item)]
    return _action_response(
        answer=_action_evidence_answer(item, was_already_found, state),
        status="success",
        dataset=dataset,
        session_id=active_session_id,
        state=state,
        allowed_level=allowed_level,
        next_allowed_level=next_allowed_level,
        evidence=formatted_evidence,
        score=1.0,
    )


def _build_prompt_context(evidence: list[dict]) -> str:
    lines: list[str] = []
    for item in evidence:
        lines.append(
            (
                f"[{item['id']}] level={item['level']} "
                f"category={item['category']}\n{item['content']}"
            )
        )
    return "\n\n".join(lines)


def _trim_history_content(content: str) -> str:
    text = " ".join(str(content or "").split())
    if len(text) <= CHAT_HISTORY_MAX_CHARS:
        return text
    return f"{text[:CHAT_HISTORY_MAX_CHARS].rstrip()}..."


def _build_history_context(chat_history: list[dict] | None) -> str:
    if not chat_history:
        return "이전 대화 없음"

    lines: list[str] = []
    for item in chat_history[-CHAT_HISTORY_LIMIT:]:
        role = "사용자" if item.get("role") == "user" else "조사관"
        status = item.get("status")
        evidence_ids = item.get("evidence_ids") or []
        meta = []
        if status:
            meta.append(f"status={status}")
        if evidence_ids:
            meta.append(f"evidence={','.join(evidence_ids)}")
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"- {role}{suffix}: {_trim_history_content(str(item.get('content', '')))}")
    return "\n".join(lines)


def _fallback_answer(best_content: str) -> str:
    content = str(best_content or "")
    if "트로이" in content and ("여주" in content or "최종" in content):
        return (
            "마지막 말은 생각보다 말이 많군요. "
            "같은 규칙을 끝까지 밀어붙이면 목적지는 거의 모습을 드러냅니다. "
            "바로 쓰기 전에, 왜 그 장소가 앞의 기록들과 같은 줄에 서는지 확인해 보세요."
        )
    if "기네스" in content or "세계 기록" in content:
        return (
            "단서는 손에 들어왔습니다. "
            "다만 친절하게 결론까지 말해줄 생각은 없어 보이네요. "
            "술, 기록, 그리고 '가장 큰'이라는 허세가 같은 방향을 가리키는지 보세요."
        )
    if "가장 큰" in content:
        return (
            "단서는 손에 들어왔습니다. "
            "문장 속에서 유난히 큰 척하는 표현이 하나 있습니다. "
            "그 말을 다른 대상들에도 똑같이 붙이면 어떤 목록이 되는지 생각해 보세요."
        )
    return (
        "단서는 손에 들어왔습니다. "
        "탐정은 역시 친절한 척만 하고, 결론은 플레이어에게 떠넘겼네요. "
        "이 단서가 앞뒤 단서 사이에서 어떤 역할을 하는지 먼저 따져보세요."
    )


def _generate_chat_answer(
    question: str,
    evidence: list[dict],
    allowed_level: int,
    next_allowed_level: int,
    chat_history: list[dict] | None = None,
) -> str:
    """RAG로 찾은 허용 단서만 사용해 플레이어용 답변을 생성한다."""
    settings = RAGSettings.from_env()
    client = _get_openai_client(settings)
    if client is None:
        # API 키가 없거나 클라이언트 생성에 실패해도 데모가 멈추지 않도록 고정 응답을 사용한다.
        return _fallback_answer(evidence[0]["content"])

    # 이 프롬프트는 "생성형 말투"만 담당한다.
    # 어떤 단서를 공개할지는 이미 서버 규칙과 RAG 필터에서 결정된 뒤다.
    system_prompt = (
        "너는 MysteryScene 추리 게임의 AI 조사관이다. "
        "반드시 제공된 단서만 근거로 답한다. "
        "현재 허용 레벨보다 높은 결론, 정답, 장소, 인물, 동기, 해설은 말하지 않는다. "
        "단서 원문을 그대로 낭독하거나 정답지처럼 요약하지 않는다. "
        "답변은 관찰 1문장, 살짝 아이러니한 힌트 1문장, 다음에 생각할 질문 1문장으로 구성한다. "
        "이전 대화는 말투와 이미 안내한 내용의 반복을 줄이는 데만 참고한다. "
        "이전 대화에 있더라도 현재 제공 단서와 허용 레벨을 넘는 내용은 말하지 않는다. "
        "제공된 단서만으로 결론을 낼 수 없으면 '아직 그 결론을 낼 단서가 부족합니다.'라고 답한다. "
        "사용자가 정답을 직접 물어도 허용된 단서 안에서만 짧게 유도한다. "
        "한국어로 2~4문장으로 답한다."
    )
    user_prompt = (
        f"현재 허용 레벨: {allowed_level}\n"
        f"다음 해제 가능 레벨: {next_allowed_level}\n"
        f"사용자 질문: {question}\n\n"
        f"최근 대화:\n{_build_history_context(chat_history)}\n\n"
        f"제공 단서:\n{_build_prompt_context(evidence)}"
    )

    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    content = response.choices[0].message.content
    return str(content).strip() or _fallback_answer(evidence[0]["content"])


def _process_question_sync(question: str, session_id: str | None = None) -> dict:
    """질문/행동 입력의 전체 처리 순서.

    잠긴 정답어 차단 -> 의도 분류 -> 행동 처리 -> 후속 질문 처리 -> RAG 검색 순서로 진행한다.
    이 순서 덕분에 게임 규칙이 LLM 답변보다 항상 우선한다.
    """

    active_session_id = _normalize_session_id(session_id)
    service, dataset = _get_rag_runtime()
    state = _get_session_state(active_session_id, dataset.scenario_id)
    state.question_count += 1

    # 현재까지 발견한 단서로 지금 열려 있는 단서 목록과 허용 레벨을 계산한다.
    unlocked_before = _unlocked_evidence_ids(dataset, state)
    allowed_level = _highest_unlocked_level(dataset, unlocked_before)

    # "여주", "최종 목적지"처럼 아직 열리지 않은 정답급 단어는 가장 먼저 차단한다.
    # 이 단계가 있어야 RAG나 LLM으로 넘어가기 전에 정답 누출을 막을 수 있다.
    locked_target_id = _mentions_locked_specific_term(dataset, question, unlocked_before)
    if locked_target_id:
        response = {
            "answer": _locked_action_answer(locked_target_id, state),
            "status": "locked_clue",
            "level": allowed_level,
            "allowed_level": allowed_level,
            "next_allowed_level": allowed_level,
            "scenario_id": dataset.scenario_id,
            "session_id": active_session_id,
            "question_count": state.question_count,
            "evidence": [],
        }
        _save_session_state(active_session_id, dataset.scenario_id, state)
        _record_question_exchange(active_session_id, dataset.scenario_id, question, response)
        return response

    # LLM 의도 분류는 행동/질문 의도를 보조 판단한다.
    # 실제 단서 공개 여부는 뒤의 _process_action_input에서 다시 검사한다.
    intent = _classify_input_intent(
        question,
        dataset,
        state,
        unlocked_before,
        active_session_id,
    )

    # 행동형 입력이 먼저 처리된다.
    # 예: "메모 조사", "가장 큰 책 검색", "흑맥주 조사"
    action_response = _process_action_input(
        question,
        dataset,
        state,
        active_session_id,
        unlocked_before,
        allowed_level,
        intent,
    )
    if action_response is not None:
        _save_session_state(active_session_id, dataset.scenario_id, state)
        _record_question_exchange(active_session_id, dataset.scenario_id, question, action_response)
        return action_response

    # 행동이 아니라 "그건 뭐랑 관련 있어?" 같은 후속 질문이면 최근 단서에 붙인다.
    followup_response = _process_followup_input(
        question,
        dataset,
        state,
        active_session_id,
        allowed_level,
        intent,
    )
    if followup_response is not None:
        _save_session_state(active_session_id, dataset.scenario_id, state)
        _record_question_exchange(active_session_id, dataset.scenario_id, question, followup_response)
        return followup_response

    # 위 규칙으로 잡히지 않은 일반 질문은 RAG 검색으로 처리한다.
    # 검색도 현재 허용 레벨 이하, 그리고 열린 단서 안에서만 수행한다.
    raw_results = service.search(
        query=question,
        scenario_id=dataset.scenario_id,
        top_k=SEARCH_POOL_SIZE,
        level_lte=allowed_level,
    )
    results = [
        result
        for result in raw_results
        if result.chunk.metadata.source_id in unlocked_before
        and _result_matches_question(dataset, result, question)
    ][:RESPONSE_TOP_K]

    if not results:
        # 단서와 연결되지 않은 입력은 막혔다고 보고 현재 다음 행동을 직접 알려준다.
        response = {
            "answer": (
                f"'{question}'에 대한 단서를 찾지 못했습니다.\n\n"
                f"지금은 {_next_guided_input(state)}라고 입력해보세요."
            ),
            "status": "no_clue",
            "level": allowed_level,
            "allowed_level": allowed_level,
            "next_allowed_level": allowed_level,
            "scenario_id": dataset.scenario_id,
            "session_id": active_session_id,
            "question_count": state.question_count,
            "evidence": [],
        }
        _save_session_state(active_session_id, dataset.scenario_id, state)
        _record_question_exchange(active_session_id, dataset.scenario_id, question, response)
        return response

    best = results[0]
    answer_context = [_format_evidence(result) for result in results]
    evidence = [_format_evidence_result(result, dataset) for result in results]
    # RAG로 찾은 단서도 열린 단서에 한해서만 발견 처리한다.
    _discover_results(state, results)
    unlocked_after = _unlocked_evidence_ids(dataset, state)
    next_allowed_level = _highest_unlocked_level(dataset, unlocked_after)
    chat_history = _recent_chat_history(active_session_id, dataset.scenario_id)
    answer = _generate_chat_answer(
        question=question,
        evidence=answer_context,
        allowed_level=allowed_level,
        next_allowed_level=next_allowed_level,
        chat_history=chat_history,
    )

    response = {
        "answer": answer,
        "status": "success",
        "level": best.chunk.metadata.level,
        "allowed_level": allowed_level,
        "next_allowed_level": next_allowed_level,
        "scenario_id": dataset.scenario_id,
        "session_id": active_session_id,
        "question_count": state.question_count,
        "score": round(best.score, 4),
        "evidence": evidence,
    }
    _save_session_state(active_session_id, dataset.scenario_id, state)
    _record_question_exchange(active_session_id, dataset.scenario_id, question, response)
    return response


async def process_question(question: str, session_id: str | None = None):
    try:
        return await asyncio.to_thread(_process_question_sync, question, session_id)
    except Exception as exc:
        return {
            "answer": f"컨트롤러 오류: {str(exc)}",
            "status": "error",
            "level": 1,
            "allowed_level": 1,
            "next_allowed_level": 1,
            "session_id": _normalize_session_id(session_id),
            "evidence": [],
        }


def _submit_solution_sync(
    answer: str,
    session_id: str | None = None,
    culprit: str | None = None,
    motive: str | None = None,
    method: str | None = None,
) -> dict:
    active_session_id = _normalize_session_id(session_id)
    _, dataset = _get_rag_runtime()
    state = _get_session_state(active_session_id, dataset.scenario_id)
    state.submission_count += 1

    required_ids = _required_evidence_ids(dataset)
    missing_ids = sorted(required_ids - state.discovered_evidence_ids)
    if missing_ids:
        response = {
            "answer": "아직 사건을 종결하기에는 확인하지 않은 핵심 단서가 있습니다.",
            "status": "needs_more_evidence",
            "solved": False,
            "scenario_id": dataset.scenario_id,
            "session_id": active_session_id,
            "submission_count": state.submission_count,
            "missing_evidence_count": len(missing_ids),
        }
        _save_session_state(active_session_id, dataset.scenario_id, state)
        _record_submission(active_session_id, dataset.scenario_id, answer, response)
        return response

    submission = _submission_text(answer, culprit, motive, method)
    if not _submission_matches_solution(dataset, submission):
        response = {
            "answer": "아직 사건 종결 보고서로 채택하기 어렵습니다. 확보한 단서와 맞지 않는 부분이 있습니다.",
            "status": "incorrect",
            "solved": False,
            "scenario_id": dataset.scenario_id,
            "session_id": active_session_id,
            "submission_count": state.submission_count,
            "missing_evidence_count": 0,
        }
        _save_session_state(active_session_id, dataset.scenario_id, state)
        _record_submission(active_session_id, dataset.scenario_id, answer, response)
        return response

    state.solved = True
    solution = dataset.solution
    clear_rank = _clear_rank(dataset, state)
    response = {
        "answer": f"사건 종결. {solution.explanation}\n\n클리어 랭크: {clear_rank['label']} - {clear_rank['description']}",
        "status": "solved",
        "solved": True,
        "scenario_id": dataset.scenario_id,
        "session_id": active_session_id,
        "question_count": state.question_count,
        "submission_count": state.submission_count,
        "missing_evidence_count": 0,
        "clear_rank": clear_rank,
        "solution": {
            "culprit_character_id": solution.culprit_character_id,
            "motive": solution.motive,
            "method": solution.method,
            "explanation": solution.explanation,
            "decisive_evidence_ids": solution.decisive_evidence_ids,
        },
    }
    _save_session_state(active_session_id, dataset.scenario_id, state)
    _record_submission(active_session_id, dataset.scenario_id, answer, response)
    return response


async def submit_solution(
    answer: str,
    session_id: str | None = None,
    culprit: str | None = None,
    motive: str | None = None,
    method: str | None = None,
):
    try:
        return await asyncio.to_thread(
            _submit_solution_sync,
            answer,
            session_id,
            culprit,
            motive,
            method,
        )
    except Exception as exc:
        return {
            "answer": f"컨트롤러 오류: {str(exc)}",
            "status": "error",
            "solved": False,
            "session_id": _normalize_session_id(session_id),
        }


async def get_case_start(session_id: str | None = None):
    try:
        return await asyncio.to_thread(_case_start_sync, session_id)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"컨트롤러 오류: {str(exc)}",
            "session_id": _normalize_session_id(session_id),
        }


async def get_progress(session_id: str | None = None):
    try:
        return await asyncio.to_thread(_progress_sync, session_id)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"컨트롤러 오류: {str(exc)}",
            "session_id": _normalize_session_id(session_id),
        }
