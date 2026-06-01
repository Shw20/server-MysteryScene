import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

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
GENERIC_LOCKED_TERMS = {
    "final",
    "finallocation",
    "solution",
    "결론",
    "단서",
    "여행지",
    "정답",
    "최종",
}
ACTION_VERB_TERMS = {
    "간다",
    "뒤져",
    "뒤진",
    "둘러",
    "본다",
    "보다",
    "살펴",
    "열어",
    "이동",
    "조사",
    "찾아",
    "찾는",
    "탐색",
    "확인",
}
STUDY_ACTION_TERMS = {"서재", "현장", "주변"}
CHAIR_ACTION_TERMS = {"안락의자", "의자"}
MEMO_ACTION_TERMS = {
    "글자",
    "눌러쓴",
    "눌러",
    "메모",
    "메모지",
    "사이드테이블",
    "용지",
    "자국",
    "종이",
    "테이블",
}
BEER_ACTION_TERMS = {
    "맥주",
    "맥주캔",
    "쓰레기통",
    "캔",
    "휴지통",
    "흑맥주",
}
RECORD_ACTION_TERMS = {"guinness", "기네스", "기네스북", "브랜드", "세계기록", "기록"}
BOOK_ACTION_TERMS = {"가장큰책", "큰책", "책"}
POSTBOX_ACTION_TERMS = {"가장큰우체통", "큰우체통", "우체통"}
SUNDIAL_ACTION_TERMS = {"가장큰해시계", "큰해시계", "해시계"}
PATTERN_ACTION_TERMS = {"공통", "공통점", "규칙", "나머지", "패턴"}
FINAL_ACTION_TERMS = {
    "마지막여행지",
    "목마",
    "최종목적지",
    "트로이",
    "트로이목마",
    "트로이의목마",
}
FOLLOWUP_REFERENCE_TERMS = {
    "그건",
    "그거",
    "그단서",
    "그럼그건",
    "방금",
    "앞서",
    "이건",
    "이거",
    "이단서",
}
FOLLOWUP_RELATION_TERMS = {
    "관계",
    "관련",
    "뭐랑",
    "무엇과",
    "연결",
    "연관",
    "이어",
}
DRINK_ACTION_TERMS = {"들이켜", "마셔", "마신", "마신다", "맛봐", "맛본", "먹어", "먹는다"}
SIT_ACTION_TERMS = {"걸터", "눕", "앉"}
CALL_ACTION_TERMS = {"전화", "연락", "문자", "카톡", "부른", "불러"}
LEAVE_ACTION_TERMS = {"나가", "나간", "나간다", "도망", "떠나", "밖으로"}
DAMAGE_ACTION_TERMS = {
    "깨",
    "던져",
    "망가",
    "부순",
    "부숴",
    "불",
    "찢",
    "태워",
    "파괴",
    "훼손",
}
ODD_OBJECT_ACTION_TERMS = {"창문", "커튼", "램프", "서랍", "바닥", "문고리"}
CLUE_ANSWER_TEMPLATES = {
    "E01": (
        "종이는 꽤 친절한 척하지만, 사실 가장 중요한 말을 슬쩍 빼먹었습니다. "
        "네 단어 앞에 같은 수식어를 붙여 보면 탐정의 장난이 조금 덜 얄미워질 겁니다."
    ),
    "E02": (
        "쓰레기통치고는 꽤 품위 있는 물건을 품고 있군요. "
        "술 이름보다, 그 이름이 붙은 '기록' 쪽이 더 수상합니다."
    ),
    "E03": (
        "맥주가 목을 축이러 온 게 아니었네요. "
        "그 브랜드가 자꾸 기록을 들먹이는 순간, 메모의 '가장 큰'도 그냥 큰소리가 아니게 됩니다."
    ),
    "E04": (
        "책이 꼭 읽히려고만 존재하는 건 아닌 모양입니다. "
        "이번엔 내용보다 크기라는 뻔뻔한 자랑을 먼저 의심해 보세요."
    ),
    "E05": (
        "우체통이 편지를 받으려고 커졌다면 너무 성실한 이야기겠죠. "
        "이 단어도 기록 목록의 한 칸처럼 굴고 있습니다."
    ),
    "E06": (
        "시간을 알려주는 물건이 이번엔 방향을 알려주는군요. "
        "책, 우체통과 같은 방식으로 '가장 큰'이라는 꼬리표를 달고 있습니다."
    ),
    "E07": (
        "앞의 셋이 같은 농담을 반복했습니다. "
        "그러니 남은 트로이의 목마도 말보다 덩치를 먼저 의심하는 편이 낫습니다."
    ),
    "E08": (
        "드디어 말이 목적지를 흘리기 시작합니다. "
        "같은 규칙을 마지막 단어에 적용하면, 지도는 여주 쪽으로 고개를 돌립니다. "
        "이제 왜 그곳이어야 하는지 앞의 세 기록과 맞춰보세요."
    ),
}
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
    return target_id


def _action_target_evidence_id(
    text: str,
    state: GameSessionState,
    unlocked_ids: set[str],
    dataset: ScenarioDataset,
) -> str | None:
    record_id = _first_existing_evidence_id(dataset, ["E03"])
    book_id = _first_existing_evidence_id(dataset, ["E04"])
    postbox_id = _first_existing_evidence_id(dataset, ["E05"])
    sundial_id = _first_existing_evidence_id(dataset, ["E06"])
    pattern_id = _first_existing_evidence_id(dataset, ["E07"])
    final_id = _first_existing_evidence_id(dataset, ["E08", "E03"])

    if _has_any_term(text, BOOK_ACTION_TERMS):
        return _target_or_intro(book_id, state, unlocked_ids)
    if _has_any_term(text, POSTBOX_ACTION_TERMS):
        return _target_or_intro(postbox_id, state, unlocked_ids)
    if _has_any_term(text, SUNDIAL_ACTION_TERMS):
        return _target_or_intro(sundial_id, state, unlocked_ids)
    if _has_any_term(text, PATTERN_ACTION_TERMS):
        return _target_or_intro(pattern_id, state, unlocked_ids)
    if _has_any_term(text, BEER_ACTION_TERMS):
        return "E02"
    if _has_any_term(text, MEMO_ACTION_TERMS):
        return "E01"
    if _has_any_term(text, FINAL_ACTION_TERMS):
        return _target_or_intro(final_id, state, unlocked_ids)
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


def _basic_clue_gap_answer(state: GameSessionState) -> str:
    missing_steps = []
    if "E01" not in state.discovered_evidence_ids:
        missing_steps.append("테이블 위 메모")
    if "E02" not in state.discovered_evidence_ids:
        missing_steps.append("쓰레기통의 흑맥주 캔")

    if missing_steps:
        return (
            "탐정이 계단을 만들어놨는데, 벌써 난간을 타고 올라가려는 셈입니다. "
            f"먼저 {_format_previous_steps(missing_steps)}부터 확인해 보세요. "
            "그 둘이 서로를 가리키기 시작해야 다음 장난이 열립니다."
        )

    return (
        "초반 단서는 챙겼지만, 아직 결론으로 가기에는 탐정의 장난이 한 겹 남았습니다. "
        "흑맥주가 어떤 기록 체계로 이어지는지 먼저 확인해 보세요."
    )


def _locked_action_answer(target_id: str | None, state: GameSessionState) -> str:
    if target_id == "E03":
        return _basic_clue_gap_answer(state)
    if target_id in {"E04", "E05", "E06"} and "E03" not in state.discovered_evidence_ids:
        if "E01" not in state.discovered_evidence_ids or "E02" not in state.discovered_evidence_ids:
            return _basic_clue_gap_answer(state)
        return (
            "개별 기록으로 바로 달려가면 탐정만 흐뭇해질 겁니다. "
            "먼저 메모와 흑맥주가 같은 '기록' 체계 안에서 만나는지 확인해 보세요. "
            "그 다음에야 책, 우체통, 해시계가 얌전히 줄을 섭니다."
        )
    if target_id == "E07":
        if "E03" not in state.discovered_evidence_ids:
            if "E01" not in state.discovered_evidence_ids or "E02" not in state.discovered_evidence_ids:
                return _basic_clue_gap_answer(state)
            return (
                "공통점을 찾겠다는 마음은 훌륭한데, 아직 공통으로 묶을 끈이 없습니다. "
                "먼저 흑맥주가 어떤 기록 체계로 이어지는지 확인해 보세요. "
                "그 끈이 생겨야 책, 우체통, 해시계가 한 줄에 섭니다."
            )
        missing_records = _missing_record_names(state)
        if missing_records:
            return (
                "공통점을 먼저 말하겠다는 건, 답안지 맨 아래부터 읽겠다는 꽤 탐정스러운 반칙입니다. "
                f"아직 {_format_previous_steps(missing_records)} 기록을 확인해야 합니다. "
                "셋이 같은 표정을 지을 때까지 조금만 더 괴롭혀 보세요."
            )
        return (
            "재료는 모였는데 아직 비빔은 안 된 상태입니다. "
            "책, 우체통, 해시계가 같은 방식으로 묶이는지 먼저 정리해 보세요."
        )
    if target_id == "E08":
        if "E03" not in state.discovered_evidence_ids:
            if "E01" not in state.discovered_evidence_ids or "E02" not in state.discovered_evidence_ids:
                return _basic_clue_gap_answer(state)
            return (
                "최종 목적지로 바로 뛰면 여행이 아니라 순간이동이죠. "
                "먼저 흑맥주가 어떤 기록 체계로 이어지는지 확인해 보세요. "
                "말은 그 다음에 꺼내도 늦지 않습니다."
            )
        missing_records = _missing_record_names(state)
        if missing_records:
            return (
                "트로이의 목마가 벌써 나서면 앞의 단서들이 체면을 잃습니다. "
                f"먼저 {_format_previous_steps(missing_records)} 기록을 확인해 보세요. "
                "앞의 셋이 같은 규칙을 증명해야 마지막 말도 움직입니다."
            )
        if "E07" not in state.discovered_evidence_ids:
            return (
                "목적지는 아직 입술만 달싹이는 중입니다. "
                "책, 우체통, 해시계가 만든 공통 패턴을 먼저 정리해 보세요. "
                "그 규칙이 마지막 단어를 데리고 갈 겁니다."
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


def _action_evidence_answer(item, was_already_found: bool) -> str:
    prefix = "이미 확인한 단서입니다. " if was_already_found else ""
    template = CLUE_ANSWER_TEMPLATES.get(getattr(item, "id", ""))
    if template:
        return f"{prefix}{template}"

    return f"{prefix}{_clue_display_text(item)}"


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

    1) 엉뚱한 자유 행동은 flavor_action으로 받아준다.
    2) LLM/규칙이 가리킨 단서가 잠겨 있으면 locked_clue로 막는다.
    3) 열린 단서만 discovered_evidence_ids에 추가한다.
    """

    text = _normalize_match_text(question)
    confident_intent = intent if _intent_is_confident(intent) else None
    if confident_intent and confident_intent.intent == "ask_relation":
        return None

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

    target_id = None
    if confident_intent and confident_intent.intent in {"inspect", "submit_guess"}:
        target = confident_intent.target
        if confident_intent.intent == "submit_guess" and target is None:
            target = "final_location"
        target_id = _intent_target_evidence_id(target, dataset)

    if target_id is None:
        target_id = _action_target_evidence_id(text, state, unlocked_before, dataset)
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
    unlocked_after = _unlocked_evidence_ids(dataset, state)
    next_allowed_level = _highest_unlocked_level(dataset, unlocked_after)
    formatted_evidence = [_format_evidence_item(item)]
    return _action_response(
        answer=_action_evidence_answer(item, was_already_found),
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
        return _fallback_answer(evidence[0]["content"])

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

    unlocked_before = _unlocked_evidence_ids(dataset, state)
    allowed_level = _highest_unlocked_level(dataset, unlocked_before)
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

    intent = _classify_input_intent(
        question,
        dataset,
        state,
        unlocked_before,
        active_session_id,
    )

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
        response = {
            "answer": f"'{question}'에 대한 단서를 찾지 못했습니다.",
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
