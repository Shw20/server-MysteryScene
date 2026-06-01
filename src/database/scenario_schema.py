from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator, validator

from .evidence_schema import EvidenceItem


def _clean_string_list(value: Any) -> list[str]:
    """시나리오 작성자가 문자열/배열 중 무엇을 넣어도 list[str]로 정규화한다."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


class ScenarioOverview(BaseModel):
    """플레이어에게 보여줄 소개와 내부 해설을 분리한다."""

    player_intro: str
    internal_summary: str


class RagPolicy(BaseModel):
    """RAG 인덱싱에 포함할 시나리오 섹션과 공개 범위를 정의한다."""

    index_sections: list[str] = Field(default_factory=lambda: ["evidence"])
    exclude_sections: list[str] = Field(default_factory=lambda: ["solution"])
    include_search_text: bool = True
    default_visibility_filter: list[Literal["public", "locked", "hidden"]] = Field(
        default_factory=lambda: ["public", "locked"]
    )


class Character(BaseModel):
    """시나리오 안의 인물 정보와 검색 alias."""

    id: str
    name: str
    role: Literal[
        "victim",
        "culprit",
        "suspect",
        "witness",
        "detective",
        "expert",
        "other",
    ]
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    known_facts: list[str] = Field(default_factory=list)

    @validator("id", "name", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """인물 id/name은 비어 있으면 참조 검증이 불가능하므로 막는다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @validator("aliases", "known_facts", pre=True)
    def normalize_lists(cls, value: Any) -> list[str]:
        """인물 별칭과 공개 사실을 중복 없는 리스트로 정리한다."""
        return _clean_string_list(value)


class Location(BaseModel):
    """단서와 타임라인에서 참조하는 장소."""

    id: str
    name: str
    description: str = ""

    @validator("id", "name", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """장소 id/name은 다른 섹션에서 참조하므로 필수로 검증한다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text


class TimelineEvent(BaseModel):
    """사건 흐름 설명용 타임라인 이벤트."""

    id: str
    time: str
    location_id: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    event: str
    source_evidence_ids: list[str] = Field(default_factory=list)

    @validator("id", "time", "event", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """타임라인의 필수 텍스트를 검증한다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @validator("location_id", pre=True)
    def normalize_optional_text(cls, value: Any) -> str | None:
        """장소가 없는 이벤트도 허용하되 공백은 None으로 정리한다."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @validator("character_ids", "source_evidence_ids", pre=True)
    def normalize_lists(cls, value: Any) -> list[str]:
        """타임라인 참조 목록을 list[str]로 맞춘다."""
        return _clean_string_list(value)


class UnlockCondition(BaseModel):
    """단서 해금 조건. 현재 게임 흐름은 주로 evidence_found를 사용한다."""

    type: Literal[
        "evidence_found",
        "level_reached",
        "question_answered",
        "fact_confirmed",
        "manual",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    min_level: int | None = None
    question_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)

    @validator("evidence_ids", "question_ids", "fact_ids", pre=True)
    def normalize_lists(cls, value: Any) -> list[str]:
        """조건에 쓰이는 참조 ID 목록을 정리한다."""
        return _clean_string_list(value)


class UnlockRule(BaseModel):
    """특정 조건이 만족되면 target_evidence_id를 열어주는 규칙."""

    id: str
    target_evidence_id: str
    condition: UnlockCondition
    description: str = ""


class ExpectedQuestion(BaseModel):
    """사건 시작 시 추천 질문과 예상되는 단서 연결."""

    id: str
    question: str
    expected_evidence_ids: list[str] = Field(default_factory=list)
    min_level: int = 1

    @validator("expected_evidence_ids", pre=True)
    def normalize_expected_ids(cls, value: Any) -> list[str]:
        """추천 질문이 기대하는 단서 ID 목록을 정리한다."""
        return _clean_string_list(value)


class Fact(BaseModel):
    """추후 사실 검증형 흐름을 위해 남겨둔 구조."""

    id: str
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    truth_status: Literal["true", "false", "unknown", "misleading"] = "true"

    @validator("evidence_ids", pre=True)
    def normalize_evidence_ids(cls, value: Any) -> list[str]:
        """사실을 뒷받침하는 단서 ID 목록을 정리한다."""
        return _clean_string_list(value)


class AccusationChecks(BaseModel):
    """정답 제출 전 반드시 확보해야 하는 단서와 허용 답변 키워드."""

    culprit_character_id: str
    required_evidence_ids: list[str] = Field(default_factory=list)
    accepted_motives: list[str] = Field(default_factory=list)
    accepted_methods: list[str] = Field(default_factory=list)

    @validator("required_evidence_ids", "accepted_motives", "accepted_methods", pre=True)
    def normalize_lists(cls, value: Any) -> list[str]:
        """채점 기준 목록을 중복 없는 리스트로 맞춘다."""
        return _clean_string_list(value)


class Solution(BaseModel):
    """사건 종결 시 서버가 사용하는 정답 데이터."""

    culprit_character_id: str
    motive: str
    method: str
    decisive_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str

    @validator("decisive_evidence_ids", pre=True)
    def normalize_decisive_ids(cls, value: Any) -> list[str]:
        """해설에 필요한 결정적 단서 ID 목록을 정리한다."""
        return _clean_string_list(value)


class ScenarioDataset(BaseModel):
    """팀원이 작성할 전체 시나리오 JSON의 v2 계약."""

    schema_version: Literal["2.0"] = "2.0"
    scenario_id: str
    title: str
    language: str = "ko"
    difficulty: int = Field(default=2, ge=1, le=5)
    overview: ScenarioOverview
    rag_policy: RagPolicy = Field(default_factory=RagPolicy)
    characters: list[Character] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    evidence: list[EvidenceItem]
    unlock_rules: list[UnlockRule] = Field(default_factory=list)
    questions: list[ExpectedQuestion] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    accusation_checks: AccusationChecks | None = None
    solution: Solution

    @validator("scenario_id", "title", "language", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """시나리오 최상위 식별 문자열을 검증한다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @root_validator(skip_on_failure=True)
    def validate_cross_references(cls, values: dict[str, Any]) -> dict[str, Any]:
        """시나리오 안의 모든 ID 참조가 실제 항목을 가리키는지 확인한다."""
        character_ids = {item.id for item in values.get("characters") or []}
        location_ids = {item.id for item in values.get("locations") or []}
        evidence_ids = {item.id for item in values.get("evidence") or []}
        question_ids = {item.id for item in values.get("questions") or []}
        fact_ids = {item.id for item in values.get("facts") or []}

        cls._require_unique("character", [item.id for item in values.get("characters") or []])
        cls._require_unique("location", [item.id for item in values.get("locations") or []])
        cls._require_unique("evidence", [item.id for item in values.get("evidence") or []])
        cls._require_unique("question", [item.id for item in values.get("questions") or []])
        cls._require_unique("fact", [item.id for item in values.get("facts") or []])

        for event in values.get("timeline") or []:
            cls._require_known("timeline.character_ids", event.character_ids, character_ids)
            cls._require_known("timeline.source_evidence_ids", event.source_evidence_ids, evidence_ids)
            if event.location_id:
                cls._require_known("timeline.location_id", [event.location_id], location_ids)

        for item in values.get("evidence") or []:
            cls._require_known("evidence.related_ids", item.related_ids, evidence_ids)
            cls._require_known("evidence.contradicts", item.contradicts, evidence_ids)
            if item.location_id:
                cls._require_known("evidence.location_id", [item.location_id], location_ids)

        for rule in values.get("unlock_rules") or []:
            cls._require_known("unlock_rules.target_evidence_id", [rule.target_evidence_id], evidence_ids)
            cls._require_known("unlock_rules.condition.evidence_ids", rule.condition.evidence_ids, evidence_ids)
            cls._require_known("unlock_rules.condition.question_ids", rule.condition.question_ids, question_ids)
            cls._require_known("unlock_rules.condition.fact_ids", rule.condition.fact_ids, fact_ids)

        for question in values.get("questions") or []:
            cls._require_known("questions.expected_evidence_ids", question.expected_evidence_ids, evidence_ids)

        for fact in values.get("facts") or []:
            cls._require_known("facts.evidence_ids", fact.evidence_ids, evidence_ids)

        accusation_checks = values.get("accusation_checks")
        if accusation_checks:
            cls._require_known(
                "accusation_checks.culprit_character_id",
                [accusation_checks.culprit_character_id],
                character_ids,
            )
            cls._require_known(
                "accusation_checks.required_evidence_ids",
                accusation_checks.required_evidence_ids,
                evidence_ids,
            )

        solution = values.get("solution")
        if solution:
            cls._require_known("solution.culprit_character_id", [solution.culprit_character_id], character_ids)
            cls._require_known("solution.decisive_evidence_ids", solution.decisive_evidence_ids, evidence_ids)

        return values

    @staticmethod
    def _require_unique(label: str, ids: list[str]) -> None:
        """같은 종류의 id가 중복되면 어느 단서를 가리키는지 모호해지므로 막는다."""
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate {label} ids: {', '.join(duplicates)}")

    @staticmethod
    def _require_known(label: str, ids: list[str], known_ids: set[str]) -> None:
        """참조한 id가 실제로 존재하는지 확인한다."""
        unknown = sorted(set(ids) - known_ids)
        if unknown:
            raise ValueError(f"{label} has unknown ids: {', '.join(unknown)}")
