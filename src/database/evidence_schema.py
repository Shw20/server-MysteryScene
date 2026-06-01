from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator, validator


def _clean_string_list(value: Any) -> list[str]:
    """문자열/리스트 입력을 중복 없는 문자열 리스트로 정리한다."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)

    return cleaned


class EvidenceEntity(BaseModel):
    """검색 필터와 키워드 확장에 쓰는 인물/장소/물건 단위."""

    name: str
    role: str = "unknown"
    aliases: list[str] = Field(default_factory=list)

    @validator("name", "role", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """필수 문자열 필드는 앞뒤 공백을 제거하고 빈 값을 막는다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @validator("aliases", pre=True)
    def normalize_aliases(cls, value: Any) -> list[str]:
        """alias는 검색 확장용이므로 항상 list[str] 형태로 맞춘다."""
        return _clean_string_list(value)


class EvidenceItem(BaseModel):
    """RAG 인덱싱에 필요한 최소 정보와 검색 품질용 메타데이터."""

    id: str
    title: str
    level: int = Field(default=1, ge=1, le=5)
    category: str
    content: str
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    entities: list[EvidenceEntity] = Field(default_factory=list)
    source_type: str = "unknown"
    location: str | None = None
    location_id: str | None = None
    event_time: str | None = None
    image_url: str | None = None
    image_alt: str | None = None
    related_ids: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    visibility: Literal["public", "locked", "hidden"] = "public"
    is_red_herring: bool = False
    spoiler_level: Literal["none", "hint", "critical", "solution"] = "hint"
    search_text: str | None = None

    @validator("id", "title", "category", "content", "source_type", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """RAG 청크 생성에 필요한 필수 텍스트를 검증한다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @validator(
        "summary",
        "location",
        "location_id",
        "event_time",
        "image_url",
        "image_alt",
        "search_text",
        pre=True,
    )
    def normalize_optional_text(cls, value: Any) -> str | None:
        """선택 텍스트 필드는 공백뿐이면 None으로 통일한다."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @validator("keywords", "tags", "related_ids", "contradicts", pre=True)
    def normalize_string_lists(cls, value: Any) -> list[str]:
        """검색 태그와 참조 ID 목록을 중복 없는 리스트로 정리한다."""
        return _clean_string_list(value)


class EvidenceDataset(BaseModel):
    """증거 JSON 파일의 v2 최상위 계약."""

    schema_version: Literal["2.0"] = "2.0"
    scenario_id: str
    title: str
    language: str = "ko"
    evidence: list[EvidenceItem]

    @validator("scenario_id", "title", "language", pre=True)
    def normalize_required_text(cls, value: Any) -> str:
        """데이터셋 식별에 필요한 최상위 문자열을 검증한다."""
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @root_validator(skip_on_failure=True)
    def validate_references(cls, values: dict[str, Any]) -> dict[str, Any]:
        """중복 evidence id와 깨진 related/contradicts 참조를 배포 전에 잡는다."""
        items: list[EvidenceItem] = values.get("evidence") or []
        ids = [item.id for item in items]
        duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicate_ids:
            raise ValueError(f"duplicate evidence ids: {', '.join(duplicate_ids)}")

        known_ids = set(ids)
        for item in items:
            unknown_related = sorted(set(item.related_ids) - known_ids)
            unknown_contradicts = sorted(set(item.contradicts) - known_ids)
            if unknown_related:
                raise ValueError(
                    f"{item.id} has unknown related_ids: {', '.join(unknown_related)}"
                )
            if unknown_contradicts:
                raise ValueError(
                    f"{item.id} has unknown contradicts: {', '.join(unknown_contradicts)}"
                )

        return values
