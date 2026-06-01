from ..schemas.chunk_schema import Chunk, ChunkMetadata


def _extract_items(raw_items: list[dict] | dict) -> tuple[list[dict], str | None]:
    """시나리오 JSON과 증거 목록 JSON을 모두 같은 items 형태로 꺼낸다."""
    if isinstance(raw_items, dict):
        scenario_id = raw_items.get("scenario_id")
        items = raw_items.get("evidence", [])
        if not isinstance(items, list):
            items = []
        return items, str(scenario_id).strip() if scenario_id else None

    return raw_items, None


def _normalize_keywords(keywords: object) -> list[str]:
    """입력 형식이 조금 달라도 최종적으로 깔끔한 태그 문자열 목록을 만든다."""
    if not isinstance(keywords, list):
        return []

    normalized: list[str] = []
    for keyword in keywords:
        value = str(keyword).strip()
        if value:
            normalized.append(value)

    return normalized


def _extract_entity_tags(entities: object) -> list[str]:
    """인물/물건/장소 entity를 검색 태그로 펼친다."""
    if not isinstance(entities, list):
        return []

    tags: list[str] = []
    for entity in entities:
        if isinstance(entity, dict):
            tags.extend(_normalize_keywords([entity.get("name")]))
            tags.extend(_normalize_keywords(entity.get("aliases", [])))
            tags.extend(_normalize_keywords([entity.get("role")]))
        else:
            tags.extend(_normalize_keywords([entity]))

    return tags


def _merge_tags(*groups: list[str]) -> list[str]:
    """키워드, 태그, entity alias를 순서 보존 중복 제거로 합친다."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for tag in group:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return merged


def _build_content(item: dict) -> str:
    """검색 대상 본문을 title/content/summary/search_text에서 조립한다."""
    parts = [
        str(item.get("title", "")).strip(),
        str(item.get("content", "")).strip(),
        str(item.get("summary", "")).strip(),
        str(item.get("search_text", "")).strip(),
    ]
    return "\n".join(part for part in parts if part)


def normalize_evidence(
    raw_items: list[dict] | dict,
    scenario_id: str | None = None,
) -> list[Chunk]:
    """원본 증거 행을 인덱싱용 공통 청크 스키마로 변환한다."""
    chunks: list[Chunk] = []
    items, document_scenario_id = _extract_items(raw_items)
    active_scenario_id = scenario_id or document_scenario_id

    if not active_scenario_id:
        raise ValueError("scenario_id is required")

    for item in items:
        source_id = str(item.get("id", "")).strip()
        content = _build_content(item)

        # 이후 검색이나 참조에 쓸 수 없는 행은 건너뛴다.
        if not source_id or not content:
            continue

        level = int(item.get("level", 1))
        category = str(item.get("category", "unknown")).strip() or "unknown"
        keywords = _merge_tags(
            _normalize_keywords(item.get("keywords", [])),
            _normalize_keywords(item.get("tags", [])),
            _extract_entity_tags(item.get("entities", [])),
        )

        chunk = Chunk(
            chunk_id=f"{active_scenario_id}_{source_id}",
            document_id=active_scenario_id,
            content=content,
            metadata=ChunkMetadata(
                scenario_id=active_scenario_id,
                level=level,
                category=category,
                tags=keywords,
                source_id=source_id,
                lang="ko",
            ),
            embedding=None,
        )
        chunks.append(chunk)

    return chunks


def normalize_team_evidence(raw_items: list[dict], scenario_id: str) -> list[Chunk]:
    """기존 호출 코드와 스모크 테스트 호환을 위해 남겨둔 별칭 함수."""
    return normalize_evidence(raw_items, scenario_id)
