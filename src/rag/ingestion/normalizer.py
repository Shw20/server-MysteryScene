from ..schemas.chunk_schema import Chunk, ChunkMetadata


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


def normalize_evidence(raw_items: list[dict], scenario_id: str) -> list[Chunk]:
    """원본 증거 행을 인덱싱용 공통 청크 스키마로 변환한다."""
    chunks: list[Chunk] = []

    for item in raw_items:
        source_id = str(item.get("id", "")).strip()
        content = str(item.get("content", "")).strip()

        # 이후 검색이나 참조에 쓸 수 없는 행은 건너뛴다.
        if not source_id or not content:
            continue

        level = int(item.get("level", 1))
        category = str(item.get("category", "unknown")).strip() or "unknown"
        keywords = _normalize_keywords(item.get("keywords", []))

        chunk = Chunk(
            chunk_id=f"{scenario_id}_{source_id}",
            document_id=scenario_id,
            content=content,
            metadata=ChunkMetadata(
                scenario_id=scenario_id,
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
