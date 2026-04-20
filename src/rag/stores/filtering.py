from ..schemas.chunk_schema import Chunk


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def _matches_scalar(actual: object, expected: object) -> bool:
    if _is_sequence(expected):
        return actual in expected
    return actual == expected


def _normalize_required_tags(value: object) -> list[str]:
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else []

    if not _is_sequence(value):
        return []

    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate:
            normalized.append(candidate)

    return normalized


def match_chunk_filters(chunk: Chunk, filters: dict | None) -> bool:
    """백엔드 종류와 무관하게 동일한 필터 규칙을 적용한다."""
    if not filters:
        return True

    for key, value in filters.items():
        if key == "scenario_id" and not _matches_scalar(
            chunk.metadata.scenario_id,
            value,
        ):
            return False

        if key == "document_id" and not _matches_scalar(chunk.document_id, value):
            return False

        if key == "category" and not _matches_scalar(chunk.metadata.category, value):
            return False

        if key == "source_id" and not _matches_scalar(chunk.metadata.source_id, value):
            return False

        if key == "lang" and not _matches_scalar(chunk.metadata.lang, value):
            return False

        if key == "level" and not _matches_scalar(chunk.metadata.level, value):
            return False

        if key == "level_lte" and chunk.metadata.level > int(value):
            return False

        if key == "level_gte" and chunk.metadata.level < int(value):
            return False

        if key == "tags_contains":
            required_tags = _normalize_required_tags(value)
            if any(tag not in chunk.metadata.tags for tag in required_tags):
                return False

    return True
