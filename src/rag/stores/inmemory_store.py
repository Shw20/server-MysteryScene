import math
from ..schemas.chunk_schema import Chunk
from .filtering import match_chunk_filters


class InMemoryVectorStore:
    """로컬 개발과 빠른 단위 테스트를 위한 간단한 인메모리 벡터 스토어."""

    def __init__(self):
        self._items: dict[str, Chunk] = {}

    @property
    def items(self) -> list[Chunk]:
        return list(self._items.values())

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            # chunk_id를 키로 쓰면 같은 청크는 자연스럽게 덮어쓰기된다.
            self._items[chunk.chunk_id] = chunk

    def delete_document(self, document_id: str) -> int:
        chunk_ids = [
            chunk_id
            for chunk_id, chunk in self._items.items()
            if chunk.document_id == document_id
        ]

        for chunk_id in chunk_ids:
            del self._items[chunk_id]

        return len(chunk_ids)

    def count(self, filters: dict | None = None) -> int:
        return sum(
            1 for chunk in self._items.values() if match_chunk_filters(chunk, filters)
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[Chunk]:
        return [
            chunk
            for _, chunk in self.search_with_scores(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
        ]

    def search_with_scores(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[float, Chunk]]:
        scored: list[tuple[float, Chunk]] = []

        for item in self._items.values():
            if item.embedding is None:
                continue
            if not match_chunk_filters(item, filters):
                continue

            score = self._cosine_similarity(query_vector, item.embedding)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
