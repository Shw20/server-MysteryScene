from typing import Protocol
from ..schemas.chunk_schema import Chunk


class VectorStore(Protocol):
    """모든 벡터 스토어 백엔드가 따라야 하는 최소 계약."""

    def upsert(self, chunks: list[Chunk]) -> None:
        ...

    def delete_document(self, document_id: str) -> int:
        ...

    def count(self, filters: dict | None = None) -> int:
        ...

    def search_with_scores(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[float, Chunk]]:
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[Chunk]:
        ...
