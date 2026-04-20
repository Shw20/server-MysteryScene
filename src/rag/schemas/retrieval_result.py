from dataclasses import dataclass

from .chunk_schema import Chunk


@dataclass(slots=True)
class RetrievalResult:
    """검색된 청크와 그 유사도 점수를 함께 담는 결과 객체."""

    chunk: Chunk
    score: float
