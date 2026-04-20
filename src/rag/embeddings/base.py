from typing import Protocol


class EmbeddingProvider(Protocol):
    """RAG 파이프라인에서 사용하는 모든 임베딩 백엔드의 공통 인터페이스."""

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
