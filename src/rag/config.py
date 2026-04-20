import os
from dataclasses import dataclass


def _to_int(value: str | None, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    return int(value)


def _to_float(value: str | None, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


@dataclass(slots=True)
class RAGSettings:
    """임베딩, 스토어, 검색 기본값을 담는 환경 기반 설정 객체."""

    embedding_provider: str = "auto"
    vector_store: str = "sqlite"
    sqlite_db_path: str = "data/rag_store.sqlite3"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str | None = None
    openai_dimensions: int | None = None
    openai_batch_size: int = 32
    default_top_k: int = 3
    default_min_score: float | None = None

    @classmethod
    def from_env(cls) -> "RAGSettings":
        """환경 변수를 읽어 나머지 코드가 설정 세부사항을 몰라도 되게 한다."""
        return cls(
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "auto"),
            vector_store=os.getenv("RAG_VECTOR_STORE", "sqlite"),
            sqlite_db_path=os.getenv("RAG_SQLITE_PATH", "data/rag_store.sqlite3"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_dimensions=_to_int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS")),
            openai_batch_size=_to_int(
                os.getenv("OPENAI_EMBEDDING_BATCH_SIZE"),
                32,
            )
            or 32,
            default_top_k=_to_int(os.getenv("RAG_DEFAULT_TOP_K"), 3) or 3,
            default_min_score=_to_float(os.getenv("RAG_MIN_SCORE")),
        )
