from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """필터링과 프롬프트 구성에 함께 쓰는 메타데이터."""

    scenario_id: str
    level: int
    category: str
    tags: list[str] = Field(default_factory=list)
    source_id: str
    lang: str = "ko"


class Chunk(BaseModel):
    """벡터 데이터베이스에 저장되는 정규화된 문서 단위."""

    chunk_id: str
    document_id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None
