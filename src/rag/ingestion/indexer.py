import json
from pathlib import Path

from ..embeddings.base import EmbeddingProvider
from ..schemas.chunk_schema import Chunk
from ..stores.base import VectorStore
from .normalizer import normalize_evidence


class Indexer:
    """원본 증거 데이터를 임베딩 가능한 청크로 바꿔 저장한다."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index_items(
        self,
        raw_items: list[dict],
        scenario_id: str,
        replace_existing: bool = False,
    ) -> list[Chunk]:
        chunks = normalize_evidence(raw_items, scenario_id)

        if not chunks:
            if replace_existing:
                self.vector_store.delete_document(scenario_id)
            return []

        vectors = self.embedding_provider.embed_texts([c.content for c in chunks])

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        if replace_existing:
            # 재색인 시에는 기존 청크를 먼저 지워 전체 갱신처럼 동작하게 한다.
            self.vector_store.delete_document(scenario_id)

        self.vector_store.upsert(chunks)
        return chunks

    def index_file(
        self,
        file_path: str,
        scenario_id: str,
        replace_existing: bool = False,
    ) -> list[Chunk]:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as f:
            raw_items = json.load(f)

        return self.index_items(
            raw_items=raw_items,
            scenario_id=scenario_id,
            replace_existing=replace_existing,
        )
