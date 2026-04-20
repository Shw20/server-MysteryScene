from .embeddings.base import EmbeddingProvider
from .ingestion.indexer import Indexer
from .retrieval.retriever import Retriever
from .schemas.chunk_schema import Chunk
from .schemas.retrieval_result import RetrievalResult
from .stores.base import VectorStore


class RAGPipeline:
    """인덱싱과 검색이 같은 의존성을 공유하도록 묶어주는 저수준 파사드."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.indexer = Indexer(embedding_provider, vector_store)
        self.retriever = Retriever(embedding_provider, vector_store)

    def index_items(
        self,
        raw_items: list[dict],
        scenario_id: str,
        replace_existing: bool = False,
    ) -> list[Chunk]:
        return self.indexer.index_items(
            raw_items=raw_items,
            scenario_id=scenario_id,
            replace_existing=replace_existing,
        )

    def index_file(
        self,
        file_path: str,
        scenario_id: str,
        replace_existing: bool = False,
    ) -> list[Chunk]:
        return self.indexer.index_file(
            file_path=file_path,
            scenario_id=scenario_id,
            replace_existing=replace_existing,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[Chunk]:
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 3,
        filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        return self.retriever.retrieve_with_scores(
            query=query,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    def delete_document(self, document_id: str) -> int:
        return self.vector_store.delete_document(document_id)

    def count(self, filters: dict | None = None) -> int:
        return self.vector_store.count(filters)
