from ..embeddings.base import EmbeddingProvider
from ..schemas.chunk_schema import Chunk
from ..schemas.retrieval_result import RetrievalResult
from ..stores.base import VectorStore


class Retriever:
    """설정된 벡터 스토어를 대상으로 시맨틱 검색을 수행한다."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[Chunk]:
        return [
            result.chunk
            for result in self.retrieve_with_scores(
                query=query,
                top_k=top_k,
                filters=filters,
                min_score=min_score,
            )
        ]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 3,
        filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        clean_query = query.strip()
        if not clean_query or top_k <= 0:
            return []

        query_vector = self.embedding_provider.embed_text(clean_query)
        scored_chunks = self.vector_store.search_with_scores(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
        results: list[RetrievalResult] = []
        for score, chunk in scored_chunks:
            # 점수 기준은 검색 후 공통 규칙으로 적용해 스토어별 차이를 줄인다.
            if min_score is not None and score < min_score:
                continue
            results.append(RetrievalResult(chunk=chunk, score=score))

        return results
