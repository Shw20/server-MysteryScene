from .config import RAGSettings
from .embeddings.base import EmbeddingProvider
from .embeddings.dummy_provider import DummyEmbeddingProvider
from .embeddings.openai_provider import OpenAIEmbeddingProvider
from .pipeline import RAGPipeline
from .service import RAGService
from .stores.base import VectorStore
from .stores.inmemory_store import InMemoryVectorStore
from .stores.sqlite_store import SQLiteVectorStore


def create_embedding_provider(
    settings: RAGSettings | None = None,
    client: object | None = None,
) -> EmbeddingProvider:
    """설정값을 기준으로 임베딩 백엔드를 고른다."""
    current_settings = settings or RAGSettings.from_env()
    provider_name = current_settings.embedding_provider.strip().lower()

    if provider_name == "auto":
        provider_name = "openai" if current_settings.openai_api_key else "dummy"

    if provider_name == "dummy":
        return DummyEmbeddingProvider()

    if provider_name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=current_settings.openai_api_key,
            model=current_settings.openai_embedding_model,
            base_url=current_settings.openai_base_url,
            dimensions=current_settings.openai_dimensions,
            batch_size=current_settings.openai_batch_size,
            client=client,
        )

    raise ValueError(f"Unsupported embedding provider: {provider_name}")


def create_vector_store(settings: RAGSettings | None = None) -> VectorStore:
    """설정에 맞는 벡터 스토어 구현체를 생성한다."""
    current_settings = settings or RAGSettings.from_env()
    store_name = current_settings.vector_store.strip().lower()

    if store_name == "inmemory":
        return InMemoryVectorStore()

    if store_name == "sqlite":
        return SQLiteVectorStore(current_settings.sqlite_db_path)

    raise ValueError(f"Unsupported vector store: {store_name}")


def create_pipeline(
    settings: RAGSettings | None = None,
    client: object | None = None,
) -> RAGPipeline:
    """임베딩과 저장소를 묶어 RAG 파이프라인을 구성한다."""
    current_settings = settings or RAGSettings.from_env()
    embedding_provider = create_embedding_provider(current_settings, client=client)
    vector_store = create_vector_store(current_settings)
    return RAGPipeline(embedding_provider, vector_store)


def create_service(
    settings: RAGSettings | None = None,
    client: object | None = None,
) -> RAGService:
    """파이프라인 위에 서비스 레이어를 얹어 상위 코드가 쓰기 쉽게 만든다."""
    current_settings = settings or RAGSettings.from_env()
    pipeline = create_pipeline(current_settings, client=client)
    return RAGService(
        pipeline=pipeline,
        default_top_k=current_settings.default_top_k,
        default_min_score=current_settings.default_min_score,
    )
