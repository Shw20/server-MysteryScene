import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.rag.bootstrap import create_embedding_provider, create_service
from src.rag.config import RAGSettings
from src.rag.embeddings.dummy_provider import DummyEmbeddingProvider
from src.rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.rag.ingestion.indexer import Indexer
from src.rag.pipeline import RAGPipeline
from src.rag.stores.inmemory_store import InMemoryVectorStore
from src.rag.stores.sqlite_store import SQLiteVectorStore


def build_raw_items() -> list[dict]:
    return [
        {
            "id": "E01",
            "level": 1,
            "category": "scene",
            "content": "poison residue found in the coffee cup",
            "keywords": ["coffee", "poison"],
        },
        {
            "id": "E02",
            "level": 3,
            "category": "alibi",
            "content": "suspect lied about the library alibi",
            "keywords": ["suspect", "alibi"],
        },
    ]


class RAGPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_provider = DummyEmbeddingProvider()

    def test_indexer_replaces_existing_document(self) -> None:
        store = InMemoryVectorStore()
        indexer = Indexer(self.embedding_provider, store)

        indexer.index_items(build_raw_items(), scenario_id="case001")
        self.assertEqual(store.count({"scenario_id": "case001"}), 2)

        indexer.index_items(
            raw_items=build_raw_items()[:1],
            scenario_id="case001",
            replace_existing=True,
        )

        self.assertEqual(store.count({"scenario_id": "case001"}), 1)
        self.assertEqual(store.items[0].metadata.source_id, "E01")

    def test_pipeline_retrieves_filtered_chunks(self) -> None:
        pipeline = RAGPipeline(self.embedding_provider, InMemoryVectorStore())
        raw_items = build_raw_items()

        pipeline.index_items(raw_items, scenario_id="case001", replace_existing=True)
        results = pipeline.retrieve(
            query=raw_items[0]["content"],
            top_k=2,
            filters={
                "scenario_id": "case001",
                "level_lte": 2,
                "tags_contains": "coffee",
            },
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata.source_id, "E01")

    def test_sqlite_vector_store_persists_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.sqlite3"

            store = SQLiteVectorStore(str(db_path))
            pipeline = RAGPipeline(self.embedding_provider, store)
            pipeline.index_items(
                build_raw_items(),
                scenario_id="case001",
                replace_existing=True,
            )

            self.assertEqual(store.count({"scenario_id": "case001"}), 2)
            store.close()

            reopened_store = SQLiteVectorStore(str(db_path))
            results = reopened_store.search(
                query_vector=self.embedding_provider.embed_text(
                    build_raw_items()[0]["content"]
                ),
                top_k=1,
                filters={"scenario_id": "case001"},
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata.source_id, "E01")
            reopened_store.close()

    def test_service_applies_default_min_score(self) -> None:
        service = create_service(
            RAGSettings(
                embedding_provider="dummy",
                vector_store="inmemory",
                default_top_k=2,
                default_min_score=1.01,
            )
        )
        service.index_items(
            build_raw_items(),
            scenario_id="case001",
            replace_existing=True,
        )

        results = service.search(
            query=build_raw_items()[0]["content"],
            scenario_id="case001",
        )

        self.assertEqual(results, [])

    def test_create_embedding_provider_uses_dummy_without_api_key(self) -> None:
        provider = create_embedding_provider(
            RAGSettings(
                embedding_provider="auto",
                vector_store="inmemory",
                openai_api_key=None,
            )
        )

        self.assertIsInstance(provider, DummyEmbeddingProvider)

    def test_openai_provider_accepts_injected_client(self) -> None:
        fake_client = FakeOpenAIClient()
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=8,
            batch_size=2,
            client=fake_client,
        )

        vectors = provider.embed_texts(
            [
                "alpha",
                "beta",
                "gamma",
            ]
        )

        self.assertEqual(len(vectors), 3)
        self.assertEqual(fake_client.embeddings.calls[0]["model"], "text-embedding-3-small")
        self.assertEqual(fake_client.embeddings.calls[0]["dimensions"], 8)
        self.assertEqual(fake_client.embeddings.calls[1]["input"], ["gamma"])


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
                for index, text in enumerate(kwargs["input"])
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()
