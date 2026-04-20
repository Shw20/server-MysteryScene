from src.database.db_handler import load_evidence
from src.rag.embeddings.dummy_provider import DummyEmbeddingProvider
from src.rag.ingestion.normalizer import normalize_team_evidence
from src.rag.retrieval.retriever import Retriever
from src.rag.stores.inmemory_store import InMemoryVectorStore


def main():
    # 외부 서비스 없이도 인메모리 경로 전체를 빠르게 점검할 수 있는 스크립트다.
    raw_items = load_evidence()
    print("=== 원본 evidence 확인 ===")
    print(f"원본 개수: {len(raw_items)}")

    chunks = normalize_team_evidence(raw_items, scenario_id="case001")
    print("\n=== 정규화 완료 ===")
    print(f"chunk 개수: {len(chunks)}")

    embedding_provider = DummyEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    vectors = embedding_provider.embed_texts([c.content for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    vector_store.upsert(chunks)
    print(f"vector store 저장 개수: {len(vector_store.items)}")

    retriever = Retriever(embedding_provider, vector_store)

    results = retriever.retrieve(
        query="사인은 뭐야?",
        top_k=3,
        filters={
            "scenario_id": "case001",
            "level_lte": 2,
        },
    )

    print("\n=== 검색 결과 ===")
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] {item.chunk_id}")
        print(f"    category={item.metadata.category}")
        print(f"    level={item.metadata.level}")
        print(f"    content={item.content}")

    assert len(raw_items) > 0, "원본 evidence가 비어 있습니다."
    assert len(chunks) == len(raw_items), "정규화 개수가 맞지 않습니다."
    assert len(vector_store.items) == len(chunks), "vector store 적재 개수가 맞지 않습니다."
    assert all(item.metadata.scenario_id == "case001" for item in results), "scenario_id filter 실패"
    assert all(item.metadata.level <= 2 for item in results), "level_lte filter 실패"

    print("\n정상: smoke test 통과")


if __name__ == "__main__":
    main()
