class DummyEmbeddingProvider:
    """로컬 테스트와 스모크 테스트에 쓰는 결정적 더미 임베딩 제공자."""

    def embed_text(self, text: str) -> list[float]:
        base = sum(ord(c) for c in text)
        return [float((base + i) % 100) for i in range(8)]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]
