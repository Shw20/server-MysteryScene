from typing import Any


class OpenAIEmbeddingProvider:
    """OpenAI 임베딩 API를 감싼 간단한 래퍼."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 32,
        client: object | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        self.client = client or self._build_client()

    def _build_client(self) -> object:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIEmbeddingProvider requires the 'openai' package to be installed."
            ) from exc

        client_kwargs: dict[str, Any] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        return OpenAI(**client_kwargs)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        request: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            request["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**request)
        # SDK가 순서를 바꿔 돌려주더라도 원래 입력 순서를 유지한다.
        sorted_data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        return [list(item.embedding) for item in sorted_data]

    def embed_text(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0] if results else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 빈 문자열은 공백 하나로 바꿔 제공자 쪽 입력 검증에 걸리지 않게 한다.
        normalized_texts = [str(text).strip() or " " for text in texts]
        if not normalized_texts:
            return []

        vectors: list[list[float]] = []
        for index in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[index : index + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        return vectors
