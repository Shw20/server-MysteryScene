from collections.abc import Callable

from .pipeline import RAGPipeline
from .schemas.chunk_schema import Chunk
from .schemas.retrieval_result import RetrievalResult


class RAGService:
    """컨트롤러나 배치 코드가 스토어 세부 구현을 몰라도 쓰게 해주는 서비스 레이어."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        default_top_k: int = 3,
        default_min_score: float | None = None,
    ):
        self.pipeline = pipeline
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score

    def index_items(
        self,
        raw_items: list[dict],
        scenario_id: str,
        replace_existing: bool = False,
    ) -> list[Chunk]:
        return self.pipeline.index_items(
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
        return self.pipeline.index_file(
            file_path=file_path,
            scenario_id=scenario_id,
            replace_existing=replace_existing,
        )

    def index_loader(
        self,
        loader: Callable[[], list[dict]],
        scenario_id: str,
        replace_existing: bool = False,
    ) -> list[Chunk]:
        return self.index_items(
            raw_items=loader(),
            scenario_id=scenario_id,
            replace_existing=replace_existing,
        )

    def search(
        self,
        query: str,
        scenario_id: str,
        top_k: int | None = None,
        min_score: float | None = None,
        level_lte: int | None = None,
        level_gte: int | None = None,
        category: str | None = None,
        tags_contains: str | list[str] | None = None,
        extra_filters: dict | None = None,
    ) -> list[RetrievalResult]:
        # 상위 레이어의 검색 옵션을 스토어 필터 형식으로 변환한다.
        filters = dict(extra_filters or {})
        filters["scenario_id"] = scenario_id

        if level_lte is not None:
            filters["level_lte"] = level_lte
        if level_gte is not None:
            filters["level_gte"] = level_gte
        if category:
            filters["category"] = category
        if tags_contains:
            filters["tags_contains"] = tags_contains

        effective_top_k = top_k or self.default_top_k
        effective_min_score = (
            min_score if min_score is not None else self.default_min_score
        )

        return self.pipeline.retrieve_with_scores(
            query=query,
            top_k=effective_top_k,
            filters=filters,
            min_score=effective_min_score,
        )

    def build_context(self, results: list[RetrievalResult]) -> str:
        """검색 결과를 프롬프트에 넣기 쉬운 텍스트 블록으로 만든다."""
        lines: list[str] = []
        for index, result in enumerate(results, start=1):
            lines.append(
                (
                    f"[{index}] score={result.score:.4f} "
                    f"level={result.chunk.metadata.level} "
                    f"category={result.chunk.metadata.category} "
                    f"content={result.chunk.content}"
                )
            )
        return "\n".join(lines)
