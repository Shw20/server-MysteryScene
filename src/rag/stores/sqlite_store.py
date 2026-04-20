import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from ..schemas.chunk_schema import Chunk, ChunkMetadata
from .filtering import match_chunk_filters


class SQLiteVectorStore:
    """프로세스가 재시작돼도 청크를 유지하는 SQLite 기반 스토어."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    lang TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _serialize_chunk(self, chunk: Chunk) -> tuple:
        return (
            chunk.chunk_id,
            chunk.document_id,
            chunk.content,
            json.dumps(chunk.embedding or []),
            chunk.metadata.scenario_id,
            chunk.metadata.level,
            chunk.metadata.category,
            json.dumps(chunk.metadata.tags, ensure_ascii=False),
            chunk.metadata.source_id,
            chunk.metadata.lang,
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["content"],
            metadata=ChunkMetadata(
                scenario_id=row["scenario_id"],
                level=row["level"],
                category=row["category"],
                tags=json.loads(row["tags"]),
                source_id=row["source_id"],
                lang=row["lang"],
            ),
            embedding=json.loads(row["embedding"]),
        )

    def _build_where_clause(self, filters: dict | None) -> tuple[str, list]:
        if not filters:
            return "", []

        clauses: list[str] = []
        params: list[object] = []

        if isinstance(filters.get("scenario_id"), str):
            clauses.append("scenario_id = ?")
            params.append(filters["scenario_id"])

        if isinstance(filters.get("document_id"), str):
            clauses.append("document_id = ?")
            params.append(filters["document_id"])

        if isinstance(filters.get("category"), str):
            clauses.append("category = ?")
            params.append(filters["category"])

        if isinstance(filters.get("source_id"), str):
            clauses.append("source_id = ?")
            params.append(filters["source_id"])

        if isinstance(filters.get("lang"), str):
            clauses.append("lang = ?")
            params.append(filters["lang"])

        if "level" in filters and isinstance(filters["level"], int):
            clauses.append("level = ?")
            params.append(filters["level"])

        if "level_lte" in filters:
            clauses.append("level <= ?")
            params.append(int(filters["level_lte"]))

        if "level_gte" in filters:
            clauses.append("level >= ?")
            params.append(int(filters["level_gte"]))

        if not clauses:
            return "", params

        return f" WHERE {' AND '.join(clauses)}", params

    def _fetch_chunks(self, filters: dict | None = None) -> list[Chunk]:
        where_clause, params = self._build_where_clause(filters)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM rag_chunks{where_clause}",
                params,
            ).fetchall()
        chunks = [self._row_to_chunk(row) for row in rows]
        # SQL은 단순 조건을 처리하고, 파이썬은 공통 필터 규칙을 유지한다.
        return [chunk for chunk in chunks if match_chunk_filters(chunk, filters)]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        with closing(self._connect()) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO rag_chunks (
                    chunk_id,
                    document_id,
                    content,
                    embedding,
                    scenario_id,
                    level,
                    category,
                    tags,
                    source_id,
                    lang
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._serialize_chunk(chunk) for chunk in chunks],
            )
            connection.commit()

    def delete_document(self, document_id: str) -> int:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM rag_chunks WHERE document_id = ?",
                (document_id,),
            )
            connection.commit()
        return cursor.rowcount if cursor.rowcount != -1 else 0

    def count(self, filters: dict | None = None) -> int:
        return len(self._fetch_chunks(filters))

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[Chunk]:
        return [
            chunk
            for _, chunk in self.search_with_scores(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
        ]

    def search_with_scores(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[float, Chunk]]:
        scored: list[tuple[float, Chunk]] = []

        for chunk in self._fetch_chunks(filters):
            if chunk.embedding is None:
                continue

            score = self._cosine_similarity(query_vector, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        return None
