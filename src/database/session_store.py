import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass
class SessionSnapshot:
    """DB와 컨트롤러 사이에서 주고받는 한 세션의 현재 상태."""

    discovered_evidence_ids: set[str] = field(default_factory=set)
    answered_question_ids: set[str] = field(default_factory=set)
    confirmed_fact_ids: set[str] = field(default_factory=set)
    question_count: int = 0
    submission_count: int = 0
    solved: bool = False


class SQLiteSessionStore:
    """데모용 SQLite 세션 저장소.

    발견 단서, 최근 대화, 제출 기록을 저장해서 새 요청에서도 맥락을 이어간다.
    운영 DB로 바꾸더라도 이 클래스의 공개 메서드 형태를 맞추면 컨트롤러는 그대로 쓸 수 있다.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """SQLite 연결을 만들고 row를 dict처럼 읽을 수 있게 설정한다."""
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """성공 시 commit, 실패 시 rollback하는 공통 트랜잭션 래퍼."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        """필요한 테이블을 최초 1회 생성한다."""
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS game_sessions (
                    session_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    submission_count INTEGER NOT NULL DEFAULT 0,
                    solved INTEGER NOT NULL DEFAULT 0,
                    answered_question_ids TEXT NOT NULL DEFAULT '[]',
                    confirmed_fact_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, scenario_id)
                );

                CREATE TABLE IF NOT EXISTS session_evidence (
                    session_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, scenario_id, evidence_id),
                    FOREIGN KEY (session_id, scenario_id)
                        REFERENCES game_sessions(session_id, scenario_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT,
                    evidence_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id, scenario_id)
                        REFERENCES game_sessions(session_id, scenario_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    solved INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id, scenario_id)
                        REFERENCES game_sessions(session_id, scenario_id)
                        ON DELETE CASCADE
                );
                """
            )

    def load_session(self, session_id: str, scenario_id: str) -> SessionSnapshot:
        """세션 상태와 발견 단서를 한 번에 읽어 컨트롤러 상태 객체로 돌려준다."""
        self.ensure_session(session_id, scenario_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT question_count, submission_count, solved,
                       answered_question_ids, confirmed_fact_ids
                FROM game_sessions
                WHERE session_id = ? AND scenario_id = ?
                """,
                (session_id, scenario_id),
            ).fetchone()
            evidence_rows = connection.execute(
                """
                SELECT evidence_id
                FROM session_evidence
                WHERE session_id = ? AND scenario_id = ?
                """,
                (session_id, scenario_id),
            ).fetchall()

        return SessionSnapshot(
            discovered_evidence_ids={row["evidence_id"] for row in evidence_rows},
            answered_question_ids=set(_json_list(row["answered_question_ids"])),
            confirmed_fact_ids=set(_json_list(row["confirmed_fact_ids"])),
            question_count=int(row["question_count"]),
            submission_count=int(row["submission_count"]),
            solved=bool(row["solved"]),
        )

    def ensure_session(self, session_id: str, scenario_id: str) -> None:
        """세션 행이 없으면 만든다. 이미 있으면 아무 일도 하지 않는다."""
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO game_sessions (
                    session_id, scenario_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (session_id, scenario_id, now, now),
            )

    def save_session(
        self,
        session_id: str,
        scenario_id: str,
        snapshot: SessionSnapshot,
    ) -> None:
        """컨트롤러에서 갱신한 상태를 DB에 저장한다."""
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO game_sessions (
                    session_id, scenario_id, question_count, submission_count,
                    solved, answered_question_ids, confirmed_fact_ids,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, scenario_id) DO UPDATE SET
                    question_count = excluded.question_count,
                    submission_count = excluded.submission_count,
                    solved = excluded.solved,
                    answered_question_ids = excluded.answered_question_ids,
                    confirmed_fact_ids = excluded.confirmed_fact_ids,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    scenario_id,
                    snapshot.question_count,
                    snapshot.submission_count,
                    int(snapshot.solved),
                    _dump_list(snapshot.answered_question_ids),
                    _dump_list(snapshot.confirmed_fact_ids),
                    now,
                    now,
                ),
            )
            for evidence_id in sorted(snapshot.discovered_evidence_ids):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_evidence (
                        session_id, scenario_id, evidence_id, discovered_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, scenario_id, evidence_id, now),
                )

    def record_chat_message(
        self,
        session_id: str,
        scenario_id: str,
        role: str,
        content: str,
        status: str | None = None,
        evidence_ids: list[str] | None = None,
    ) -> None:
        """최근 대화 메모리와 디버깅을 위해 사용자/조사관 메시지를 저장한다."""
        self.ensure_session(session_id, scenario_id)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    session_id, scenario_id, role, content, status,
                    evidence_ids, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    scenario_id,
                    role,
                    content,
                    status,
                    _dump_list(evidence_ids or []),
                    _utc_now(),
                ),
            )

    def load_recent_chat_messages(
        self,
        session_id: str,
        scenario_id: str,
        limit: int = 8,
    ) -> list[dict]:
        """LLM 답변에 넣을 최근 대화를 오래된 순서로 돌려준다."""
        self.ensure_session(session_id, scenario_id)
        safe_limit = max(0, min(int(limit), 20))
        if safe_limit == 0:
            return []

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, status, evidence_ids, created_at
                FROM chat_messages
                WHERE session_id = ? AND scenario_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, scenario_id, safe_limit),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "status": row["status"],
                "evidence_ids": _json_list(row["evidence_ids"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def record_submission(
        self,
        session_id: str,
        scenario_id: str,
        answer: str,
        status: str,
        solved: bool,
    ) -> None:
        """정답 제출 이력을 저장한다."""
        self.ensure_session(session_id, scenario_id)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO submissions (
                    session_id, scenario_id, answer, status, solved, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, scenario_id, answer, status, int(solved), _utc_now()),
            )


def _json_list(value: str) -> list[str]:
    """DB에 문자열로 저장된 JSON 배열을 안전하게 list[str]로 복원한다."""
    try:
        raw = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _dump_list(values) -> str:
    """set/list 값을 정렬된 JSON 배열 문자열로 저장한다."""
    return json.dumps(sorted({str(value) for value in values}), ensure_ascii=False)


def _utc_now() -> str:
    """DB timestamp를 UTC ISO 문자열로 통일한다."""
    return datetime.now(timezone.utc).isoformat()
