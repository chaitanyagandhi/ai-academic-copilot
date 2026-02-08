import os
import time
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Text,
    Float,
    ForeignKey,
    text,
)


_BACKEND_DIR = Path(__file__).resolve().parents[2]
DB_PATH = str(_BACKEND_DIR / "data" / "app.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)

metadata = MetaData()

courses = Table(
    "courses",
    metadata,
    Column("course_id", String, primary_key=True),
    Column("created_at", Float, nullable=False),
)

lectures = Table(
    "lectures",
    metadata,
    Column("course_id", String, ForeignKey("courses.course_id"), primary_key=True),
    Column("lecture_id", String, primary_key=True),
    Column("created_at", Float, nullable=False),
)

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("course_id", String, ForeignKey("courses.course_id"), nullable=False),
    Column("lecture_id", String, nullable=True),
    Column("source_name", String, nullable=False),
    Column("created_at", Float, nullable=False),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("document_id", Integer, ForeignKey("documents.id"), nullable=False),
    Column("chunk_id", String, nullable=False, unique=True),
    Column("text", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)

chunk_embeddings = Table(
    "chunk_embeddings",
    metadata,
    Column("chunk_id", String, ForeignKey("chunks.chunk_id"), primary_key=True),
    Column("model", String, nullable=False),
    Column("vector_json", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)

questions = Table(
    "questions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("course_id", String, ForeignKey("courses.course_id"), nullable=False),
    Column("lecture_id", String, nullable=True),
    Column("user_id", String, nullable=False),
    Column("question", Text, nullable=False),
    Column("confusion", Float, nullable=False),
    Column("timestamp", Float, nullable=False),
)


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    metadata.create_all(engine)
    _ensure_column("documents", "lecture_id", "TEXT")
    _ensure_column("questions", "lecture_id", "TEXT")


@contextmanager
def db_conn():
    with engine.begin() as conn:
        yield conn


def ensure_course(conn, course_id: str) -> None:
    exists = conn.execute(
        courses.select().where(courses.c.course_id == course_id)
    ).first()
    if exists is None:
        conn.execute(
            courses.insert().values(course_id=course_id, created_at=time.time())
        )


def ensure_lecture(conn, course_id: str, lecture_id: str | None) -> None:
    if not lecture_id:
        return
    ensure_course(conn, course_id)
    exists = conn.execute(
        lectures.select()
        .where(lectures.c.course_id == course_id)
        .where(lectures.c.lecture_id == lecture_id)
    ).first()
    if exists is None:
        conn.execute(
            lectures.insert().values(
                course_id=course_id,
                lecture_id=lecture_id,
                created_at=time.time(),
            )
        )


def _ensure_column(table_name: str, column_name: str, column_type: str) -> None:
    with engine.begin() as conn:
        rows = conn.execute(
            text(f"PRAGMA table_info({table_name})")
        ).fetchall()
        existing = {r[1] for r in rows}
        if column_name in existing:
            return
        conn.execute(
            text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        )
