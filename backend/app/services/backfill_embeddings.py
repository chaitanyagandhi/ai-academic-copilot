import json
import time
from typing import List, Tuple

from sqlalchemy import select

from app.services.db import db_conn, chunks as chunks_table, chunk_embeddings
from app.services.embeddings import embed_texts, DEFAULT_EMBEDDING_MODEL


def _fetch_missing_chunks(batch_size: int = 64) -> List[Tuple[str, str]]:
    with db_conn() as conn:
        stmt = (
            select(chunks_table.c.chunk_id, chunks_table.c.text)
            .select_from(
                chunks_table.outerjoin(
                    chunk_embeddings,
                    chunk_embeddings.c.chunk_id == chunks_table.c.chunk_id,
                )
            )
            .where(chunk_embeddings.c.chunk_id.is_(None))
            .limit(batch_size)
        )
        rows = conn.execute(stmt).fetchall()
    return [(r[0], r[1]) for r in rows]


def backfill_embeddings(batch_size: int = 64) -> int:
    """
    Backfill missing embeddings. Returns number of chunks embedded.
    """
    missing = _fetch_missing_chunks(batch_size=batch_size)
    if not missing:
        return 0

    texts = [t for _, t in missing]
    vectors = embed_texts(texts)
    if vectors is None:
        return 0

    rows = []
    now = time.time()
    for (chunk_id, _), vec in zip(missing, vectors):
        rows.append(
            {
                "chunk_id": chunk_id,
                "model": DEFAULT_EMBEDDING_MODEL,
                "vector_json": json.dumps(vec),
                "created_at": now,
            }
        )

    with db_conn() as conn:
        conn.execute(chunk_embeddings.insert(), rows)

    return len(rows)
