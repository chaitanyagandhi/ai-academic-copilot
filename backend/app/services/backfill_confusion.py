from typing import List, Tuple, Optional

from sqlalchemy import select, bindparam

from app.services.db import db_conn, questions
from app.services.confusion_score import compute_confusion


def _fetch_questions(batch_size: int = 200, last_id: Optional[int] = None) -> List[Tuple[int, str]]:
    with db_conn() as conn:
        stmt = select(questions.c.id, questions.c.question)
        if last_id is not None:
            stmt = stmt.where(questions.c.id > last_id)
        stmt = stmt.order_by(questions.c.id.asc()).limit(batch_size)
        rows = conn.execute(stmt).fetchall()
    return [(r[0], r[1]) for r in rows]


def backfill_confusion(batch_size: int = 200, last_id: Optional[int] = None) -> Tuple[int, Optional[int]]:
    """
    Recompute confusion scores using current scoring (ML-first).
    Returns (number of rows updated, last_id processed).
    """
    batch = _fetch_questions(batch_size=batch_size, last_id=last_id)
    if not batch:
        return 0, last_id

    rows = []
    for qid, text in batch:
        rows.append({"b_id": qid, "b_confusion": compute_confusion(text)})

    with db_conn() as conn:
        conn.execute(
            questions.update()
            .where(questions.c.id == bindparam("b_id"))
            .values(confusion=bindparam("b_confusion")),
            rows,
        )

    return len(rows), batch[-1][0]
