from typing import List, Optional
import time

from sqlalchemy import select

from app.services.confusion_score import compute_confusion
from app.services.db import db_conn, ensure_course, ensure_lecture, questions


def log_question(
    course_id: str,
    user_id: str,
    question: str,
    lecture_id: Optional[str] = None,
):
    confusion = compute_confusion(question)
    ts = time.time()
    with db_conn() as conn:
        ensure_course(conn, course_id)
        ensure_lecture(conn, course_id, lecture_id)
        conn.execute(
            questions.insert().values(
                course_id=course_id,
                lecture_id=lecture_id,
                user_id=user_id,
                question=question,
                confusion=confusion,
                timestamp=ts,
            )
        )


def get_questions(course_id: str, lecture_id: Optional[str] = None) -> List[dict]:
    with db_conn() as conn:
        stmt = (
            select(
                questions.c.user_id,
                questions.c.question,
                questions.c.lecture_id,
                questions.c.confusion,
                questions.c.timestamp,
            )
            .where(questions.c.course_id == course_id)
            .order_by(questions.c.timestamp.asc())
        )
        if lecture_id:
            stmt = stmt.where(questions.c.lecture_id == lecture_id)
        rows = conn.execute(stmt).fetchall()

    return [
        {
            "user_id": r[0],
            "question": r[1],
            "lecture_id": r[2],
            "confusion": r[3],
            "timestamp": r[4],
        }
        for r in rows
    ]
