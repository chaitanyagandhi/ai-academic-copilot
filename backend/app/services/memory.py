import time
from typing import List, Optional, Dict

from sqlalchemy import select

from app.services.db import db_conn, ensure_course, ensure_lecture, conversation_turns


def add_turn(
    course_id: str,
    user_id: str,
    role: str,
    content: str,
    lecture_id: Optional[str] = None,
    max_turns: int = 6,
) -> None:
    with db_conn() as conn:
        ensure_course(conn, course_id)
        ensure_lecture(conn, course_id, lecture_id)
        conn.execute(
            conversation_turns.insert().values(
                course_id=course_id,
                lecture_id=lecture_id,
                user_id=user_id,
                role=role,
                content=content,
                timestamp=time.time(),
            )
        )

        # Keep only the most recent max_turns per user/course/lecture.
        stmt = (
            select(conversation_turns.c.id)
            .where(conversation_turns.c.course_id == course_id)
            .where(conversation_turns.c.user_id == user_id)
            .where(conversation_turns.c.lecture_id == lecture_id)
            .order_by(conversation_turns.c.timestamp.desc())
            .offset(max_turns)
        )
        old_ids = [r[0] for r in conn.execute(stmt).fetchall()]
        if old_ids:
            conn.execute(
                conversation_turns.delete().where(conversation_turns.c.id.in_(old_ids))
            )


def get_recent_turns(
    course_id: str,
    user_id: str,
    lecture_id: Optional[str] = None,
    limit: int = 6,
) -> List[Dict[str, str]]:
    with db_conn() as conn:
        stmt = (
            select(
                conversation_turns.c.role,
                conversation_turns.c.content,
            )
            .where(conversation_turns.c.course_id == course_id)
            .where(conversation_turns.c.user_id == user_id)
            .where(conversation_turns.c.lecture_id == lecture_id)
            .order_by(conversation_turns.c.timestamp.desc())
            .limit(limit)
        )
        rows = conn.execute(stmt).fetchall()

    # Return in chronological order
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]
