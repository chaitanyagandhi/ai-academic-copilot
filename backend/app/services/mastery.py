import re
import time
from typing import List, Optional

from sqlalchemy import select
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.services.db import db_conn, ensure_course, ensure_lecture, student_concepts


_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def extract_concepts(texts: List[str], max_terms: int = 6) -> List[str]:
    """
    Very lightweight concept extractor: top frequent non-stopwords.
    """
    counts = {}
    for t in texts:
        for m in _TOKEN_RE.findall(t.lower()):
            if m in ENGLISH_STOP_WORDS:
                continue
            counts[m] = counts.get(m, 0) + 1

    if not counts:
        return []

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:max_terms]]


def update_student_mastery(
    course_id: str,
    user_id: str,
    concepts: List[str],
    confusion: float,
    lecture_id: Optional[str] = None,
) -> None:
    if not concepts:
        return

    with db_conn() as conn:
        ensure_course(conn, course_id)
        ensure_lecture(conn, course_id, lecture_id)

        for concept in concepts:
            row = conn.execute(
                select(
                    student_concepts.c.id,
                    student_concepts.c.count,
                    student_concepts.c.confusion_sum,
                )
                .where(student_concepts.c.course_id == course_id)
                .where(student_concepts.c.user_id == user_id)
                .where(student_concepts.c.concept == concept)
                .where(student_concepts.c.lecture_id == lecture_id)
            ).first()

            now = time.time()
            if row is None:
                conn.execute(
                    student_concepts.insert().values(
                        course_id=course_id,
                        lecture_id=lecture_id,
                        user_id=user_id,
                        concept=concept,
                        count=1,
                        confusion_sum=confusion,
                        last_updated=now,
                    )
                )
            else:
                new_count = int(row[1]) + 1
                new_sum = float(row[2]) + confusion
                conn.execute(
                    student_concepts.update()
                    .where(student_concepts.c.id == row[0])
                    .values(
                        count=new_count,
                        confusion_sum=new_sum,
                        last_updated=now,
                    )
                )
