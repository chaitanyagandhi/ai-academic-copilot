from fastapi import APIRouter
from app.services.question_log import get_questions
from app.services.question_cluster import cluster_questions
from app.services.confusion_trend import compute_confusion_trend
from app.services.db import db_conn, student_concepts
from sqlalchemy import select
from app.services.alerts import detect_confusion_spike, create_alert, recent_alert_exists, list_alerts, debug_alert_metrics
from app.services.recommendations import generate_recommendations



router = APIRouter()

@router.get("/questions")
def get_course_questions(course_id: str, lecture_id: str | None = None):
    """
    Instructor endpoint:
    Returns all student questions for a course.
    """
    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        "total_questions": len(get_questions(course_id, lecture_id=lecture_id)),
        "questions": get_questions(course_id, lecture_id=lecture_id)
    }

@router.get("/clusters")
def get_question_clusters(course_id: str, lecture_id: str | None = None):
    """
    Instructor endpoint:
    Groups student questions into topic clusters.
    """
    qs = get_questions(course_id, lecture_id=lecture_id)
    result = cluster_questions(qs)
    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        "total_questions": len(qs),
        **result
    }

@router.get("/confusion_trend")
def get_confusion_trend(course_id: str, lecture_id: str | None = None):
    qs = get_questions(course_id, lecture_id=lecture_id)
    trend = compute_confusion_trend(qs)
    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        "points": trend
    }


@router.get("/alerts")
def get_alerts(
    course_id: str,
    lecture_id: str | None = None,
    avg_threshold: float = 0.45,
    slope_threshold: float = 0.08,
    include_history: bool = True,
    bucket_minutes: float = 0.25,
    min_questions: int = 6,
):
    detected = detect_confusion_spike(
        course_id=course_id,
        lecture_id=lecture_id,
        avg_threshold=avg_threshold,
        slope_threshold=slope_threshold,
        bucket_minutes=bucket_minutes,
        min_questions=min_questions,
    )
    alerts_out = []
    if detected:
        alert, avg_conf, slope = detected
        if not recent_alert_exists(course_id, lecture_id, alert["type"]):
            create_alert(course_id, lecture_id, alert)
        alert_with_meta = dict(alert)
        alert_with_meta["avg_confusion"] = round(avg_conf, 3)
        alert_with_meta["slope"] = round(slope, 3)
        alerts_out.append(alert_with_meta)

    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        "alerts": alerts_out,
        "history": list_alerts(course_id, lecture_id) if include_history else [],
    }


@router.get("/alerts_debug")
def get_alerts_debug(
    course_id: str,
    lecture_id: str | None = None,
    bucket_minutes: float = 0.25,
    min_questions: int = 6,
):
    metrics = debug_alert_metrics(
        course_id=course_id,
        lecture_id=lecture_id,
        bucket_minutes=bucket_minutes,
        min_questions=min_questions,
    )
    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        **metrics,
    }


@router.get("/recommendations")
def get_recommendations(course_id: str, lecture_id: str | None = None):
    data = generate_recommendations(course_id, lecture_id=lecture_id)
    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        **data,
    }


@router.get("/student_mastery")
def get_student_mastery(course_id: str, user_id: str, lecture_id: str | None = None):
    with db_conn() as conn:
        stmt = (
            select(
                student_concepts.c.concept,
                student_concepts.c.count,
                student_concepts.c.confusion_sum,
                student_concepts.c.last_updated,
                student_concepts.c.lecture_id,
            )
            .where(student_concepts.c.course_id == course_id)
            .where(student_concepts.c.user_id == user_id)
        )
        if lecture_id:
            stmt = stmt.where(student_concepts.c.lecture_id == lecture_id)
        rows = conn.execute(stmt).fetchall()

    out = []
    for r in rows:
        count = int(r[1])
        avg_conf = float(r[2]) / max(count, 1)
        out.append(
            {
                "concept": r[0],
                "count": count,
                "avg_confusion": round(avg_conf, 3),
                "last_updated": r[3],
                "lecture_id": r[4],
            }
        )

    out.sort(key=lambda x: (-x["count"], -x["avg_confusion"]))
    return {
        "course_id": course_id,
        "user_id": user_id,
        "lecture_id": lecture_id,
        "concepts": out,
    }
