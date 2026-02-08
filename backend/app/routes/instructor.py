from fastapi import APIRouter
from app.services.question_log import get_questions
from app.services.question_cluster import cluster_questions
from app.services.confusion_trend import compute_confusion_trend



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

