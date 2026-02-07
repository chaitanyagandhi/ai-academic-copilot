from fastapi import APIRouter
from app.services.question_log import get_questions
from app.services.question_cluster import cluster_questions


router = APIRouter()

@router.get("/questions")
def get_course_questions(course_id: str):
    """
    Instructor endpoint:
    Returns all student questions for a course.
    """
    return {
        "course_id": course_id,
        "total_questions": len(get_questions(course_id)),
        "questions": get_questions(course_id)
    }

@router.get("/clusters")
def get_question_clusters(course_id: str):
    """
    Instructor endpoint:
    Groups student questions into topic clusters.
    """
    qs = get_questions(course_id)
    result = cluster_questions(qs)
    return {
        "course_id": course_id,
        "total_questions": len(qs),
        **result
    }

