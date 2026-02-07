from fastapi import APIRouter
from app.services.question_log import get_questions

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
