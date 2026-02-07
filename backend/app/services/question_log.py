from collections import defaultdict
from typing import List, Dict
import time

from app.services.confusion_score import compute_confusion


# Simple in-memory store (course_id -> list of questions)
_question_log: Dict[str, List[dict]] = defaultdict(list)

def log_question(course_id: str, user_id: str, question: str):
    _question_log[course_id].append({
        "user_id": user_id,
        "question": question,
        "confusion": compute_confusion(question),
        "timestamp": time.time()
    })

def get_questions(course_id: str) -> List[dict]:
    return _question_log.get(course_id, [])
