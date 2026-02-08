from typing import Dict

from app.services.confusion_model import predict_confusion

CONFUSION_KEYWORDS = [
    "why",
    "how",
    "confused",
    "not clear",
    "doesn't",
    "does not",
    "what happens",
    "difference",
    "help"
]

def compute_confusion(question: str) -> float:
    """
    Returns a confusion score between 0 and 1.
    Uses ML model if available; falls back to heuristic.
    """
    model_score = predict_confusion(question)
    if model_score is not None:
        return max(0.0, min(model_score, 1.0))

    q = question.lower()
    score = 0.0

    # Keyword-based signal
    for kw in CONFUSION_KEYWORDS:
        if kw in q:
            score += 0.15

    # Length-based signal
    if len(q) > 80:
        score += 0.2
    if len(q) > 120:
        score += 0.2

    return min(score, 1.0)
