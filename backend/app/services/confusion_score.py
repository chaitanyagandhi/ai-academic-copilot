from typing import Dict

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
    Simple heuristic for MVP.
    """
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
