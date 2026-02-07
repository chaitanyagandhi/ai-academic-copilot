from typing import List, Dict, Any
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def _choose_k(n: int) -> int:
    # Simple heuristic for MVP
    if n < 6:
        return 1
    if n < 10:
        return 2
    if n < 18:
        return 3
    return 4


def cluster_questions(questions: List[dict]) -> Dict[str, Any]:
    """
    Input: list of dicts like {"user_id":..., "question":..., "timestamp":...}
    Output: clusters with keywords + grouped questions
    """
    texts = [q["question"] for q in questions]
    n = len(texts)

    if n == 0:
        return {"k": 0, "clusters": []}
    if n < 6:
        # Not enough data: return one cluster
        return {
            "k": 1,
            "clusters": [
                {
                    "cluster_id": 0,
                    "keywords": [],
                    "questions": questions
                }
            ],
        }

    k = _choose_k(n)

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    # Group questions by cluster
    grouped: Dict[int, List[dict]] = {}
    for q, label in zip(questions, labels):
        grouped.setdefault(int(label), []).append(q)

    # Compute top keywords per cluster using centroid terms
    feature_names = vec.get_feature_names_out()
    centers = km.cluster_centers_

    clusters_out = []
    for cid in sorted(grouped.keys()):
        center = centers[cid]
        top_idx = center.argsort()[::-1][:6]
        keywords = [feature_names[i] for i in top_idx if center[i] > 0]

        avg_confusion = sum(
            q.get("confusion", 0.0) for q in grouped[cid]
        ) / max(len(grouped[cid]), 1)

        clusters_out.append({
            "cluster_id": cid,
            "keywords": keywords,
            "count": len(grouped[cid]),
            "avg_confusion": round(avg_confusion, 3),
            "questions": grouped[cid],
        })

    # Sort clusters by size desc (nice for dashboard)
    clusters_out.sort(key=lambda c: c["avg_confusion"], reverse=True)

    return {"k": k, "clusters": clusters_out}
