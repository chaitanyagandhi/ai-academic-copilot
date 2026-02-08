from typing import Dict, Optional, List

from app.services.question_cluster import cluster_questions
from app.services.question_log import get_questions
from app.services.store import course_store
from app.services.llm import generate_recommendations_with_openai, fix_citations_with_openai
from app.services.citation_guard import needs_fix, all_citations_valid


def _cluster_summary(cluster: Dict) -> str:
    keywords = ", ".join(cluster.get("keywords", []))
    examples = [q.get("question", "") for q in cluster.get("questions", [])[:3]]
    examples_block = "\n".join(f"- {e}" for e in examples if e)
    return (
        f"Cluster keywords: {keywords}\n"
        f"Avg confusion: {cluster.get('avg_confusion')}\n"
        f"Example questions:\n{examples_block}"
    )


def generate_recommendations(
    course_id: str,
    lecture_id: Optional[str] = None,
) -> Dict:
    qs = get_questions(course_id, lecture_id=lecture_id)
    if not qs:
        return {
            "status": "no_questions",
            "recommendations": "No questions found yet for this course.",
            "citations": [],
        }

    clusters = cluster_questions(qs).get("clusters", [])
    if not clusters:
        return {
            "status": "no_clusters",
            "recommendations": "Not enough questions to form clusters yet.",
            "citations": [],
        }

    top = clusters[0]
    summary = _cluster_summary(top)

    query = " ".join(top.get("keywords", [])[:5]) or " ".join(
        q.get("question", "") for q in top.get("questions", [])[:2]
    )
    hits = course_store.search(course_id, query, k=5, lecture_id=lecture_id)
    contexts = [f"[{h.source_name} | {h.chunk_id}]\n{h.text}" for h in hits]
    allowed_ids = {h.chunk_id for h in hits}

    recs = generate_recommendations_with_openai(summary, contexts)
    if recs is not None:
        if needs_fix(recs, allowed_ids):
            repaired = fix_citations_with_openai(recs, contexts)
            if repaired is not None and all_citations_valid(repaired, allowed_ids):
                recs = repaired
            else:
                recs = None

    if recs is None:
        fallback = (
            "Consider revisiting the following topics and provide a worked example: "
            f"{', '.join(top.get('keywords', [])[:6])}."
        )
        recs = fallback

    return {
        "status": "ok",
        "cluster_id": top.get("cluster_id"),
        "cluster_keywords": top.get("keywords", []),
        "avg_confusion": top.get("avg_confusion"),
        "recommendations": recs,
        "citations": [
            {
                "source_name": h.source_name,
                "chunk_id": h.chunk_id,
            }
            for h in hits
        ],
    }
