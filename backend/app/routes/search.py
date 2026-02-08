from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from app.services.store import course_store

router = APIRouter()


class SearchRequest(BaseModel):
    course_id: str
    lecture_id: Optional[str] = None
    query: str
    k: int = Field(5, ge=1, le=10)


@router.post("/")
def search(req: SearchRequest):
    hits = course_store.search_with_scores(
        req.course_id,
        req.query,
        k=req.k,
        lecture_id=req.lecture_id,
    )
    return {
        "course_id": req.course_id,
        "lecture_id": req.lecture_id,
        "query": req.query,
        "results": [
            {
                "chunk_id": h.chunk_id,
                "source_name": h.source_name,
                "preview": h.text[:220] + ("..." if len(h.text) > 220 else ""),
                "scores": {
                    "tfidf": round(tf, 4),
                    "embedding": round(em, 4),
                    "hybrid": round(hy, 4),
                },
            }
            for (h, tf, em, hy) in hits
        ],
    }
