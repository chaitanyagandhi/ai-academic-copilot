from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

from app.services.store import course_store

router = APIRouter()


class SearchRequest(BaseModel):
    course_id: str
    query: str
    k: int = Field(5, ge=1, le=10)


@router.post("/")
def search(req: SearchRequest):
    hits = course_store.search(req.course_id, req.query, k=req.k)
    return {
        "course_id": req.course_id,
        "query": req.query,
        "results": [
            {
                "chunk_id": h.chunk_id,
                "source_name": h.source_name,
                "preview": h.text[:220] + ("..." if len(h.text) > 220 else ""),
            }
            for h in hits
        ],
    }
