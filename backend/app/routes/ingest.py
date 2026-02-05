from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chunking import chunk_text

router = APIRouter()

class IngestRequest(BaseModel):
    course_id: str
    source_name: str
    text: str

@router.post("/")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    first_preview = chunks[0][:120] if chunks else ""

    return {
        "course_id": req.course_id,
        "source_name": req.source_name,
        "chunks_added": len(chunks),
        "first_chunk_preview": first_preview,
    }
