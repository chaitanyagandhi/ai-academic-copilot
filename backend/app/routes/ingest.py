from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chunking import chunk_text
from app.services.store import course_store


router = APIRouter()

class IngestRequest(BaseModel):
    course_id: str
    source_name: str
    text: str

@router.post("/")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    added = course_store.add_chunks(req.course_id, req.source_name, chunks)
    first_preview = chunks[0][:120] if chunks else ""

    return {
        "course_id": req.course_id,
        "source_name": req.source_name,
        "chunks_added": added,
        "first_chunk_preview": first_preview,
    }
