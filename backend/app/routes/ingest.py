from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class IngestRequest(BaseModel):
    course_id: str
    source_name: str
    text: str

@router.post("/")
def ingest(req: IngestRequest):
    return {
        "course_id": req.course_id,
        "source_name": req.source_name,
        "text_length": len(req.text),
    }
