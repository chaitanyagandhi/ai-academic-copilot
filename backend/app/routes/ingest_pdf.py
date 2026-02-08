import os
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form

from app.services.pdf_extract import extract_pdf_text_by_page
from app.services.chunking import chunk_text
from app.services.store import course_store

router = APIRouter()

UPLOAD_DIR = "backend/data/uploads"

@router.post("/pdf")
def ingest_pdf(
    course_id: str = Form(...),
    lecture_id: Optional[str] = Form(None),
    source_name: str = Form(...),
    file: UploadFile = File(...),
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save file locally (temporary)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext != ".pdf":
        return {"error": "Please upload a .pdf file"}

    saved_name = f"{uuid.uuid4().hex}.pdf"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(saved_path, "wb") as f:
        f.write(file.file.read())

    # Extract page text
    pages = extract_pdf_text_by_page(saved_path)

    total_chunks = 0
    pages_ingested = 0

    for page_num, page_text in pages:
        if not page_text:
            continue

        pages_ingested += 1

        # chunk per page so citations can point to a page
        chunks = chunk_text(page_text)

        # we embed page into the source_name for citations
        page_source = f"{source_name} (page {page_num})"
        total_chunks += course_store.add_chunks(
            course_id,
            page_source,
            chunks,
            lecture_id=lecture_id,
        )

    return {
        "course_id": course_id,
        "lecture_id": lecture_id,
        "source_name": source_name,
        "pages_total": len(pages),
        "pages_ingested": pages_ingested,
        "chunks_added": total_chunks
    }
