from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from app.services.store import course_store
from app.services.llm import generate_answer_with_openai


router = APIRouter()

Mode = Literal["normal", "simple", "practice"]

class ChatRequest(BaseModel):
    course_id: str
    user_id: str
    message: str
    mode: Mode = Field("normal")

class Citation(BaseModel):
    source_name: str
    chunk_id: str
    preview: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    note: Optional[str] = None


def synthesize_answer(question: str, contexts: List[str], mode: str) -> str:
    # v1: No LLM yet. We still produce a helpful, grounded response.
    if not contexts:
        return (
            "I couldn’t find anything in the course content that matches your question yet. "
            "Try rephrasing, or ingest more lecture text/slides."
        )

    top_context = contexts[0]

    if mode == "simple":
        return (
            "Here’s a simpler explanation based on your course material:\n\n"
            f"{top_context}\n\n"
            "If you tell me what part is confusing (definition, intuition, or example), I’ll focus there."
        )

    if mode == "practice":
        return (
            "Here are practice prompts based on the relevant lecture section:\n\n"
            f"{top_context}\n\n"
            "Try these:\n"
            "1) Summarize the idea in 1–2 sentences.\n"
            "2) Give one example and one non-example.\n"
            "3) What assumption is required for this to work?\n"
        )

    # normal
    return (
        "Answer grounded in your lecture material:\n\n"
        f"{top_context}\n\n"
        "Want a step-by-step walkthrough or an example?"
    )


@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    hits = course_store.search(req.course_id, req.message, k=5)

    citations = [
        Citation(
            source_name=h.source_name,
            chunk_id=h.chunk_id,
            preview=h.text[:220] + ("..." if len(h.text) > 220 else ""),
        )
        for h in hits
    ]

    contexts = [
    f"[{h.source_name} | {h.chunk_id}]\n{h.text}"
    for h in hits
    ]
    llm_answer = generate_answer_with_openai(req.message, contexts, req.mode)
    if llm_answer is not None:
        answer = llm_answer
        note = "LLM enabled: RAG (retrieve + generate)"
    else:
        answer = synthesize_answer(req.message, contexts, req.mode)
        note = "LLM not configured: retrieval + template fallback"

    return ChatResponse(
        answer=answer,
        citations=citations,
        note=note,
    )
