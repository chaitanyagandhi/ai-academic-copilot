from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from app.services.store import course_store
from app.services.llm import generate_answer_with_openai
from app.services.citation_guard import needs_fix, all_citations_valid
from app.services.llm import generate_answer_with_openai, fix_citations_with_openai
from app.services.question_log import log_question
from app.services.memory import add_turn, get_recent_turns
from app.services.mastery import extract_concepts, update_student_mastery




router = APIRouter()

Mode = Literal["normal", "simple", "practice"]

class ChatRequest(BaseModel):
    course_id: str
    lecture_id: Optional[str] = None
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
    memory = get_recent_turns(
        req.course_id,
        req.user_id,
        lecture_id=req.lecture_id,
        limit=6,
    )
    confusion = log_question(req.course_id, req.user_id, req.message, lecture_id=req.lecture_id)
    add_turn(
        req.course_id,
        req.user_id,
        role="user",
        content=req.message,
        lecture_id=req.lecture_id,
    )
    hits = course_store.search(
        req.course_id,
        req.message,
        k=5,
        lecture_id=req.lecture_id,
    )
    allowed_ids = {h.chunk_id for h in hits}


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

    # Update student mastery from question + retrieved context
    concept_texts = [req.message] + [h.text for h in hits[:3]]
    concepts = extract_concepts(concept_texts)
    update_student_mastery(
        req.course_id,
        req.user_id,
        concepts,
        confusion,
        lecture_id=req.lecture_id,
    )

    llm_answer = generate_answer_with_openai(req.message, contexts, req.mode, memory_turns=memory)

    # If LLM is enabled, enforce citation rules
    if llm_answer is not None:
        if needs_fix(llm_answer, allowed_ids):
            repaired = fix_citations_with_openai(llm_answer, contexts)
            if repaired is not None and all_citations_valid(repaired, allowed_ids):
                answer = repaired
                note = "LLM enabled: answer repaired to enforce valid citations"
            else:
                # fallback
                answer = synthesize_answer(req.message, [h.text for h in hits], req.mode)
                note = "LLM enabled but citation validation failed: fallback used"
        else:
            answer = llm_answer
            note = "LLM enabled: RAG (retrieve + generate)"
    else:
        answer = synthesize_answer(req.message, [h.text for h in hits], req.mode)
        note = "LLM not configured: retrieval + template fallback"


    add_turn(
        req.course_id,
        req.user_id,
        role="assistant",
        content=answer,
        lecture_id=req.lecture_id,
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
        note=note,
    )


@router.get("/memory")
def get_memory(course_id: str, user_id: str, lecture_id: str | None = None):
    turns = get_recent_turns(course_id, user_id, lecture_id=lecture_id, limit=6)
    return {
        "course_id": course_id,
        "user_id": user_id,
        "lecture_id": lecture_id,
        "turns": turns,
    }
