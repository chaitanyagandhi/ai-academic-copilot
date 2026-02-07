import os
from typing import List, Optional

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def generate_answer_with_openai(question: str, contexts: List[str], mode: str) -> Optional[str]:
    """
    Returns a generated answer string if OpenAI is configured.
    Returns None if not configured (so we can fall back).
    """
    if not has_openai_key():
        return None

    # Import inside the function so the app still runs even if openai isn't installed
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI()

    # Keep context short-ish for now (MVP). We’ll improve later.
    context_block = "\n\n".join(contexts[:4])

    system = (
    "You are an academic tutor.\n"
    "Use ONLY the provided COURSE CONTEXT.\n"
    "Every claim MUST include an inline citation in this exact format: [source_name | chunk_id].\n"
    "If the context does not contain the answer, say: "
    "'I don't have enough information in the course content to answer that.' "
    "Then suggest what content to ingest (slides/pages/lecture transcript) that would help.\n"
    "Do NOT cite anything that is not in the context."
    )



    if mode == "simple":
        style = "Explain simply, using short sentences and one example."
    elif mode == "practice":
        style = "Give 3 practice questions and 1 short answer key."
    else:
        style = "Be clear and structured. Give an intuitive explanation and a concrete example if possible."

    user = (
    f"COURSE CONTEXT:\n{context_block}\n\n"
    f"STUDENT QUESTION:\n{question}\n\n"
    f"INSTRUCTIONS:\n{style}\n\n"
    "Important: Put citations at the end of each sentence that depends on the context.\n"
    )



    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content

def fix_citations_with_openai(original_answer: str, contexts: List[str]) -> Optional[str]:
    """
    Ask the LLM to rewrite the answer so that every sentence has valid citations.
    Returns None if OpenAI not configured.
    """
    if not has_openai_key():
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI()
    except Exception:
        return None

    context_block = "\n\n".join(contexts[:4])

    system = (
        "You are a strict editor.\n"
        "Rewrite the answer so that EVERY sentence includes an inline citation in this exact format: [source_name | chunk_id].\n"
        "You MUST ONLY use citations that appear in the provided COURSE CONTEXT labels.\n"
        "Do not introduce any new facts that aren't supported by the context.\n"
        "Keep the meaning the same but make it properly cited."
    )

    user = (
        f"COURSE CONTEXT (with allowed citation labels):\n{context_block}\n\n"
        f"ORIGINAL ANSWER:\n{original_answer}\n\n"
        "Return ONLY the corrected answer text."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )

    return resp.choices[0].message.content

