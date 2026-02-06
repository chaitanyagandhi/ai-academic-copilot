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
    "You are an academic tutor. Answer the student's question using ONLY the provided course context.\n"
    "Every factual statement MUST be supported by a citation.\n"
    "Use inline citations in this exact format: [source_name | chunk_id].\n"
    "If the answer is not present in the context, say you don't have enough information "
    "and suggest what additional lecture content should be ingested."
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
        "Remember: include citations after each explanation."
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
