import os
from typing import List, Optional, Dict

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def generate_answer_with_openai(
    question: str,
    contexts: List[str],
    mode: str,
    memory_turns: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
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
    memory_block = ""
    if memory_turns:
        lines = []
        for t in memory_turns[-6:]:
            lines.append(f"{t['role'].capitalize()}: {t['content']}")
        memory_block = "\n".join(lines)

    system = (
    "You are an academic tutor.\n"
    "Use ONLY the provided COURSE CONTEXT.\n"
    "The recent conversation is for disambiguation only; do not use it as factual source.\n"
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
    f"RECENT CONVERSATION (do not cite):\n{memory_block}\n\n"
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


def generate_recommendations_with_openai(
    cluster_summary: str,
    contexts: List[str],
) -> Optional[str]:
    if not has_openai_key():
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI()
    context_block = "\n\n".join(contexts[:4])

    system = (
        "You are an expert teaching assistant.\n"
        "Use ONLY the provided COURSE CONTEXT.\n"
        "Produce 2-4 actionable teaching recommendations for the instructor.\n"
        "Every recommendation sentence MUST include an inline citation: [source_name | chunk_id].\n"
        "Do NOT use any knowledge outside the context."
    )

    user = (
        f"CLUSTER SUMMARY:\n{cluster_summary}\n\n"
        f"COURSE CONTEXT:\n{context_block}\n\n"
        "Return recommendations as short bullet points."
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
