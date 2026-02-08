import os
from typing import List, Optional


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def embed_texts(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[List[List[float]]]:
    if not texts:
        return []
    if not has_openai_key():
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI()
        resp = client.embeddings.create(
            model=model,
            input=texts,
        )
        # API returns embeddings in the same order as inputs
        return [d.embedding for d in resp.data]
    except Exception:
        return None
