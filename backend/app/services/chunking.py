import re
from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    # Normalize whitespace (makes chunking predictable)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks
