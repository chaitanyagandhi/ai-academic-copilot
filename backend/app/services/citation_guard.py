import re
from typing import List, Set, Tuple

# Matches: [source_name | chunk_id]
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\|\s*([a-zA-Z0-9]{6,12})\]")

def extract_citations(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, chunk_id) citations found in the answer.
    """
    out = []
    for m in CITATION_PATTERN.finditer(text):
        source = m.group(1).strip()
        chunk_id = m.group(2).strip()
        out.append((source, chunk_id))
    return out

def has_any_citation(text: str) -> bool:
    return bool(CITATION_PATTERN.search(text))

def all_citations_valid(answer: str, allowed_chunk_ids: Set[str]) -> bool:
    cites = extract_citations(answer)
    if not cites:
        return False
    for _, cid in cites:
        if cid not in allowed_chunk_ids:
            return False
    return True

def needs_fix(answer: str, allowed_chunk_ids: Set[str]) -> bool:
    """
    We require:
    - at least one citation
    - all cited chunk_ids must be from retrieved chunks
    """
    if not has_any_citation(answer):
        return True
    if not all_citations_valid(answer, allowed_chunk_ids):
        return True
    return False
