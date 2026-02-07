from typing import List, Tuple
from pypdf import PdfReader

def extract_pdf_text_by_page(file_path: str) -> List[Tuple[int, str]]:
    """
    Returns a list of (page_number, text).
    page_number starts at 1.
    """
    reader = PdfReader(file_path)
    out: List[Tuple[int, str]] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        out.append((i + 1, text))

    return out
