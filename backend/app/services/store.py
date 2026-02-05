from dataclasses import dataclass
from typing import Dict, List
import uuid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class StoredChunk:
    chunk_id: str
    source_name: str
    text: str


class CourseStore:
    """
    v1 storage: in-memory per course_id (fine for MVP).
    Later we’ll persist to DB + vector index.
    """
    def __init__(self):
        self._chunks_by_course: Dict[str, List[StoredChunk]] = {}
        self._vectorizers: Dict[str, TfidfVectorizer] = {}
        self._matrices: Dict[str, object] = {}

    def add_chunks(self, course_id: str, source_name: str, chunks: List[str]) -> int:
        arr = self._chunks_by_course.setdefault(course_id, [])
        for c in chunks:
            arr.append(
                StoredChunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    source_name=source_name,
                    text=c,
                )
            )
        self._rebuild_index(course_id)
        return len(chunks)

    def _rebuild_index(self, course_id: str) -> None:
        docs = [c.text for c in self._chunks_by_course.get(course_id, [])]
        if not docs:
            return
        vec = TfidfVectorizer(stop_words="english", max_features=40000)
        mat = vec.fit_transform(docs)
        self._vectorizers[course_id] = vec
        self._matrices[course_id] = mat

    def search(self, course_id: str, query: str, k: int = 5) -> List[StoredChunk]:
        chunks = self._chunks_by_course.get(course_id, [])
        if not chunks:
            return []

        vec = self._vectorizers.get(course_id)
        mat = self._matrices.get(course_id)
        if vec is None or mat is None:
            self._rebuild_index(course_id)
            vec = self._vectorizers.get(course_id)
            mat = self._matrices.get(course_id)

        qv = vec.transform([query])
        sims = cosine_similarity(qv, mat).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [chunks[i] for i in top_idx]


course_store = CourseStore()
