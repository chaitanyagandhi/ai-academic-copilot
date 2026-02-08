from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import time
import uuid

import numpy as np
from sqlalchemy import func, select
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.db import (
    db_conn,
    ensure_course,
    ensure_lecture,
    documents,
    chunks as chunks_table,
    chunk_embeddings,
)
from app.services.embeddings import embed_texts, DEFAULT_EMBEDDING_MODEL


@dataclass
class StoredChunk:
    chunk_id: str
    source_name: str
    text: str


class CourseStore:
    """
    v2 storage: persisted in SQLite with in-memory TF-IDF cache.
    """
    def __init__(self):
        self._vectorizers: Dict[str, TfidfVectorizer] = {}
        self._matrices: Dict[str, object] = {}
        self._chunk_cache: Dict[str, List[StoredChunk]] = {}
        self._chunk_counts: Dict[str, int] = {}
        self._embedding_cache: Dict[str, List[Optional[List[float]]]] = {}

    def add_chunks(
        self,
        course_id: str,
        source_name: str,
        chunks: List[str],
        lecture_id: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0

        with db_conn() as conn:
            ensure_course(conn, course_id)
            ensure_lecture(conn, course_id, lecture_id)
            doc_id = conn.execute(
                documents.insert().values(
                    course_id=course_id,
                    lecture_id=lecture_id,
                    source_name=source_name,
                    created_at=time.time(),
                )
            ).inserted_primary_key[0]

            rows = [
                {
                    "document_id": doc_id,
                    "chunk_id": str(uuid.uuid4())[:8],
                    "text": c,
                    "created_at": time.time(),
                }
                for c in chunks
            ]
            conn.execute(chunks_table.insert(), rows)

            embeddings = embed_texts([r["text"] for r in rows])
            if embeddings is not None:
                emb_rows = []
                for r, vec in zip(rows, embeddings):
                    emb_rows.append(
                        {
                            "chunk_id": r["chunk_id"],
                            "model": DEFAULT_EMBEDDING_MODEL,
                            "vector_json": json.dumps(vec),
                            "created_at": time.time(),
                        }
                    )
                if emb_rows:
                    conn.execute(chunk_embeddings.insert(), emb_rows)

        self._invalidate_cache(course_id)
        return len(chunks)

    def _invalidate_cache(self, course_id: str) -> None:
        prefix = f"{course_id}::"
        keys = [k for k in self._chunk_counts.keys() if k.startswith(prefix)]
        for k in keys:
            self._vectorizers.pop(k, None)
            self._matrices.pop(k, None)
            self._chunk_cache.pop(k, None)
            self._chunk_counts.pop(k, None)
            self._embedding_cache.pop(k, None)

    def _load_chunks(self, course_id: str, lecture_id: Optional[str]) -> List[StoredChunk]:
        with db_conn() as conn:
            stmt = (
                select(
                    chunks_table.c.chunk_id,
                    documents.c.source_name,
                    chunks_table.c.text,
                )
                .select_from(chunks_table.join(documents, chunks_table.c.document_id == documents.c.id))
                .where(documents.c.course_id == course_id)
            )
            if lecture_id:
                stmt = stmt.where(documents.c.lecture_id == lecture_id)
            rows = conn.execute(stmt).fetchall()
        return [StoredChunk(chunk_id=r[0], source_name=r[1], text=r[2]) for r in rows]

    def _load_embeddings(self, course_id: str, lecture_id: Optional[str]) -> Dict[str, List[float]]:
        with db_conn() as conn:
            stmt = (
                select(
                    chunk_embeddings.c.chunk_id,
                    chunk_embeddings.c.vector_json,
                )
                .select_from(
                    chunk_embeddings.join(
                        chunks_table, chunk_embeddings.c.chunk_id == chunks_table.c.chunk_id
                    ).join(
                        documents, chunks_table.c.document_id == documents.c.id
                    )
                )
                .where(documents.c.course_id == course_id)
            )
            if lecture_id:
                stmt = stmt.where(documents.c.lecture_id == lecture_id)
            rows = conn.execute(stmt).fetchall()
        out: Dict[str, List[float]] = {}
        for r in rows:
            try:
                out[r[0]] = json.loads(r[1])
            except Exception:
                continue
        return out

    def _get_chunk_count(self, course_id: str, lecture_id: Optional[str]) -> int:
        with db_conn() as conn:
            stmt = (
                select(func.count())
                .select_from(chunks_table.join(documents, chunks_table.c.document_id == documents.c.id))
                .where(documents.c.course_id == course_id)
            )
            if lecture_id:
                stmt = stmt.where(documents.c.lecture_id == lecture_id)
            return int(conn.execute(stmt).scalar() or 0)

    def _rebuild_index(self, course_id: str, stored_chunks: List[StoredChunk]) -> None:
        docs = [c.text for c in stored_chunks]
        if not docs:
            return
        vec = TfidfVectorizer(stop_words="english", max_features=40000)
        mat = vec.fit_transform(docs)
        self._vectorizers[course_id] = vec
        self._matrices[course_id] = mat

    def search(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        lecture_id: Optional[str] = None,
    ) -> List[StoredChunk]:
        count = self._get_chunk_count(course_id, lecture_id)
        if count == 0:
            return []

        cache_key = f"{course_id}::{lecture_id or 'all'}"
        cached_count = self._chunk_counts.get(cache_key)
        if cached_count != count:
            stored_chunks = self._load_chunks(course_id, lecture_id)
            self._chunk_cache[cache_key] = stored_chunks
            self._chunk_counts[cache_key] = count
            self._rebuild_index(cache_key, stored_chunks)
            emb_map = self._load_embeddings(course_id, lecture_id)
            self._embedding_cache[cache_key] = [
                emb_map.get(c.chunk_id) for c in stored_chunks
            ]

        chunks_list = self._chunk_cache.get(cache_key, [])
        if not chunks_list:
            return []

        vec = self._vectorizers.get(cache_key)
        mat = self._matrices.get(cache_key)
        if vec is None or mat is None:
            self._rebuild_index(cache_key, chunks_list)
            vec = self._vectorizers.get(cache_key)
            mat = self._matrices.get(cache_key)

        qv = vec.transform([query])
        tfidf_sims = cosine_similarity(qv, mat).flatten()

        hybrid_sims = tfidf_sims
        emb_list = self._embedding_cache.get(cache_key, [])
        if emb_list:
            q_emb = embed_texts([query])
            if q_emb:
                q_vec = np.array(q_emb[0], dtype=np.float32)
                emb_matrix = np.array(
                    [e if e is not None else np.zeros_like(q_vec) for e in emb_list],
                    dtype=np.float32,
                )
                # cosine similarity
                denom = (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_vec) + 1e-8)
                emb_sims = (emb_matrix @ q_vec) / denom

                # normalize both to 0..1
                tf_min, tf_max = float(tfidf_sims.min()), float(tfidf_sims.max())
                em_min, em_max = float(emb_sims.min()), float(emb_sims.max())
                tf_norm = (tfidf_sims - tf_min) / (tf_max - tf_min + 1e-8)
                em_norm = (emb_sims - em_min) / (em_max - em_min + 1e-8)

                alpha = 0.6
                hybrid_sims = alpha * tf_norm + (1.0 - alpha) * em_norm

        top_idx = hybrid_sims.argsort()[::-1][:k]
        return [chunks_list[i] for i in top_idx]

    def search_with_scores(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        lecture_id: Optional[str] = None,
    ) -> List[Tuple[StoredChunk, float, float, float]]:
        """
        Returns (chunk, tfidf_score, embedding_score, hybrid_score).
        embedding_score is 0 if embeddings are unavailable.
        """
        count = self._get_chunk_count(course_id, lecture_id)
        if count == 0:
            return []

        cache_key = f"{course_id}::{lecture_id or 'all'}"
        cached_count = self._chunk_counts.get(cache_key)
        if cached_count != count:
            stored_chunks = self._load_chunks(course_id, lecture_id)
            self._chunk_cache[cache_key] = stored_chunks
            self._chunk_counts[cache_key] = count
            self._rebuild_index(cache_key, stored_chunks)
            emb_map = self._load_embeddings(course_id, lecture_id)
            self._embedding_cache[cache_key] = [
                emb_map.get(c.chunk_id) for c in stored_chunks
            ]

        chunks_list = self._chunk_cache.get(cache_key, [])
        if not chunks_list:
            return []

        vec = self._vectorizers.get(cache_key)
        mat = self._matrices.get(cache_key)
        if vec is None or mat is None:
            self._rebuild_index(cache_key, chunks_list)
            vec = self._vectorizers.get(cache_key)
            mat = self._matrices.get(cache_key)

        qv = vec.transform([query])
        tfidf_sims = cosine_similarity(qv, mat).flatten()

        emb_list = self._embedding_cache.get(cache_key, [])
        emb_sims = None
        if emb_list:
            q_emb = embed_texts([query])
            if q_emb:
                q_vec = np.array(q_emb[0], dtype=np.float32)
                emb_matrix = np.array(
                    [e if e is not None else np.zeros_like(q_vec) for e in emb_list],
                    dtype=np.float32,
                )
                denom = (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_vec) + 1e-8)
                emb_sims = (emb_matrix @ q_vec) / denom

        # normalize both to 0..1 for hybrid
        tf_min, tf_max = float(tfidf_sims.min()), float(tfidf_sims.max())
        tf_norm = (tfidf_sims - tf_min) / (tf_max - tf_min + 1e-8)

        if emb_sims is not None:
            em_min, em_max = float(emb_sims.min()), float(emb_sims.max())
            em_norm = (emb_sims - em_min) / (em_max - em_min + 1e-8)
        else:
            em_norm = np.zeros_like(tf_norm)

        alpha = 0.6
        hybrid = alpha * tf_norm + (1.0 - alpha) * em_norm

        top_idx = hybrid.argsort()[::-1][:k]
        out: List[Tuple[StoredChunk, float, float, float]] = []
        for i in top_idx:
            out.append(
                (
                    chunks_list[i],
                    float(tf_norm[i]),
                    float(em_norm[i]),
                    float(hybrid[i]),
                )
            )
        return out


course_store = CourseStore()
