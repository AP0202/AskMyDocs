"""
Dense/vector retrieval over document chunks.

Two backends are supported behind one interface:

* "sbert"  - a real sentence-transformers bi-encoder (semantic embeddings).
             Requires internet on first run to download model weights.
* "tfidf"  - TF-IDF + truncated SVD (a small local "LSA" embedding space).
             100% offline, no downloads, used automatically as a fallback
             so the app still runs in air-gapped / CI environments.

Set EMBEDDING_BACKEND=tfidf in your .env to force offline mode.
"""

import logging
from typing import List, Tuple

import numpy as np

from app.config import EMBEDDING_BACKEND, SBERT_MODEL_NAME
from app.ingest import Chunk

logger = logging.getLogger(__name__)


class _SBertBackend:
    name = "sbert"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))

    def transform(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class _TfidfBackend:
    """Offline fallback: TF-IDF followed by truncated SVD (a mini LSA space)."""

    name = "tfidf"

    def __init__(self, n_components: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.n_components = n_components
        self.svd = None
        self._svd_cls = TruncatedSVD

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.n_components, max(2, tfidf.shape[1] - 1), tfidf.shape[0] - 1)
        n_components = max(n_components, 2)
        self.svd = self._svd_cls(n_components=n_components, random_state=42)
        emb = self.svd.fit_transform(tfidf)
        return self._normalize(emb)

    def transform(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        emb = self.svd.transform(tfidf)
        return self._normalize(emb)

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def _build_backend():
    """Try the configured backend, gracefully fall back to TF-IDF."""
    if EMBEDDING_BACKEND == "sbert":
        try:
            backend = _SBertBackend(SBERT_MODEL_NAME)
            logger.info("Vector backend: sentence-transformers (%s)", SBERT_MODEL_NAME)
            return backend
        except Exception as exc:  # noqa: BLE001 - broad by design, this is a fallback
            logger.warning(
                "Falling back to offline TF-IDF vector backend "
                "(sentence-transformers unavailable: %s)",
                exc,
            )
    return _TfidfBackend()


class VectorIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.backend = _build_backend()
        texts = [f"{c.section}. {c.text}" for c in chunks]
        self.embeddings = (
            self.backend.fit_transform(texts) if texts else np.zeros((0, 1))
        )

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        query_vec = self.backend.transform([query])[0]
        sims = self.embeddings @ query_vec
        ranked_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in ranked_idx if sims[i] > 0]
