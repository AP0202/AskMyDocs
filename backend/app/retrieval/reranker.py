"""
Reranking stage: re-score the (query, chunk) pairs that survived hybrid
fusion using a cross-encoder, which reads the query and chunk *together*
and is much more precise than the bi-encoder / BM25 scores used for
first-stage retrieval (which score them independently for speed).

Like the vector backend, this degrades gracefully to a lexical-overlap
reranker if sentence-transformers / the cross-encoder weights aren't
available offline, so the whole pipeline still runs in CI / air-gapped
environments -- just with weaker reranking quality.
"""

import logging
import re
from typing import List, Tuple

from app.config import CROSS_ENCODER_MODEL_NAME, RERANK_BACKEND, FINAL_TOP_K
from app.ingest import Chunk
from app.retrieval.bm25_search import tokenize

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self.model.predict(pairs)]


class LexicalOverlapReranker:
    """
    Offline fallback reranker. Not a real cross-encoder, but gives a
    query-aware second-pass score: weighted token overlap (Jaccard-ish, with
    a bonus for exact substring / phrase matches) so the pipeline's
    *structure* (retrieve -> fuse -> rerank -> generate) still holds even
    without downloading a model.
    """

    name = "lexical_overlap"

    def score(self, query: str, texts: List[str]) -> List[float]:
        q_tokens = set(tokenize(query))
        q_lower = query.lower()
        scores = []
        for text in texts:
            t_tokens = set(tokenize(text))
            if not q_tokens or not t_tokens:
                scores.append(0.0)
                continue
            overlap = len(q_tokens & t_tokens) / len(q_tokens)
            phrase_bonus = 0.25 if q_lower in text.lower() else 0.0
            scores.append(overlap + phrase_bonus)
        return scores


def build_reranker():
    if RERANK_BACKEND == "cross_encoder":
        try:
            reranker = CrossEncoderReranker(CROSS_ENCODER_MODEL_NAME)
            logger.info("Reranker backend: cross-encoder (%s)", CROSS_ENCODER_MODEL_NAME)
            return reranker
        except Exception as exc:  # noqa: BLE001 - intentional fallback
            logger.warning(
                "Falling back to offline lexical-overlap reranker "
                "(cross-encoder unavailable: %s)",
                exc,
            )
    return LexicalOverlapReranker()


class Reranker:
    """Thin wrapper exposing a stable `rerank()` API to the rest of the app."""

    def __init__(self):
        self._impl = build_reranker()

    @property
    def backend_name(self) -> str:
        return self._impl.name

    def rerank(
        self, query: str, candidates: List[Tuple[Chunk, dict]], top_k: int = FINAL_TOP_K
    ) -> List[Tuple[Chunk, dict]]:
        if not candidates:
            return []
        texts = [c.text for c, _ in candidates]
        scores = self._impl.score(query, texts)
        enriched = []
        for (chunk, debug), score in zip(candidates, scores):
            debug = dict(debug)
            debug["rerank_score"] = score
            enriched.append((chunk, debug))
        enriched.sort(key=lambda pair: pair[1]["rerank_score"], reverse=True)
        return enriched[:top_k]
