"""
BM25 lexical retrieval over document chunks.

BM25 is the "keyword" half of the hybrid retriever: it is excellent at
matching exact clinical terms (e.g. "LDL", "metformin", "hemoglobin") that
dense embeddings can sometimes blur together.
"""

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9%]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._tokenized = [tokenize(c.text + " " + c.section) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(c, float(s)) for c, s in ranked[:top_k] if s > 0]
