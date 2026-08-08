"""
Hybrid retrieval: fuse BM25 (lexical) and vector (semantic) search results.

We use Reciprocal Rank Fusion (RRF) rather than a raw weighted score sum,
because BM25 and cosine-similarity scores live on different, incompatible
scales -- RRF only needs each retriever's *rank order*, which makes fusion
robust without any score-normalization tuning.

    RRF(chunk) = sum over retrievers of  1 / (k + rank_in_that_retriever)
"""

from typing import Dict, List, Tuple

from app.config import BM25_TOP_K, VECTOR_TOP_K, FUSION_TOP_K, RRF_K
from app.ingest import Chunk
from app.retrieval.bm25_search import BM25Index
from app.retrieval.vector_search import VectorIndex


class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.vector = VectorIndex(chunks)

    def retrieve(
        self, query: str, top_k: int = FUSION_TOP_K, chunk_filter=None
    ) -> List[Tuple[Chunk, dict]]:
        """
        Returns a list of (chunk, debug_scores) tuples, ranked by fused score,
        where debug_scores holds the raw bm25/vector scores + ranks for
        transparency (useful for the eval pipeline and for showing the user
        "why" a chunk was retrieved).
        """
        bm25_hits = self.bm25.search(query, top_k=BM25_TOP_K)
        vector_hits = self.vector.search(query, top_k=VECTOR_TOP_K)

        if chunk_filter is not None:
            bm25_hits = [(c, s) for c, s in bm25_hits if chunk_filter(c)]
            vector_hits = [(c, s) for c, s in vector_hits if chunk_filter(c)]

        rrf_scores: Dict[str, float] = {}
        debug: Dict[str, dict] = {}
        chunk_lookup: Dict[str, Chunk] = {}

        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            chunk_lookup[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            debug.setdefault(chunk.chunk_id, {})["bm25_score"] = score
            debug[chunk.chunk_id]["bm25_rank"] = rank

        for rank, (chunk, score) in enumerate(vector_hits, start=1):
            chunk_lookup[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            debug.setdefault(chunk.chunk_id, {})["vector_score"] = score
            debug[chunk.chunk_id]["vector_rank"] = rank

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for cid in ranked_ids[:top_k]:
            d = debug[cid]
            d["rrf_score"] = rrf_scores[cid]
            results.append((chunk_lookup[cid], d))
        return results
