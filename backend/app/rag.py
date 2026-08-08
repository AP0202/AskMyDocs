"""
RAG orchestration: retrieve -> fuse -> rerank -> generate -> enforce citations.

This module ties together the hybrid retriever, the reranker, and the LLM
layer, and is responsible for the "citation enforcement" contract: every
answer returned to the user must be traceable back to specific retrieved
chunks, and if the model produced an answer that isn't adequately cited,
we replace it with a safe, fully-grounded fallback rather than let an
ungrounded claim reach a patient.
"""

import re
import time
from dataclasses import dataclass
from typing import List, Optional

from app.config import FUSION_TOP_K, FINAL_TOP_K, MIN_CITED_SENTENCE_RATIO
from app.ingest import Chunk, Document, load_all_documents
from app.llm import generate_answer, _extractive_fallback_answer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.config import OPENAI_API_KEY
from app import metrics

DISCLAIMER = (
    "This information is generated from your own uploaded documents to help you "
    "understand them. It is not medical advice and does not replace guidance from "
    "your doctor or care team. If you have urgent symptoms, seek medical care "
    "immediately."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    marker: str
    debug: dict


class RagPipeline:
    """
    Loads all documents once at startup and builds the hybrid index + reranker.
    A single instance is shared by the FastAPI app (see app/main.py) and by
    the evaluation scripts, so eval always measures the exact same pipeline
    the API serves.
    """

    def __init__(self):
        self.documents: List[Document] = load_all_documents()
        self.chunks: List[Chunk] = [c for d in self.documents for c in d.chunks]
        self.retriever = HybridRetriever(self.chunks)
        self.reranker = Reranker()

    def get_document(self, doc_id: str) -> Optional[Document]:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        return None

    def list_documents(self) -> List[Document]:
        return self.documents

    def retrieve(
        self, question: str, document_id: Optional[str] = None, final_top_k: int = FINAL_TOP_K
    ):
        chunk_filter = None
        if document_id:
            chunk_filter = lambda c: c.doc_id == document_id  # noqa: E731

        t0 = time.perf_counter()
        fused = self.retriever.retrieve(
            question, top_k=FUSION_TOP_K, chunk_filter=chunk_filter
        )
        metrics.record_stage("hybrid_retrieval", time.perf_counter() - t0)

        t1 = time.perf_counter()
        reranked = self.reranker.rerank(question, fused, top_k=final_top_k)
        metrics.record_stage("rerank", time.perf_counter() - t1)

        return reranked  # List[(Chunk, debug_dict)]

    def answer(self, question: str, document_id: Optional[str] = None):
        request_start = time.perf_counter()

        reranked = self.retrieve(question, document_id=document_id)
        chunks = [c for c, _ in reranked]

        t2 = time.perf_counter()
        raw_answer = generate_answer(question, chunks)
        metrics.record_stage("llm_generate", time.perf_counter() - t2)

        grounded, final_answer = enforce_citations(question, raw_answer, chunks)

        metrics.REQUEST_LATENCY.observe(time.perf_counter() - request_start)
        metrics.record_request(
            grounded=grounded, llm_backend="openai" if OPENAI_API_KEY else "extractive"
        )

        citations = [
            {
                "marker": f"[{i}]",
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "section": c.section,
                "text": c.text,
                "score": reranked[i - 1][1].get("rerank_score", 0.0),
            }
            for i, c in enumerate(chunks, start=1)
        ]

        debug = {
            "bm25_top_k": len(chunks),
            "vector_backend": self.retriever.vector.backend_name,
            "rerank_backend": self.reranker.backend_name,
            "candidates": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "section": c.section,
                    **debug_scores,
                }
                for c, debug_scores in reranked
            ],
        }

        return {
            "answer": final_answer,
            "citations": citations,
            "grounded": grounded,
            "disclaimer": DISCLAIMER,
            "retrieval_debug": debug,
        }


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def enforce_citations(question: str, answer: str, chunks: List[Chunk]):
    """
    Validate that the LLM's answer is adequately grounded/cited.

    Returns (grounded: bool, final_answer: str).

    If fewer than MIN_CITED_SENTENCE_RATIO of the answer's sentences carry a
    [n] citation marker pointing at a retrieved chunk, we don't trust the
    answer's grounding and swap in the deterministic extractive fallback
    instead -- this is the CI-testable "citation enforcement" guarantee.
    """
    if not chunks:
        return False, answer

    sentences = _split_sentences(answer)
    if not sentences:
        return False, answer

    max_marker = len(chunks)
    cited = 0
    for sentence in sentences:
        markers = _CITATION_MARKER_RE.findall(sentence)
        valid = [m for m in markers if 1 <= int(m.strip("[]")) <= max_marker]
        if valid:
            cited += 1

    ratio = cited / len(sentences)
    if ratio >= MIN_CITED_SENTENCE_RATIO:
        return True, answer

    # Not adequately grounded -> fall back to a safe, fully-cited extract.
    safe_answer = _extractive_fallback_answer(question, chunks)
    safe_answer += (
        "\n\n(Note: the AI-generated response did not meet our citation "
        "requirements, so we've shown you the source passages directly instead.)"
    )
    return False, safe_answer
