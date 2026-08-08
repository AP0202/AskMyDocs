"""
Unit tests for the retrieval and citation-enforcement building blocks,
independent of the API layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from app.ingest import load_all_documents  # noqa: E402
from app.retrieval.bm25_search import BM25Index  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.rag import enforce_citations  # noqa: E402


def _all_chunks():
    docs = load_all_documents()
    return [c for d in docs for c in d.chunks]


def test_documents_load_and_chunk():
    docs = load_all_documents()
    assert len(docs) >= 5
    for d in docs:
        assert d.chunks, f"{d.doc_id} produced no chunks"


def test_bm25_finds_exact_keyword():
    chunks = _all_chunks()
    index = BM25Index(chunks)
    hits = index.search("metformin lactic acidosis", top_k=5)
    assert hits
    assert any("metformin" in c.text.lower() or "metformin" in c.section.lower() for c, _ in hits)


def test_hybrid_retriever_respects_document_filter():
    chunks = _all_chunks()
    retriever = HybridRetriever(chunks)
    results = retriever.retrieve(
        "results", top_k=5, chunk_filter=lambda c: c.doc_id == "lab_cbc_001"
    )
    assert all(c.doc_id == "lab_cbc_001" for c, _ in results)


def test_enforce_citations_accepts_well_cited_answer():
    chunks = _all_chunks()[:2]
    answer = "Your white blood cell count is elevated. [1] This can happen with infection. [2]"
    grounded, final = enforce_citations("q", answer, chunks)
    assert grounded is True
    assert final == answer


def test_enforce_citations_rejects_uncited_answer():
    chunks = _all_chunks()[:2]
    answer = "Your results look fine and there is nothing to worry about at all."
    grounded, final = enforce_citations("q", answer, chunks)
    assert grounded is False
    assert "[1]" in final  # fell back to the extractive, cited answer
