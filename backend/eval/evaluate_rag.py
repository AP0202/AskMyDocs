#!/usr/bin/env python
"""
End-to-end RAG evaluation: for each ground-truth question, run the full
pipeline (retrieve -> rerank -> generate -> enforce citations) and check:

  1. groundedness  - did the citation-enforcement pass accept the answer,
                      or did it have to fall back to the safe extract?
  2. citation validity - do all citation markers in the answer point at
                      chunks that were actually retrieved (no hallucinated
                      sources)?
  3. keyword coverage  - does the answer mention at least one of the
                      expected clinical keywords for that question (a
                      lightweight relevance signal that doesn't require an
                      LLM judge / API key)?

Like evaluate_retrieval.py, this exits non-zero below threshold so it can
gate CI.

Usage:
    cd backend
    python eval/evaluate_rag.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from app.rag import RagPipeline  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
CITATION_RE = re.compile(r"\[(\d+)\]")


def load_ground_truth():
    return json.loads(GROUND_TRUTH_PATH.read_text())


def citations_are_valid(answer: str, n_citations: int) -> bool:
    markers = {int(m) for m in CITATION_RE.findall(answer)}
    if not markers:
        return n_citations == 0
    return all(1 <= m <= n_citations for m in markers)


def keyword_covered(answer: str, keywords) -> bool:
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def evaluate(pipeline: RagPipeline, ground_truth):
    rows = []
    for item in ground_truth:
        result = pipeline.answer(item["question"], document_id=None)
        answer = result["answer"]
        n_citations = len(result["citations"])

        rows.append(
            {
                "question": item["question"],
                "grounded": result["grounded"],
                "valid_citations": citations_are_valid(answer, n_citations),
                "keyword_covered": keyword_covered(answer, item["expected_keywords"]),
                "n_citations": n_citations,
                "answer_preview": answer[:160].replace("\n", " "),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end RAG quality.")
    parser.add_argument("--min-grounded-rate", type=float, default=0.75)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.70)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    pipeline = RagPipeline()

    rows = evaluate(pipeline, ground_truth)
    n = len(rows)

    grounded_rate = sum(r["grounded"] for r in rows) / n
    valid_citation_rate = sum(r["valid_citations"] for r in rows) / n
    keyword_rate = sum(r["keyword_covered"] for r in rows) / n

    print(f"RAG evaluation over {n} questions")
    print(f"  Grounded (passed citation enforcement): {grounded_rate:.2%}")
    print(f"  Valid citations (no hallucinated refs): {valid_citation_rate:.2%}")
    print(f"  Keyword coverage:                       {keyword_rate:.2%}")

    if args.verbose:
        print("\nPer-question detail:")
        for r in rows:
            print(
                f"  grounded={r['grounded']!s:5} valid_cites={r['valid_citations']!s:5} "
                f"kw={r['keyword_covered']!s:5} n_cites={r['n_citations']} "
                f"'{r['question']}' -> {r['answer_preview']}..."
            )

    passed = (
        grounded_rate >= args.min_grounded_rate
        and valid_citation_rate == 1.0
        and keyword_rate >= args.min_keyword_coverage
    )
    print(f"\n-> {'PASS' if passed else 'FAIL'}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
