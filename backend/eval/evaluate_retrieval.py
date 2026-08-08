#!/usr/bin/env python
"""
Retrieval evaluation: Hit Rate@k and Mean Reciprocal Rank (MRR) of the
hybrid + reranked retriever against a small hand-labeled ground truth set.

This is the script CI runs to *gate* merges: if retrieval quality regresses
below the configured thresholds, the script exits non-zero and the pipeline
fails, the same pattern the reference fitness-assistant project uses
notebook-side, just made CI-executable.

Usage:
    cd backend
    python eval/evaluate_retrieval.py
    python eval/evaluate_retrieval.py --min-hit-rate 0.8 --min-mrr 0.6
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from app.rag import RagPipeline  # noqa: E402
from app.config import FUSION_TOP_K  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


def load_ground_truth():
    return json.loads(GROUND_TRUTH_PATH.read_text())


def evaluate(pipeline: RagPipeline, ground_truth, top_k: int = FUSION_TOP_K):
    hits = 0
    reciprocal_ranks = []
    rows = []

    for item in ground_truth:
        question = item["question"]
        expected_doc_id = item["expected_doc_id"]

        results = pipeline.retrieve(question, final_top_k=top_k)
        doc_ids_in_order = [chunk.doc_id for chunk, _ in results]

        rank = None
        for i, doc_id in enumerate(doc_ids_in_order, start=1):
            if doc_id == expected_doc_id:
                rank = i
                break

        hit = rank is not None
        rr = 1.0 / rank if hit else 0.0
        hits += int(hit)
        reciprocal_ranks.append(rr)

        rows.append(
            {
                "question": question,
                "expected_doc_id": expected_doc_id,
                "retrieved_doc_ids": doc_ids_in_order,
                "hit": hit,
                "rank": rank,
            }
        )

    n = len(ground_truth)
    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    return hit_rate, mrr, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval quality.")
    parser.add_argument("--min-hit-rate", type=float, default=0.80)
    parser.add_argument("--min-mrr", type=float, default=0.60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    pipeline = RagPipeline()

    print(
        f"Indexed {len(pipeline.chunks)} chunks from {len(pipeline.documents)} documents "
        f"(vector_backend={pipeline.retriever.vector.backend_name}, "
        f"rerank_backend={pipeline.reranker.backend_name})"
    )

    hit_rate, mrr, rows = evaluate(pipeline, ground_truth)

    print(f"\nRetrieval evaluation over {len(ground_truth)} questions")
    print(f"  Hit Rate@{FUSION_TOP_K}: {hit_rate:.2%}")
    print(f"  MRR:        {mrr:.3f}")

    if args.verbose:
        print("\nPer-question detail:")
        for row in rows:
            status = "HIT " if row["hit"] else "MISS"
            print(
                f"  [{status}] rank={row['rank']}  '{row['question']}' "
                f"-> expected={row['expected_doc_id']} retrieved={row['retrieved_doc_ids']}"
            )

    passed = hit_rate >= args.min_hit_rate and mrr >= args.min_mrr
    print(
        f"\nThresholds: hit_rate >= {args.min_hit_rate:.0%}, mrr >= {args.min_mrr:.2f} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
