#!/usr/bin/env python
"""
Generate sample traffic against a running Ask My Docs instance so the
Grafana dashboard has something to show immediately, instead of empty
panels until real usage arrives.

Usage:
    python scripts/generate_demo_traffic.py
    python scripts/generate_demo_traffic.py --base-url http://localhost:8000 --requests 60 --delay 0.5
"""

import argparse
import random
import time

import requests

QUESTIONS = [
    "Why is my white blood cell count flagged?",
    "Is my hemoglobin normal?",
    "What does my platelet count result mean?",
    "What does my LDL cholesterol result mean?",
    "Are my triglycerides high?",
    "Why was I admitted to the hospital?",
    "What medications do I need to take after discharge?",
    "What warning signs mean I should go back to the ER?",
    "What did my chest X-ray show?",
    "Was there fluid around my lung?",
    "How should I take my metformin?",
    "What are the serious side effects I should watch for?",
    "What does this document say?",  # deliberately vague, exercises the fallback path
]


def main():
    parser = argparse.ArgumentParser(description="Generate demo traffic for the Grafana dashboard.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=40, help="number of /api/ask calls to make")
    parser.add_argument("--delay", type=float, default=0.75, help="seconds between requests")
    args = parser.parse_args()

    ok, failed = 0, 0
    for i in range(args.requests):
        question = random.choice(QUESTIONS)
        try:
            res = requests.post(
                f"{args.base_url}/api/ask", json={"question": question}, timeout=30
            )
            res.raise_for_status()
            body = res.json()
            print(
                f"[{i + 1}/{args.requests}] grounded={body['grounded']} "
                f"citations={len(body['citations'])}  '{question}'"
            )
            ok += 1
            # occasionally send feedback too, so the feedback panel has data
            if random.random() < 0.4:
                requests.post(
                    f"{args.base_url}/api/feedback",
                    json={
                        "conversation_id": body["conversation_id"],
                        "feedback": random.choice([1, 1, 1, -1]),
                    },
                    timeout=10,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{args.requests}] request failed: {exc}")
            failed += 1
        time.sleep(args.delay)

    print(f"\nDone. {ok} succeeded, {failed} failed.")
    print("Open Grafana at http://localhost:3000 (admin/admin) to see the dashboard.")


if __name__ == "__main__":
    main()
