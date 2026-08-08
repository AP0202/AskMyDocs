"""
Prometheus instrumentation for Ask My Docs.

This module defines every metric the app exposes at GET /metrics (scraped
by Prometheus, visualized in Grafana -- see monitoring/) and small helper
functions used by rag.py / llm.py / main.py to record them, so the
instrumentation call-sites stay one line each.

Metrics exposed:
  askmydocs_ask_request_duration_seconds   (Histogram)  end-to-end /api/ask latency
  askmydocs_pipeline_stage_duration_seconds(Histogram)  latency per RAG stage
  askmydocs_requests_total                 (Counter)    requests by grounded/backend
  askmydocs_errors_total                   (Counter)    failed requests by endpoint
  askmydocs_llm_tokens_total               (Counter)    prompt/completion tokens by model
  askmydocs_llm_cost_usd_total             (Counter)    estimated USD cost by model
  askmydocs_feedback_total                 (Counter)    thumbs up/down by value
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
    REGISTRY,
)
import os

# Standard registry is fine here: the app runs as a single process
# (uvicorn without --workers > 1). If you scale to multiple workers, switch
# to the multiprocess collector -- see Prometheus client docs.
_registry = REGISTRY

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20)

REQUEST_LATENCY = Histogram(
    "askmydocs_ask_request_duration_seconds",
    "End-to-end time to answer a question via /api/ask",
    buckets=LATENCY_BUCKETS,
)

STAGE_LATENCY = Histogram(
    "askmydocs_pipeline_stage_duration_seconds",
    "Time spent in each RAG pipeline stage",
    ["stage"],  # hybrid_retrieval | rerank | llm_generate
    buckets=LATENCY_BUCKETS,
)

REQUESTS_TOTAL = Counter(
    "askmydocs_requests_total",
    "Total number of /api/ask requests",
    ["grounded", "llm_backend"],  # grounded: true|false, llm_backend: openai|extractive
)

ERRORS_TOTAL = Counter(
    "askmydocs_errors_total",
    "Total number of failed requests",
    ["endpoint"],
)

TOKENS_TOTAL = Counter(
    "askmydocs_llm_tokens_total",
    "Total LLM tokens consumed",
    ["type", "model"],  # type: prompt|completion
)

COST_TOTAL = Counter(
    "askmydocs_llm_cost_usd_total",
    "Estimated cumulative USD cost of LLM calls (see MODEL_PRICING in config.py)",
    ["model"],
)

FEEDBACK_TOTAL = Counter(
    "askmydocs_feedback_total",
    "User feedback submitted on answers",
    ["value"],  # positive | negative | neutral
)


def record_stage(stage: str, seconds: float) -> None:
    STAGE_LATENCY.labels(stage=stage).observe(seconds)


def record_request(grounded: bool, llm_backend: str) -> None:
    REQUESTS_TOTAL.labels(grounded=str(grounded).lower(), llm_backend=llm_backend).inc()


def record_error(endpoint: str) -> None:
    ERRORS_TOTAL.labels(endpoint=endpoint).inc()


def record_llm_usage(model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    if prompt_tokens:
        TOKENS_TOTAL.labels(type="prompt", model=model).inc(prompt_tokens)
    if completion_tokens:
        TOKENS_TOTAL.labels(type="completion", model=model).inc(completion_tokens)
    if cost_usd:
        COST_TOTAL.labels(model=model).inc(cost_usd)


def record_feedback(value: int) -> None:
    label = "positive" if value > 0 else ("negative" if value < 0 else "neutral")
    FEEDBACK_TOTAL.labels(value=label).inc()


def render_metrics() -> bytes:
    """Return the current metrics snapshot in Prometheus text exposition format."""
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(_registry)


METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST
